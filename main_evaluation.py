import os
import gc
import time
import logging
import multiprocessing
import numpy as np
import torch
import torch._utils  # Bypass Kaggle Dynamo bug
import torch.nn as nn
import mne
import warnings
from scipy.signal import hilbert, savgol_filter
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================
# [1] Environment Setup & Global Split
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True 

DATA_DIR_PHYSIONET = './raw_data/files' 
SAVE_DIR = './results_ultimate_geosnn'
os.makedirs(SAVE_DIR, exist_ok=True)

# English Only, No Emojis
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(os.path.join(SAVE_DIR, "training_ultimate.log")), logging.StreamHandler()])

EXCLUDED_SUBJECTS = [88, 92, 100, 104]
VALID_SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]

TRAIN_SUBJECTS = VALID_SUBJECTS[:84]
TEST_SUBJECTS = VALID_SUBJECTS[84:]

EPOCHS = 200
BATCH_SIZE = 128 

# ==========================================
# [2] Preprocessing & Data Augmentation
# ==========================================
def euclidean_alignment(X):
    trials, channels, time_pts = X.shape
    X_flat = X.transpose(1, 0, 2).reshape(channels, -1)
    cov = np.cov(X_flat)
    cov += np.eye(channels) * 1e-6
    R_inv_half = np.real(fractional_matrix_power(cov, -0.5))
    X_aligned_flat = R_inv_half @ X_flat
    return X_aligned_flat.reshape(channels, trials, time_pts).transpose(1, 0, 2)

def augment_with_sliding_window(features, labels, window_size=320, step_size=32):
    N, C, T, F = features.shape
    aug_features, aug_labels = [], []
    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        aug_features.append(features[:, :, start:end, :])
        aug_labels.append(labels)
    return torch.cat(aug_features, dim=0), torch.cat(aug_labels, dim=0)

def load_local_runs(subject, run_list):
    raws = []
    for run in run_list:
        try:
            raw_path = mne.datasets.eegbci.load_data(subject, run, update_path=False, verbose=False)[0]
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
            raw.filter(l_freq=8., h_freq=30., fir_design='firwin', verbose=False) 
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage('standard_1005', match_case=False, on_missing='ignore')
            raws.append(raw)
        except Exception as e:
            return None
    return mne.concatenate_raws(raws) if raws else None

def load_and_align_subject_data(subject):
    try:
        epochs_base_list, epochs_csd_list, labels_list = [], [], []
        raw_lr = load_local_runs(subject, [4, 8, 12])
        if raw_lr:
            raw_lr_csd = mne.preprocessing.compute_current_source_density(raw_lr.copy())
            ev_raw, ev_id = mne.events_from_annotations(raw_lr, verbose=False)
            t1, t2 = ev_id.get('T1'), ev_id.get('T2')
            ev = ev_raw[(ev_raw[:, 2] == t1) | (ev_raw[:, 2] == t2)].copy()
            ev[ev[:, 2] == t1, 2], ev[ev[:, 2] == t2, 2] = 0, 1
            ep_base = mne.Epochs(raw_lr, ev, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            ep_csd = mne.Epochs(raw_lr_csd, ev, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            epochs_base_list.append(ep_base.get_data()[:, :, :480])
            epochs_csd_list.append(ep_csd.get_data()[:, :, :480])
            labels_list.append(ep_base.events[:, -1])
            
        raw_bf = load_local_runs(subject, [6, 10, 14])
        if raw_bf:
            raw_bf_csd = mne.preprocessing.compute_current_source_density(raw_bf.copy())
            ev_raw, ev_id = mne.events_from_annotations(raw_bf, verbose=False)
            t1_b, t2_f = ev_id.get('T1'), ev_id.get('T2')
            ev = ev_raw[(ev_raw[:, 2] == t1_b) | (ev_raw[:, 2] == t2_f)].copy()
            ev[ev[:, 2] == t1_b, 2], ev[ev[:, 2] == t2_f, 2] = 2, 3
            ep_base = mne.Epochs(raw_bf, ev, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            ep_csd = mne.Epochs(raw_bf_csd, ev, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            epochs_base_list.append(ep_base.get_data()[:, :, :480])
            epochs_csd_list.append(ep_csd.get_data()[:, :, :480])
            labels_list.append(ep_base.events[:, -1])
            ch_names = ep_base.ch_names
            
        if not epochs_base_list: return None, None, None, None
        
        X_base = np.concatenate(epochs_base_list, axis=0)
        X_csd = np.concatenate(epochs_csd_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        
        X_base_aligned = euclidean_alignment(X_base)
        X_csd_aligned = euclidean_alignment(X_csd)
        
        return X_base_aligned, X_csd_aligned, y, ch_names
    except Exception as e: 
        return None, None, None, None

def extract_4d_features(X_base, X_csd):
    win_size = 61
    poly = 3
    gap = 15
    
    u = savgol_filter(np.abs(hilbert(X_base, axis=-1)), win_size, poly, axis=-1)
    v = savgol_filter(np.abs(hilbert(X_csd, axis=-1)), win_size, poly, axis=-1)
    
    vu = savgol_filter(u, win_size, poly, deriv=1, axis=-1)
    vv = savgol_filter(v, win_size, poly, deriv=1, axis=-1)
    
    speed = np.sqrt(vu**2 + vv**2) + 1e-6
    vu_s = np.roll(vu, gap, axis=-1)
    vv_s = np.roll(vv, gap, axis=-1)
    
    curv = 1.0 - np.clip((vu*vu_s + vv*vv_s) / (speed * np.sqrt(vu_s**2 + vv_s**2) + 1e-6), -1.0, 1.0)
    areal_velocity = np.log1p(np.abs(u*vv - v*vu) * 1000.0)
    
    feat = np.stack([u, v, curv, areal_velocity], axis=-1).astype(np.float32)
    
    # Edge effect cropping (480 -> 416)
    pad = 32
    feat_cropped = feat[:, :, pad:-pad, :]
    
    for i in range(4): 
        f_mean = feat_cropped[..., i].mean(axis=(0, 1, 2), keepdims=True)
        f_std = feat_cropped[..., i].std(axis=(0, 1, 2), keepdims=True) + 1e-6
        feat_cropped[..., i] = (feat_cropped[..., i] - f_mean) / f_std
        
    return feat_cropped

def compute_gcn_laplacian(ch_names):
    pos = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    coords = np.array([pos[ch] if ch in pos else [0,0,0] for ch in ch_names])
    A = np.exp(-(squareform(pdist(coords))**2)/(2*0.05**2))
    A[A<0.1]=0.0; np.fill_diagonal(A,1.0)
    D_inv = np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))
    return torch.tensor(D_inv @ A @ D_inv, dtype=torch.float32)

# ==========================================
# [3] Model Definition: Ultimate Hybrid Geo-SNN
# ==========================================
class DPPooling1d(nn.Module):
    def __init__(self, window_size=32, n_dp=16):
        super().__init__()
        self.window_size = window_size
        self.n_dp = n_dp 

    def forward(self, x):
        b, c, t = x.shape
        num_windows = t // self.window_size
        x_split = x.view(b, c, num_windows, self.window_size)
        start_mean = x_split[:, :, :, :self.n_dp].mean(dim=-1)
        end_mean = x_split[:, :, :, -self.n_dp:].mean(dim=-1)
        return end_mean - start_mean

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, alpha=2.0): 
        ctx.save_for_backward(i)
        ctx.alpha = alpha
        return (i > 0).float()
    @staticmethod
    def backward(ctx, grad): 
        i, = ctx.saved_tensors
        return grad / (ctx.alpha * torch.abs(i) + 1.0)**2, None

class UltimateGeoSNN(nn.Module):
    def __init__(self, num_channels=15, num_classes=4, A_norm=None):
        super().__init__()
        self.A_norm = nn.Parameter(A_norm, requires_grad=False)
        
        self.W_u = nn.Linear(num_channels, num_channels)
        self.W_v = nn.Linear(num_channels, num_channels)
        self.W_c = nn.Linear(num_channels, num_channels)
        self.W_a = nn.Linear(num_channels, num_channels)
        
        self.bn_inj = nn.BatchNorm1d(num_channels)
        
        # Reduced channel capacity
        self.spatial_conv = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=1)
        self.bn_spatial = nn.BatchNorm1d(16)
        
        self.conv1 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.shortcut1 = nn.Conv1d(16, 32, kernel_size=1) 
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(32, 64, kernel_size=1) 
        
        self.dp_pool = DPPooling1d(window_size=32, n_dp=16)
        
        self.pre_fc_bn = nn.BatchNorm1d(64 * 10)
        self.dropout = nn.Dropout(p=0.5) 
        self.fc_out = nn.Linear(64 * 10, num_classes)
        
        self.v_th1, self.v_th2, self.v_th3 = 3.0, 2.0, 1.5
        self.tau_base = 35.0
        
        self.w_v2 = nn.Parameter(torch.tensor(0.0)) 
        self.w_v3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b, c, t, _ = x.shape
        device = x.device
        
        if self.training:
            x = x + torch.randn_like(x) * 0.1
            
        v1 = torch.zeros(b, c, device=device)
        spks1 = []
        for step in range(t):
            inj = self.bn_inj(
                self.W_u(x[:,:,step,0] @ self.A_norm) + 
                self.W_v(x[:,:,step,1] @ self.A_norm) +
                self.W_c(x[:,:,step,2] @ self.A_norm) +
                self.W_a(x[:,:,step,3] @ self.A_norm)
            )
            
            v1 = v1 * torch.exp(torch.tensor(-1.0 / self.tau_base, device=device)) + inj
            s1 = FastSigmoid.apply(v1 - self.v_th1)
            v1 -= s1 * self.v_th1
            spks1.append(s1)
            
        s1_stack = torch.stack(spks1, dim=2)
        s_spatial = self.bn_spatial(self.spatial_conv(s1_stack))
        c1 = self.bn1(self.conv1(s_spatial)) + self.shortcut1(s_spatial)
        
        v2, spks2 = torch.zeros(b, 32, device=device), []
        decay2 = torch.sigmoid(self.w_v2) 
        for step in range(t):
            v2 = v2 * decay2 + c1[:,:,step]
            s2 = FastSigmoid.apply(v2 - self.v_th2)
            v2 -= s2 * self.v_th2
            spks2.append(s2)
            
        s2_stack = torch.stack(spks2, dim=2)
        c2 = self.bn2(self.conv2(s2_stack)) + self.shortcut2(s2_stack)
        
        v3, spks3 = torch.zeros(b, 64, device=device), []
        decay3 = torch.sigmoid(self.w_v3)
        for step in range(t):
            v3 = v3 * decay3 + c2[:,:,step]
            s3 = FastSigmoid.apply(v3 - self.v_th3)
            v3 -= s3 * self.v_th3
            spks3.append(s3)
            
        s3_stack = torch.stack(spks3, dim=2)
        
        pooled_out = self.dp_pool(s3_stack)
        out = self.pre_fc_bn(pooled_out.view(b, -1))
        out = self.dropout(out) 
        
        return self.fc_out(out)

# ==========================================
# [4] Main Execution
# ==========================================
if __name__ == "__main__":
    logging.info(f"System Ready. Device: {device}")
    
    _, _, _, sample_ch = load_and_align_subject_data(1)
    target_motor_chs = ['FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4']
    motor_indices = [i for i, ch in enumerate(sample_ch) if ch.upper() in target_motor_chs]
    A_norm_motor = compute_gcn_laplacian([sample_ch[i] for i in motor_indices]).to(device)
    
    def get_data_for_subjects(subject_list, desc):
        X_list, y_list = [], []
        for sub in tqdm(subject_list, desc=desc):
            X_b, X_c, y, _ = load_and_align_subject_data(sub)
            if X_b is not None:
                feat = extract_4d_features(X_b, X_c)
                X_list.append(torch.tensor(feat[:, motor_indices, :, :], dtype=torch.float32))
                y_list.append(torch.tensor(y, dtype=torch.long))
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

    logging.info(f"Extracting {len(TRAIN_SUBJECTS)} Train Subjects...")
    X_train_raw, y_train_raw = get_data_for_subjects(TRAIN_SUBJECTS, "Train")
    X_train, y_train = augment_with_sliding_window(X_train_raw, y_train_raw)
    del X_train_raw, y_train_raw
    
    logging.info(f"Extracting {len(TEST_SUBJECTS)} Test Subjects...")
    X_test_raw, y_test_raw = get_data_for_subjects(TEST_SUBJECTS, "Test")
    X_test, y_test = augment_with_sliding_window(X_test_raw, y_test_raw)
    del X_test_raw, y_test_raw
    
    gc.collect()
    
    workers = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    
    model = UltimateGeoSNN(num_channels=15, num_classes=4, A_norm=A_norm_motor).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, foreach=False)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    logging.info(f"Starting Phase 1: Global Training (Epochs: {EPOCHS}, Batch: {BATCH_SIZE})")
    best_acc = 0.0
    NUM_WINDOWS_PER_TRIAL = 4 
    
    for epoch in range(EPOCHS):
        start_t = time.time()
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(xb)
                loss = criterion(outputs, yb)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()
            
        scheduler.step()
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        correct, total = 0, 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                with autocast():
                    outputs = model(xb)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        num_original_trials = len(all_preds) // NUM_WINDOWS_PER_TRIAL
        
        for i in range(num_original_trials):
            start_idx = i * NUM_WINDOWS_PER_TRIAL
            end_idx = start_idx + NUM_WINDOWS_PER_TRIAL
            window_preds = all_preds[start_idx:end_idx]
            final_pred = np.bincount(window_preds).argmax() 
            
            if final_pred == all_labels[start_idx]: correct += 1
            total += 1
                
        test_acc = 100 * correct / total
        
        if test_acc > best_acc: 
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_ultimate_model.pth'))
        
        epoch_time = time.time() - start_t
        curr_lr = scheduler.get_last_lr()[0]
        
        logging.info(f"Epoch [{epoch+1:03d}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}% | LR: {curr_lr:.5f} | Time: {epoch_time:.1f}s")

    logging.info(f"Phase 1 Complete. Best Zero-shot Test Accuracy: {best_acc:.2f}%")

    # ==========================================
    # [5] Phase 2: Subject-Specific Transfer Learning (SSTL)
    # ==========================================
    logging.info("Starting Phase 2: SSTL Calibration")

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_ultimate_model.pth')))

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc_out.parameters():
        param.requires_grad = True

    optimizer_ft = torch.optim.Adam(model.fc_out.parameters(), lr=0.005, weight_decay=1e-3, foreach=False)
    criterion_ft = nn.CrossEntropyLoss()

    dataset_size = len(X_test)
    calib_size = int(dataset_size * 0.2)
    
    np.random.seed(42)
    indices = np.random.permutation(dataset_size)
    calib_indices, real_test_indices = indices[:calib_size], indices[calib_size:]

    calib_loader = DataLoader(TensorDataset(X_test[calib_indices], y_test[calib_indices]), batch_size=BATCH_SIZE, shuffle=True)
    real_test_loader = DataLoader(TensorDataset(X_test[real_test_indices], y_test[real_test_indices]), batch_size=BATCH_SIZE, shuffle=False)

    FT_EPOCHS = 15 
    best_ft_acc = 0.0

    for epoch in range(FT_EPOCHS):
        model.train()
        for xb, yb in calib_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer_ft.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(xb)
                loss = criterion_ft(outputs, yb)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_ft)
            scaler.update()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in real_test_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast():
                    outputs = model(xb)
                _, predicted = outputs.max(1)
                total += yb.size(0)
                correct += predicted.eq(yb).sum().item()
        
        ft_acc = 100. * correct / total
        if ft_acc > best_ft_acc:
            best_ft_acc = ft_acc
            
        logging.info(f"[SSTL Fine-tuning] Epoch [{epoch+1:02d}/{FT_EPOCHS}] | Target Test Acc: {ft_acc:.2f}% (Best: {best_ft_acc:.2f}%)")
        
    logging.info(f"Phase 2 Complete. Best SSTL Target Accuracy: {best_ft_acc:.2f}%")
    
    os.system("sleep 60 && runpodctl stop pod $RUNPOD_POD_ID")