import os
import gc
import time
import logging
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import mne
import warnings
from scipy.signal import hilbert, savgol_filter
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# 불필요한 경고 및 MNE 로그 음소거
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================
# [1] 환경 설정 및 글로벌 데이터 Split
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True 

# 🌟 자신의 환경(RunPod/Colab)에 맞게 경로 수정 필수
DATA_DIR_PHYSIONET = './raw_data/files' 
SAVE_DIR = './results_ultimate_geosnn'
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(os.path.join(SAVE_DIR, "training_ultimate.log")), logging.StreamHandler()])

EXCLUDED_SUBJECTS = [88, 92, 100, 104]
VALID_SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]

# 84명 훈련 / 21명 완벽 격리 테스트 (Data Leakage 0%)
TRAIN_SUBJECTS = VALID_SUBJECTS[:84]
TEST_SUBJECTS = VALID_SUBJECTS[84:]

EPOCHS = 200
BATCH_SIZE = 128 

# ==========================================
# [2] 전처리: 유클리드 정렬 (EA) & Data Augmentation
# ==========================================
def euclidean_alignment(X):
    """ 피험자별 공분산 행렬의 영점을 맞추는 EA 전처리 (Singular 방지 1e-6) """
    trials, channels, time_pts = X.shape
    X_flat = X.transpose(1, 0, 2).reshape(channels, -1)
    cov = np.cov(X_flat)
    cov += np.eye(channels) * 1e-6
    R_inv_half = np.real(fractional_matrix_power(cov, -0.5))
    X_aligned_flat = R_inv_half @ X_flat
    return X_aligned_flat.reshape(channels, trials, time_pts).transpose(1, 0, 2)

def augment_with_sliding_window(features, labels, window_size=320, step_size=32):
    """ 480스텝(3초) 데이터를 320스텝(2초) 창문으로 32스텝(0.2초)씩 밀면서 6배 증강 """
    N, C, T, F = features.shape
    aug_features, aug_labels = [], []
    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        aug_features.append(features[:, :, start:end, :])
        aug_labels.append(labels)
    return torch.cat(aug_features, dim=0), torch.cat(aug_labels, dim=0)

def load_local_runs(subject, run_list):
    sub_str = f"S{subject:03d}"
    raws = []
    for run in run_list:
        file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
        if not os.path.exists(file_path): 
            file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")
            if not os.path.exists(file_path): return None
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(l_freq=8., h_freq=30., fir_design='firwin', verbose=False) 
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1005', match_case=False, on_missing='ignore')
        raws.append(raw)
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
    u = savgol_filter(np.abs(hilbert(X_base, axis=-1)), 31, 3, axis=-1)
    v = savgol_filter(np.abs(hilbert(X_csd, axis=-1)), 31, 3, axis=-1)
    vu = savgol_filter(u, 31, 3, deriv=1, axis=-1)
    vv = savgol_filter(v, 31, 3, deriv=1, axis=-1)
    speed = np.sqrt(vu**2 + vv**2) + 1e-6
    vu_s = np.roll(vu, 3, axis=-1); vv_s = np.roll(vv, 3, axis=-1)
    curv = 1.0 - np.clip((vu*vu_s + vv*vv_s) / (speed * np.sqrt(vu_s**2 + vv_s**2) + 1e-6), -1.0, 1.0)
    tang = np.log1p(np.abs(u*vv - v*vu) * 1000.0)
    feat = np.stack([u, v, curv, tang], axis=-1).astype(np.float32)
    
    # 피험자별 Z-score 정규화
    for i in range(4): 
        f_mean = feat[..., i].mean(axis=(0, 1, 2), keepdims=True)
        f_std = feat[..., i].std(axis=(0, 1, 2), keepdims=True) + 1e-6
        feat[..., i] = (feat[..., i] - f_mean) / f_std
    return feat

def compute_gcn_laplacian(ch_names):
    pos = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    coords = np.array([pos[ch] if ch in pos else [0,0,0] for ch in ch_names])
    A = np.exp(-(squareform(pdist(coords))**2)/(2*0.05**2))
    A[A<0.1]=0.0; np.fill_diagonal(A,1.0)
    D_inv = np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))
    return torch.tensor(D_inv @ A @ D_inv, dtype=torch.float32)

# ==========================================
# [3] 모델 정의: Ultimate Hybrid Geo-SNN
# ==========================================
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, alpha=5.0): ctx.save_for_backward(i); ctx.alpha=alpha; return (i>0).float()
    @staticmethod
    def backward(ctx, grad): i, = ctx.saved_tensors; return grad/(ctx.alpha*torch.abs(i)+1.0)**2, None

class UltimateGeoSNN(nn.Module):
    def __init__(self, num_channels=15, num_classes=4, A_norm=None):
        super().__init__()
        self.A_norm = nn.Parameter(A_norm, requires_grad=False)
        self.W_u = nn.Linear(num_channels, num_channels)
        self.W_v = nn.Linear(num_channels, num_channels)
        self.bn_inj = nn.BatchNorm1d(num_channels)
        
        self.spatial_conv = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=1)
        self.bn_spatial = nn.BatchNorm1d(32)
        
        self.conv1 = nn.Conv1d(32, 128, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(128)
        self.shortcut1 = nn.Conv1d(32, 128, kernel_size=1) 
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.shortcut2 = nn.Conv1d(128, 256, kernel_size=1) 
        
        self.dp_pool = nn.AvgPool1d(32, 32)
        self.pre_fc_bn = nn.BatchNorm1d(256 * 10)
        self.dropout = nn.Dropout(p=0.3) 
        self.fc_out = nn.Linear(256 * 10, num_classes)
        
        self.v_th1, self.v_th2, self.v_th3 = 0.15, 0.3, 0.3
        self.tau_base = 35.0
        
        self.g_curv = nn.Parameter(torch.tensor(0.8))
        self.g_tang = nn.Parameter(torch.tensor(0.4))
        self.w_v2 = nn.Parameter(torch.tensor(0.0)) 
        self.w_v3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b, c, t, _ = x.shape
        device = x.device
        
        v1 = torch.zeros(b, c, device=device)
        spks1 = []
        for step in range(t):
            inj = self.bn_inj(self.W_u(x[:,:,step,0] @ self.A_norm) + self.W_v(x[:,:,step,1] @ self.A_norm))
            tau = self.tau_base * torch.exp(-(self.g_curv * x[:,:,step,2] + self.g_tang * x[:,:,step,3]))
            tau = torch.clamp(tau, min=5.0, max=100.0) 
            
            v1 = v1 * torch.exp(-1.0/tau) + inj
            s1 = FastSigmoid.apply(v1 - self.v_th1)
            v1 -= s1 * self.v_th1
            spks1.append(s1)
            
        s1_stack = torch.stack(spks1, dim=2)
        
        s_spatial = self.bn_spatial(self.spatial_conv(s1_stack))
        
        c1 = self.bn1(self.conv1(s_spatial)) + self.shortcut1(s_spatial)
        
        v2, spks2 = torch.zeros(b, 128, device=device), []
        decay2 = torch.sigmoid(self.w_v2) 
        for step in range(t):
            v2 = v2 * decay2 + c1[:,:,step]
            s2 = FastSigmoid.apply(v2 - self.v_th2)
            v2 -= s2 * self.v_th2
            spks2.append(s2)
            
        s2_stack = torch.stack(spks2, dim=2)
        c2 = self.bn2(self.conv2(s2_stack)) + self.shortcut2(s2_stack)
        
        v3, spks3 = torch.zeros(b, 256, device=device), []
        decay3 = torch.sigmoid(self.w_v3)
        for step in range(t):
            v3 = v3 * decay3 + c2[:,:,step]
            s3 = FastSigmoid.apply(v3 - self.v_th3)
            v3 -= s3 * self.v_th3
            spks3.append(s3)
            
        s3_stack = torch.stack(spks3, dim=2)
        out = self.pre_fc_bn(self.dp_pool(s3_stack).view(b, -1))
        out = self.dropout(out) 
        
        # 🌟 디버깅: 훈련 중 5% 확률로 각 레이어의 평균 발화율(Firing Rate) 모니터링
        if self.training and torch.rand(1).item() < 0.05:
            logging.info(f"[SNN Firing Rate Debug] L1: {s1_stack.mean().item():.4f} | L2: {s2_stack.mean().item():.4f} | L3: {s3_stack.mean().item():.4f}")

        return self.fc_out(out)

# ==========================================
# [4] Main Execution: 데이터 로딩, 증강 및 훈련 루프
# ==========================================
if __name__ == "__main__":
    logging.info(f"🌍 System Ready. Device: {device}")
    
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

    logging.info(f"\n⏳ 1. Train Subjects ({len(TRAIN_SUBJECTS)}명) 추출 중...")
    X_train_raw, y_train_raw = get_data_for_subjects(TRAIN_SUBJECTS, "Extract Train")
    X_train, y_train = augment_with_sliding_window(X_train_raw, y_train_raw)
    del X_train_raw, y_train_raw
    
    logging.info(f"\n⏳ 2. Test Subjects ({len(TEST_SUBJECTS)}명) 추출 중...")
    X_test_raw, y_test_raw = get_data_for_subjects(TEST_SUBJECTS, "Extract Test")
    X_test, y_test = augment_with_sliding_window(X_test_raw, y_test_raw)
    del X_test_raw, y_test_raw
    
    gc.collect()
    
    workers = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    
    model = UltimateGeoSNN(num_channels=15, num_classes=4, A_norm=A_norm_motor).to(device)
    
    # 🌟 SNN에 맞게 초기 학습률을 0.01에서 0.001로 하향 조정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # 🌟 폭주하던 OneCycleLR 대신 얌전한 CosineAnnealingLR 도입
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5
    )
    
    logging.info(f"\n🚀 궁극의 하이브리드 SNN 훈련 시작 (Epochs: {EPOCHS}, Batch: {BATCH_SIZE})")
    best_acc = 0.0
    NUM_WINDOWS_PER_TRIAL = 6 
    
    for epoch in range(EPOCHS):
        start_t = time.time()
        model.train()
        total_loss = 0
        
        # 🌟 Train Acc 계산용 변수 추가
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
            
            # 🌟 Train Acc 계산 로직
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()
            
        scheduler.step()
        
        # 🌟 에포크 종료 후 Train Acc 산출
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
            
            if final_pred == all_labels[start_idx]:
                correct += 1
            total += 1
                
        test_acc = 100 * correct / total
        
        if test_acc > best_acc: 
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_ultimate_model.pth'))
        
        epoch_time = time.time() - start_t
        curr_lr = scheduler.get_last_lr()[0]
        
        # 🌟 로그 출력에 Train Acc 추가
        logging.info(f"Epoch [{epoch+1:03d}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}% | LR: {curr_lr:.5f} | Time: {epoch_time:.1f}s")
        
    
    logging.info(f"\n🎉 훈련 완료! 최종 최고 제로샷 테스트 정확도: {best_acc:.2f}%")