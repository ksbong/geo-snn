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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.signal import hilbert, savgol_filter
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# ==========================================
# [0] Configuration & Style Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True 

DATA_DIR_PHYSIONET = './raw_data/files' # RunPod Local Path
SAVE_DIR = './results_deep_geosnn'
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(os.path.join(SAVE_DIR, "training.log")), logging.StreamHandler()])

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16

# ==========================================
# [1] Data Loader (Base + CSD)
# ==========================================
def load_local_runs(subject, run_list):
    sub_str = f"S{subject:03d}"
    raws = []
    for run in run_list:
        file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
        if not os.path.exists(file_path): return None
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(l_freq=8., h_freq=30., fir_design='firwin', verbose=False) 
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1005', match_case=False, on_missing='ignore')
        raws.append(raw)
    if not raws: return None
    return mne.concatenate_raws(raws)

def load_physionet_dual_runpod(subject):
    try:
        epochs_base_list, epochs_csd_list, labels_list = [], [], []
        
        # Left/Right Hand
        raw_lr = load_local_runs(subject, [4, 8, 12])
        if raw_lr is not None:
            raw_lr_csd = mne.preprocessing.compute_current_source_density(raw_lr.copy())
            ev_lr_raw, ev_id_lr = mne.events_from_annotations(raw_lr, verbose=False)
            t1, t2 = ev_id_lr.get('T1'), ev_id_lr.get('T2')
            ev_lr = ev_lr_raw[(ev_lr_raw[:, 2] == t1) | (ev_lr_raw[:, 2] == t2)].copy()
            ev_lr[ev_lr[:, 2] == t1, 2] = 0 
            ev_lr[ev_lr[:, 2] == t2, 2] = 1 
            
            ep_base = mne.Epochs(raw_lr, ev_lr, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            ep_csd = mne.Epochs(raw_lr_csd, ev_lr, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            epochs_base_list.append(ep_base.get_data()[:, :, :480])
            epochs_csd_list.append(ep_csd.get_data()[:, :, :480])
            labels_list.append(ep_base.events[:, -1])

        # Both Hands/Feet
        raw_bf = load_local_runs(subject, [6, 10, 14])
        if raw_bf is not None:
            raw_bf_csd = mne.preprocessing.compute_current_source_density(raw_bf.copy())
            ev_bf_raw, ev_id_bf = mne.events_from_annotations(raw_bf, verbose=False)
            t1_b, t2_f = ev_id_bf.get('T1'), ev_id_bf.get('T2')
            
            ev_bf = ev_bf_raw[(ev_bf_raw[:, 2] == t1_b) | (ev_bf_raw[:, 2] == t2_f)].copy()
            ev_bf[ev_bf[:, 2] == t1_b, 2] = 2 
            ev_bf[ev_bf[:, 2] == t2_f, 2] = 3 
            
            ep_base = mne.Epochs(raw_bf, ev_bf, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            ep_csd = mne.Epochs(raw_bf_csd, ev_bf, tmin=0., tmax=2.99, baseline=None, preload=True, verbose=False)
            epochs_base_list.append(ep_base.get_data()[:, :, :480])
            epochs_csd_list.append(ep_csd.get_data()[:, :, :480])
            labels_list.append(ep_base.events[:, -1])
            ch_names = ep_base.ch_names
            
        if not epochs_base_list: return None, None, None, None
        
        X_base = np.concatenate(epochs_base_list, axis=0)
        X_csd = np.concatenate(epochs_csd_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        return X_base, X_csd, y, ch_names
    except Exception as e:
        logging.warning(f"Error loading Subject {subject}: {e}")
        return None, None, None, None

# ==========================================
# [2] 4D Feature Engine
# ==========================================
def extract_original_4d_features(X_base, X_csd):
    u_raw = np.abs(hilbert(X_base, axis=-1))
    v_raw = np.abs(hilbert(X_csd, axis=-1))
    
    u = savgol_filter(u_raw, 31, 3, axis=-1)
    v = savgol_filter(v_raw, 31, 3, axis=-1)
    
    vu = savgol_filter(u, 31, 3, deriv=1, axis=-1)
    vv = savgol_filter(v, 31, 3, deriv=1, axis=-1)
    speed = np.sqrt(vu**2 + vv**2) + 1e-6
    
    vu_shift = np.roll(vu, shift=3, axis=-1)
    vv_shift = np.roll(vv, shift=3, axis=-1)
    vu_shift[..., :3] = vu[..., :3]
    vv_shift[..., :3] = vv[..., :3]
    
    cos_theta = np.clip((vu*vu_shift + vv*vv_shift) / (speed * np.sqrt(vu_shift**2 + vv_shift**2) + 1e-6), -1.0, 1.0)
    curv = 1.0 - cos_theta
    tang = np.log1p(np.abs(u*vv - v*vu) * 1000.0)
    
    features = np.stack([u, v, curv, tang], axis=-1).astype(np.float32) 
    
    for i in range(4):
        f_min = features[..., i].min(axis=2, keepdims=True)
        f_max = features[..., i].max(axis=2, keepdims=True)
        features[..., i] = (features[..., i] - f_min) / (f_max - f_min + 1e-6)
        
    return features

def compute_gcn_laplacian(ch_names, sigma=0.05):
    pos_dict = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    coords = np.array([pos_dict[ch] if ch in pos_dict else [0,0,0] for ch in ch_names])
    dist_mat = squareform(pdist(coords, metric='euclidean'))
    A = np.exp(-(dist_mat**2) / (2 * sigma**2))
    A[A < 0.1] = 0.0 
    np.fill_diagonal(A, 1.0)
    D_inv_sqrt = np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))
    return torch.tensor(np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt), dtype=torch.float32)

# ==========================================
# [3] Deep Geo-Conv SNN Model
# ==========================================
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha=5.0): 
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (ctx.alpha * torch.abs(input) + 1.0)**2, None

class DeepGeoConvSNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=4, A_norm=None):
        super(DeepGeoConvSNN, self).__init__()
        self.A_norm = nn.Parameter(A_norm, requires_grad=False)
        self.W_u = nn.Linear(num_channels, num_channels, bias=True)
        self.W_v = nn.Linear(num_channels, num_channels, bias=True)
        self.bn_inj = nn.BatchNorm1d(num_channels)
        self.v_th1, self.tau_base, self.gamma_curv, self.gamma_tang = 0.15, 35.0, 0.8, 0.4
        
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(128)
        self.shortcut1 = nn.Conv1d(num_channels, 128, kernel_size=1)
        self.v_th2, self.tau2 = 0.3, 2.0
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.shortcut2 = nn.Conv1d(128, 256, kernel_size=1)
        self.v_th3, self.tau3 = 0.3, 2.0
        
        self.dp_pool = nn.AvgPool1d(kernel_size=32, stride=32)
        self.pre_fc_bn = nn.BatchNorm1d(256 * 15)
        self.fc_out = nn.Linear(256 * 15, num_classes)

    def forward(self, features, return_features=False):
        batch_size, channels, time_steps, _ = features.shape
        device = features.device
        
        v_mem1 = torch.zeros(batch_size, channels, device=device)
        spike1_record = []
        for t in range(time_steps):
            u_t, v_t = features[:, :, t, 0], features[:, :, t, 1]
            curv_t, tang_t = features[:, :, t, 2], features[:, :, t, 3]
            
            i_inj = self.bn_inj(self.W_u(torch.matmul(u_t, self.A_norm)) + self.W_v(torch.matmul(v_t, self.A_norm)))
            tau_t = self.tau_base * torch.exp(-(self.gamma_curv * curv_t + self.gamma_tang * tang_t))
            
            v_mem1 = v_mem1 * torch.exp(-1.0 / tau_t) + i_inj
            spike1 = FastSigmoid.apply(v_mem1 - self.v_th1)
            v_mem1 = v_mem1 - spike1 * self.v_th1
            spike1_record.append(spike1)
        spikes1 = torch.stack(spike1_record, dim=2) 
        
        c1 = self.bn1(self.conv1(spikes1))
        s1 = self.shortcut1(spikes1)
        v_mem2 = torch.zeros(batch_size, 128, device=device)
        spike2_record = []
        for t in range(time_steps):
            v_mem2 = v_mem2 * (1.0 - 1.0/self.tau2) + c1[:, :, t] + s1[:, :, t]
            spike2 = FastSigmoid.apply(v_mem2 - self.v_th2)
            v_mem2 = v_mem2 - spike2 * self.v_th2
            spike2_record.append(spike2)
        spikes2 = torch.stack(spike2_record, dim=2) 
        
        c2 = self.bn2(self.conv2(spikes2))
        s2 = self.shortcut2(spikes2)
        v_mem3 = torch.zeros(batch_size, 256, device=device)
        spike3_record = []
        for t in range(time_steps):
            v_mem3 = v_mem3 * (1.0 - 1.0/self.tau3) + c2[:, :, t] + s2[:, :, t]
            spike3 = FastSigmoid.apply(v_mem3 - self.v_th3)
            v_mem3 = v_mem3 - spike3 * self.v_th3
            spike3_record.append(spike3)
        spikes3 = torch.stack(spike3_record, dim=2) 
        
        pooled_spikes = self.dp_pool(spikes3) 
        flattened = pooled_spikes.view(batch_size, -1) 
        
        out = self.fc_out(self.pre_fc_bn(flattened))
        
        if return_features:
            return out, spikes1, spikes3, flattened
        return out

# ==========================================
# [4] Main Pipeline
# ==========================================
if __name__ == "__main__":
    logging.info(f"System Ready. Device: {device}")
    
    X_features, y_all = [], []
    ch_names = None
    target_subjects = list(range(1, 106)) 
    
    logging.info("Loading Data and Extracting Features...")
    for sub in target_subjects:
        X_base, X_csd, y_raw, ch = load_physionet_dual_runpod(sub)
        if X_base is None: continue
        ch_names = ch
        feat_4d = extract_original_4d_features(X_base, X_csd)
        X_features.append(feat_4d)
        y_all.append(y_raw)
        del X_base, X_csd; gc.collect()
        
    X_tensor = torch.tensor(np.concatenate(X_features, axis=0), dtype=torch.float32)
    y_tensor = torch.tensor(np.concatenate(y_all, axis=0), dtype=torch.long)
    logging.info(f"Dataset Shape: {X_tensor.shape}, Labels: {y_tensor.shape}")
    
    workers = min(4, multiprocessing.cpu_count())
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True, 
                        num_workers=workers, pin_memory=True, persistent_workers=True)
    
    A_norm = compute_gcn_laplacian(ch_names)
    model = DeepGeoConvSNN(num_channels=64, num_classes=4, A_norm=A_norm.to(device)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    history = {'loss': [], 'acc': []}
    best_acc = 0.0
    
    logging.info("Start 200 Epoch Full-Training...")
    for epoch in range(200):
        start_t = time.time()
        model.train()
        total_loss, correct = 0, 0
        
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            
        scheduler.step()
        epoch_time = time.time() - start_t
        acc = correct / len(y_tensor) * 100
        
        history['loss'].append(total_loss/len(loader))
        history['acc'].append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            
        logging.info(f"Epoch [{epoch+1}/200] Loss: {history['loss'][-1]:.4f} | Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f} | Time: {epoch_time:.1f}s")
        
    logging.info(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

    # ==========================================
    # [5] Data Export & Plot Generation
    # ==========================================
    logging.info("Extracting features for post-hoc analysis...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    model.eval()
    
    all_preds, all_labels, all_feats = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            outputs, spk1, spk3, flat_feat = model(x_batch, return_features=True)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_feats.extend(flat_feat.cpu().numpy())
            
            # 단일 샘플 스파이크 래스터 추출 (시각화용)
            if len(all_preds) == len(y_batch): 
                sample_spk1 = spk1[0].cpu().numpy()
                sample_spk3 = spk3[0].cpu().numpy()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_feats = np.array(all_feats)
    
    # [핵심] 사후 재생성을 위한 원시 데이터(Raw data) .npz 저장
    npz_path = os.path.join(SAVE_DIR, 'snn_plot_data.npz')
    np.savez_compressed(npz_path, 
                        loss=history['loss'], acc=history['acc'], 
                        preds=all_preds, labels=all_labels, 
                        feats=all_feats, spk1=sample_spk1, spk3=sample_spk3)
    logging.info(f"Raw plotting data saved to: {npz_path}")

    # [시각화] 개별 패널 및 6-Panel 통합 피겨 생성
    class_names = ['Left Hand', 'Right Hand', 'Both Hands', 'Feet']
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
    
    def save_panel(fig_obj, filename):
        fig_obj.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight', facecolor='white')
        
    fig_main = plt.figure(figsize=(18, 12), dpi=150)
    
    # Panel A: Training Curves
    fig_A = plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax1.plot(history['loss'], label='Loss', color='#e74c3c', linewidth=2)
    ax1.set_ylabel('Cross Entropy Loss', color='#e74c3c', fontweight='bold')
    ax2 = ax1.twinx()
    ax2.plot(history['acc'], label='Accuracy', color='#3498db', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', color='#3498db', fontweight='bold')
    plt.title("A. Training Dynamics", fontsize=14, fontweight='bold')
    save_panel(fig_A, 'Panel_A_Dynamics.png')
    
    # Panel B: Raster Layer 1
    fig_B = plt.figure(figsize=(6, 4))
    ax3 = plt.gca()
    time_axis = np.arange(sample_spk1.shape[1])
    for ch_idx in range(sample_spk1.shape[0]):
        t_spikes = time_axis[sample_spk1[ch_idx] > 0]
        ax3.scatter(t_spikes, [ch_idx]*len(t_spikes), s=2, color='black', alpha=0.5)
    ax3.set_title("B. Geometric Spike Raster (Layer 1)", fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (Samples)'); ax3.set_ylabel('EEG Channels')
    save_panel(fig_B, 'Panel_B_Raster_L1.png')
    
    # Panel C: Raster Layer 3
    fig_C = plt.figure(figsize=(6, 4))
    ax4 = plt.gca()
    for ch_idx in range(sample_spk3.shape[0]):
        t_spikes = time_axis[sample_spk3[ch_idx] > 0]
        ax4.scatter(t_spikes, [ch_idx]*len(t_spikes), s=2, color='navy', alpha=0.5)
    ax4.set_title("C. Deep Spatiotemporal Raster (Layer 3)", fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (Samples)'); ax4.set_ylabel('Conv Channels')
    save_panel(fig_C, 'Panel_C_Raster_L3.png')
    
    # Panel D: t-SNE
    fig_D = plt.figure(figsize=(6, 4))
    ax5 = plt.gca()
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_feats), min(2000, len(all_feats)), replace=False)
    tsne_res = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_feats[sample_indices])
    tsne_labels = all_labels[sample_indices]
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=[class_names[y] for y in tsne_labels], 
                    palette=colors, ax=ax5, s=40, edgecolor='k', alpha=0.7)
    ax5.set_title("D. t-SNE of DP-Pooled SNN Features", fontsize=14, fontweight='bold')
    save_panel(fig_D, 'Panel_D_tSNE.png')

    # Panel E: Confusion Matrix
    fig_E = plt.figure(figsize=(6, 4))
    ax6 = plt.gca()
    cm = confusion_matrix(all_labels, all_preds)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', ax=ax6, 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    ax6.set_title(f"E. Confusion Matrix\n(Best Acc: {best_acc:.1f}%)", fontsize=14, fontweight='bold')
    ax6.set_ylabel('True Task'); ax6.set_xlabel('Predicted Task')
    save_panel(fig_E, 'Panel_E_Confusion_Matrix.png')

    # Panel F: Sparsity Barplot
    fig_F = plt.figure(figsize=(6, 4))
    ax7 = plt.gca()
    layer_rates = [sample_spk1.mean(), sample_spk3.mean()]
    sns.barplot(x=['Layer 1 (Geometric)', 'Layer 3 (Deep)'], y=layer_rates, ax=ax7, palette='viridis', edgecolor='k')
    ax7.set_title("F. Mean Spike Firing Rate", fontsize=14, fontweight='bold')
    ax7.set_ylabel("Spikes per Timestep")
    save_panel(fig_F, 'Panel_F_Sparsity.png')
    
    plt.close('all') # 메모리 정리
    logging.info("Individual panels and data saved successfully.")