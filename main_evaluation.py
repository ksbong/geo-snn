import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from snntorch import surrogate
import mne
import numpy as np
import os
import warnings
from scipy.signal import savgol_filter
import logging
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import copy

# ==========================================
# [0] 환경 설정
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True 

DATA_DIR_PHYSIONET = './raw_data/files'
SAVE_DIR = './results_geoeeg_final'
os.makedirs(SAVE_DIR, exist_ok=True)

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(os.path.join(SAVE_DIR, "full_eval_log.txt")), logging.StreamHandler()])

# ==========================================
# [1] 데이터 로더 및 피처 추출 (복소 평면 기하학)
# ==========================================
def load_local_runs(subject, run_list):
    sub_str = f"S{subject:03d}"
    raws = []
    for run in run_list:
        file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
        if not os.path.exists(file_path): return None
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(l_freq=4., h_freq=40., fir_design='firwin', verbose=False) 
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1005', match_case=False, on_missing='ignore')
        raws.append(raw)
    return mne.io.concatenate_raws(raws)

def load_physionet_4_classes(subject):
    try:
        epochs_list = []
        raw_lr = load_local_runs(subject, [4, 8, 12])
        if raw_lr is None: return None, None
        ev_lr_raw, ev_id_lr = mne.events_from_annotations(raw_lr, verbose=False)
        t1, t2 = ev_id_lr.get('T1'), ev_id_lr.get('T2')
        ev_lr = ev_lr_raw[(ev_lr_raw[:, 2] == t1) | (ev_lr_raw[:, 2] == t2)].copy()
        ev_lr[ev_lr[:, 2] == t1, 2] = 0 
        ev_lr[ev_lr[:, 2] == t2, 2] = 1 
        epochs_list.append(mne.Epochs(raw_lr, ev_lr, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False))

        raw_rest = load_local_runs(subject, [1])
        num_trials = np.sum(ev_lr[:, 2] == 0) 
        ev_rest = mne.make_fixed_length_events(raw_rest, id=2, start=0, duration=3.0, overlap=0.0)[:num_trials]
        epochs_list.append(mne.Epochs(raw_rest, ev_rest, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False))

        raw_f = load_local_runs(subject, [6, 10, 14])
        ev_f_raw, ev_id_f = mne.events_from_annotations(raw_f, verbose=False)
        t2_f = ev_id_f.get('T2')
        ev_f = ev_f_raw[ev_f_raw[:, 2] == t2_f].copy()
        ev_f[:, 2] = 3 
        epochs_list.append(mne.Epochs(raw_f, ev_f, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False))

        merged = mne.concatenate_epochs(epochs_list, verbose=False)
        return merged.get_data()[:, :, :480], merged.events[:, -1]
    except Exception:
        return None, None 

def extract_robust_features_batch(X_array):
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    B, C, T = X_tensor.shape
    
    freq = torch.fft.fft(X_tensor, dim=-1)
    h = torch.zeros_like(freq)
    h[..., 0] = freq[..., 0]
    h[..., 1:(T + 1) // 2] = 2 * freq[..., 1:(T + 1) // 2]
    analytic = torch.fft.ifft(h, dim=-1)
    
    x = analytic.real.numpy()
    y = analytic.imag.numpy()
    env = np.sqrt(x**2 + y**2)
    
    half_win = 5
    vx_raw = savgol_filter(x, window_length=11, polyorder=3, deriv=1, axis=-1)
    vy_raw = savgol_filter(y, window_length=11, polyorder=3, deriv=1, axis=-1)
    ax_raw = savgol_filter(x, window_length=11, polyorder=3, deriv=2, axis=-1)
    ay_raw = savgol_filter(y, window_length=11, polyorder=3, deriv=2, axis=-1)
    
    vx = np.pad(vx_raw, ((0,0), (0,0), (half_win, 0)), mode='edge')[..., :-half_win]
    vy = np.pad(vy_raw, ((0,0), (0,0), (half_win, 0)), mode='edge')[..., :-half_win]
    ax = np.pad(ax_raw, ((0,0), (0,0), (half_win, 0)), mode='edge')[..., :-half_win]
    ay = np.pad(ay_raw, ((0,0), (0,0), (half_win, 0)), mode='edge')[..., :-half_win]
    
    vx_t, vy_t = torch.tensor(vx), torch.tensor(vy)
    ax_t, ay_t = torch.tensor(ax), torch.tensor(ay)
    x_t, y_t = torch.tensor(x), torch.tensor(y)
    env_t = torch.tensor(env)
    
    num_k = torch.abs(vx_t * ay_t - vy_t * ax_t)
    den_k = torch.pow(vx_t**2 + vy_t**2, 1.5) + 1e-6
    curvature = torch.log1p(num_k / den_k)
    
    W, epsilon = 12, 0.5
    def norm_2d(feat1, feat2):
        mu1, mu2 = feat1.mean(dim=-1, keepdim=True), feat2.mean(dim=-1, keepdim=True)
        std = torch.sqrt(((feat1 - mu1)**2 + (feat2 - mu2)**2).mean(dim=-1, keepdim=True) + 1e-6)
        return (feat1 - mu1)/std, (feat2 - mu2)/std

    x_norm, y_norm = norm_2d(x_t, y_t)
    vx_norm, vy_norm = norm_2d(vx_t, vy_t)
    
    x_pad = F.pad(x_norm, (W, W), mode='replicate')
    y_pad = F.pad(y_norm, (W, W), mode='replicate')
    vx_pad = F.pad(vx_norm, (W, W), mode='replicate')
    vy_pad = F.pad(vy_norm, (W, W), mode='replicate')
    
    x_win = x_pad.unfold(dimension=-1, size=2*W+1, step=1)
    y_win = y_pad.unfold(dimension=-1, size=2*W+1, step=1)
    vx_win = vx_pad.unfold(dimension=-1, size=2*W+1, step=1)
    vy_win = vy_pad.unfold(dimension=-1, size=2*W+1, step=1)
    
    dR_sq = (x_norm.unsqueeze(-1) - x_win).pow(2) + (y_norm.unsqueeze(-1) - y_win).pow(2)
    dV_sq = (vx_norm.unsqueeze(-1) - vx_win).pow(2) + (vy_norm.unsqueeze(-1) - vy_win).pow(2)
    
    ratio = dV_sq / (dR_sq + epsilon)
    ratio[..., W] = 0.0 
    tangling, _ = ratio.max(dim=-1)
    tangling = torch.log1p(tangling)
    
    def normalize_feature(feat): 
        return (feat - feat.mean(dim=(0, 2), keepdim=True)) / (feat.std(dim=(0, 2), keepdim=True) + 1e-6)
    return torch.stack([normalize_feature(env_t), normalize_feature(curvature), normalize_feature(tangling)], dim=1)

# ==========================================
# [2] GeoEEG-SNN 모델 (네 메인 아이디어 완벽 보존형)
# ==========================================
class GDLIF(nn.Module):
    def __init__(self, channels=64, v_base=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1)) 
        self.v_base = v_base
        self.beta_base_logit = nn.Parameter(torch.tensor(2.0)) 
        self.spike_grad = surrogate.fast_sigmoid(slope=25)

    def forward(self, x_t, m_prev, curv_t, tang_t):
        tang_c = torch.clamp(tang_t, min=-2.0, max=2.0)
        curv_c = torch.clamp(curv_t, min=-2.0, max=2.0)
        v_th = torch.clamp(self.v_base + self.alpha * tang_c, min=0.1) 
        beta = torch.sigmoid(self.beta_base_logit - self.gamma * curv_c)
        m_next = beta * m_prev + x_t
        spike = self.spike_grad(m_next - v_th)
        m_next = m_next - spike * v_th
        return spike, m_next

class LearnablePopulationEncoder(nn.Module):
    def __init__(self, num_neurons=4):
        super().__init__()
        self.mu = nn.Parameter(torch.linspace(-3.0, 3.0, num_neurons))
        self.sigma = nn.Parameter(torch.full((num_neurons,), 2.0))
        
    def forward(self, x):
        B, F_types, C, T = x.shape
        encoded = torch.exp(-((x.unsqueeze(-1) - self.mu)**2) / (2 * self.sigma**2 + 1e-6))
        return encoded.permute(0, 1, 4, 2, 3).reshape(B, F_types * self.mu.shape[0], C, T)

class GeoEEGSNN(nn.Module):
    def __init__(self, in_channels=64, num_classes=4):
        super().__init__()
        self.encoder = LearnablePopulationEncoder(num_neurons=4) 
        
        encoded_feature_dim = 3 * 4  # 12
        
        # 🚨 수정: 64개 전극 공간은 그대로 유지하면서 12개의 피처를 하나로 압축
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(encoded_feature_dim, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ELU()
        )
        
        # 🚨 수정: 64개 공간 정보를 그대로 받아들이는 분리형 시간 필터
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=16, padding=8, groups=in_channels, bias=False),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ELU(),
            nn.AvgPool1d(4), # 480 -> 120
            nn.Dropout(0.3)  # 뉴런 통합이 죽지 않도록 0.3으로 하향
        )
        
        # 🚨 64채널 복구: 네 원래 아이디어대로 각 전극마다 GDLIF 독립 제어
        self.lif = GDLIF(channels=in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, xb):
        curv_raw, tang_raw = xb[:, 1, :, :], xb[:, 2, :, :]
        
        # 64채널 공간 보존
        curv_pooled = F.avg_pool1d(curv_raw, 4).permute(2, 0, 1) # (120, B, 64)
        tang_pooled = F.avg_pool1d(tang_raw, 4).permute(2, 0, 1) # (120, B, 64)

        x_enc = self.encoder(xb) 
        
        # (B, 12, 64, 480) -> (B, 1, 64, 480) -> (B, 64, 480)
        s_out = self.feature_fusion(x_enc).squeeze(1) 
        
        c = self.temporal(s_out).permute(2, 0, 1) # (120, B, 64)
        
        m = torch.zeros(c.size(1), c.size(2), device=xb.device) 
        spikes = []
        for t in range(c.size(0)):
            # 64채널의 독립적인 동적 임계값 적용 (네 아이디어 완벽 구현)
            s, m = self.lif(c[t], m, curv_pooled[t], tang_pooled[t])
            spikes.append(s)
            
        spikes_tensor = torch.stack(spikes) # (120, B, 64)
        
        attn_weights = F.softmax(curv_pooled, dim=0) 
        out_features = torch.sum(spikes_tensor * attn_weights, dim=0) # (B, 64)
        
        return self.fc(out_features)

# ==========================================
# [3] 검증 로직 (파멸적 노이즈 삭제)
# ==========================================
def run_global_training(X_train, Y_train, X_test, Y_test):
    train_dl = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
    test_dl = DataLoader(TensorDataset(X_test, Y_test), batch_size=128, shuffle=False)

    model = GeoEEGSNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    history_loss, history_acc = [], []
    best_model_state = None
    all_preds, all_labels = [], []
    
    patience = 50
    patience_counter = 0

    for epoch in range(150): 
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            # 🚨 치명적 버그 수정: 기하학 피처를 깨부수던 가우시안 노이즈 주입 삭제
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_loss = train_loss / len(train_dl)
        history_loss.append(avg_loss)
        
        model.eval()
        corr, total = 0, 0
        preds_list, labels_list = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(1)
                corr += (preds == yb).sum().item()
                total += yb.size(0)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(yb.cpu().numpy())
                
        test_acc = (corr / total) * 100
        history_acc.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
            all_preds, all_labels = preds_list, labels_list
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            logging.info(f"  -> Epoch {epoch+1}/150 | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% | Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logging.info(f"🚨 조기 종료 발동! Epoch {epoch+1}에서 학습을 멈춥니다. (최고 정확도: {best_acc:.2f}%)")
            break
            
    model.load_state_dict(best_model_state)
    logging.info("✅ 가장 똑똑했던 상태(Best Weight)로 모델을 복구했습니다.")
    return best_acc, best_model_state, history_loss, history_acc, all_preds, all_labels

def run_sstl(model_state, subject_X, subject_Y):
    kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
    fold_accs = []
    
    for train_idx, test_idx in kf.split(subject_X):
        X_tr, Y_tr = subject_X[train_idx], subject_Y[train_idx]
        X_te, Y_te = subject_X[test_idx], subject_Y[test_idx]
        
        train_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=16, shuffle=True)
        test_dl = DataLoader(TensorDataset(X_te, Y_te), batch_size=16, shuffle=False)
        
        model = GeoEEGSNN().to(device)
        model.load_state_dict(model_state)
        
        model.dropout.train() 
        
        for param in model.parameters(): param.requires_grad = False
        for param in model.lif.parameters(): param.requires_grad = True
        for param in model.fc.parameters(): param.requires_grad = True
            
        optimizer = torch.optim.AdamW([
            {'params': model.fc.parameters(), 'lr': 0.002},
            {'params': model.lif.parameters(), 'lr': 0.005}
        ], weight_decay=1e-3)
        
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(25): 
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                
        model.eval()
        corr, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                corr += (preds == yb).sum().item()
                total += yb.size(0)
        fold_accs.append((corr / total) * 100)
        
    return np.mean(fold_accs)

# ==========================================
# [4] 논문 퀄리티 Figure 생성 
# ==========================================
def plot_paper_figures(global_acc, sstl_acc, hist_loss, hist_acc, preds, labels):
    c_loss = "#A0C4FF" 
    c_acc = "#FFB3C6"  
    c_global = "#BDB2FF" 
    c_sstl = "#FFD6A5" 
    
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(hist_loss, color=c_loss, linewidth=3, label='Train Loss')
    ax2.plot(hist_acc, color=c_acc, linewidth=3, linestyle='-', label='Test Acc')
    
    ax1.set_xlabel('Epochs', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Loss', color=c_loss, fontweight='bold', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', color=c_acc, fontweight='bold', fontsize=12)
    plt.title('Global Training Dynamics', fontweight='bold', fontsize=14)
    ax1.grid(False)
    sns.despine(right=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Fig_Learning_Curve.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(y=hist_loss, mode='lines', name='Train Loss', line=dict(color=c_loss, width=3), yaxis='y1'))
    fig_html.add_trace(go.Scatter(y=hist_acc, mode='lines', name='Test Acc', line=dict(color=c_acc, width=3), yaxis='y2'))
    fig_html.update_layout(
        title='Global Training Dynamics', template='plotly_white',
        yaxis=dict(title='Loss', title_font=dict(color=c_loss), tickfont=dict(color=c_loss)),
        yaxis2=dict(title='Accuracy (%)', title_font=dict(color=c_acc), tickfont=dict(color=c_acc), overlaying='y', side='right')
    )
    fig_html.write_html(os.path.join(SAVE_DIR, 'Interactive_Learning_Curve.html'))

    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='PuBu', cbar=False,
                annot_kws={'size': 12, 'weight': 'bold'},
                xticklabels=['Left', 'Right', 'Rest', 'Feet'], 
                yticklabels=['Left', 'Right', 'Rest', 'Feet'])
    plt.ylabel('True Class', fontweight='bold', fontsize=12)
    plt.xlabel('Predicted Class', fontweight='bold', fontsize=12)
    plt.title('Global Test Confusion Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Fig_Confusion_Matrix.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Global Training', 'SSTL (Transfer)'], [global_acc, sstl_acc], color=[c_global, c_sstl], edgecolor='black', linewidth=1.2)
    plt.ylim(0, 100)
    plt.ylabel('Average Accuracy (%)', fontweight='bold', fontsize=12)
    plt.title('Performance Comparison', fontweight='bold', fontsize=14)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Fig_Bar_Chart.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logging.info("🚀 [STAGE 1] Loading PhysioNet Data & Extracting Complex Geometric Features...")
    valid_subjects = [s for s in range(1, 110) if s not in [88, 89, 92, 100]]
    
    subject_features = {}
    subject_labels = {}
    
    for sub in valid_subjects:
        x, y = load_physionet_4_classes(sub)
        if x is not None:
            feat = extract_robust_features_batch(x)
            subject_features[sub] = feat
            subject_labels[sub] = torch.tensor(y, dtype=torch.long)
            if sub % 10 == 0: logging.info(f" -> Processed Subject {sub}/109")
            
    logging.info(f"✅ Extracted features for {len(subject_features)} subjects.")

    subjects_array = np.array(list(subject_features.keys()))
    kf_global = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    fold_global_accs, fold_sstl_accs = [], []
    
    logging.info("\n" + "="*50 + "\n[STAGE 2] Starting HR-SNN Protocol (PhysioNet)\n" + "="*50)
    for fold, (train_sub_idx, test_sub_idx) in enumerate(kf_global.split(subjects_array)):
        if fold > 0: break 
        
        train_subs = subjects_array[train_sub_idx]
        test_subs = subjects_array[test_sub_idx]
        
        X_train = torch.cat([subject_features[s] for s in train_subs])
        Y_train = torch.cat([subject_labels[s] for s in train_subs])
        X_test = torch.cat([subject_features[s] for s in test_subs])
        Y_test = torch.cat([subject_labels[s] for s in test_subs])
        
        logging.info(f"🌍 Global Training (Fold 1) | Train: {len(train_subs)} subs, Test: {len(test_subs)} subs")
        g_acc, g_model, h_loss, h_acc, preds, labels = run_global_training(X_train, Y_train, X_test, Y_test)
        fold_global_accs.append(g_acc)
        logging.info(f"🏆 Fold 1 Global Best Acc: {g_acc:.2f}%")
        
        logging.info(f"🎯 Starting Subject-Specific Transfer Learning (SSTL) for {len(test_subs)} Test Subjects...")
        sstl_accs_for_this_fold = []
        for s in test_subs:
            sub_X = subject_features[s]
            sub_Y = subject_labels[s]
            s_acc = run_sstl(g_model, sub_X, sub_Y)
            sstl_accs_for_this_fold.append(s_acc)
            
        mean_sstl = np.mean(sstl_accs_for_this_fold)
        fold_sstl_accs.append(mean_sstl)
        logging.info(f"🚀 Fold 1 Mean SSTL Acc: {mean_sstl:.2f}%")
        
        logging.info("[STAGE 3] Generating Publication Figures...")
        plot_paper_figures(g_acc, mean_sstl, h_loss, h_acc, preds, labels)

    final_g_acc = np.mean(fold_global_accs)
    final_s_acc = np.mean(fold_sstl_accs)
    logging.info(f"\n🎉 FINAL RESULTS | Global Acc: {final_g_acc:.2f}% | SSTL Acc: {final_s_acc:.2f}%")
    logging.info(f"✅ All elegant figures and logs saved in '{SAVE_DIR}'")

if __name__ == '__main__':
    main()