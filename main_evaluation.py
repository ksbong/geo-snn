import os
import shutil
import mne
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import sqrtm, logm, inv
from sklearn.model_selection import KFold
import snntorch as snn
from snntorch import surrogate
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =====================================================================
# 1. 기하학적 특징 및 라플라시안 그래프 전처리
# =====================================================================
def project_to_hardy_space(data, sfreq, l_freq=8.0, h_freq=30.0):
    n_times = data.shape[-1]
    freqs = fftfreq(n_times, 1/sfreq)
    mask = np.zeros_like(freqs)
    mask[(freqs >= l_freq) & (freqs <= h_freq)] = 2.0  
    X = fft(data, axis=-1)
    return ifft(X * mask, axis=-1)

def compute_spd_cov(envelope):
    cov = np.cov(envelope)
    epsilon = 1e-4 * (np.trace(cov) / cov.shape[0])
    return cov + np.eye(cov.shape[0]) * epsilon

DATA_DIR = './raw_data/files/'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './raw_data/files'

SAVE_DIR = './processed_graph_tensors'
os.makedirs(SAVE_DIR, exist_ok=True)

exclude_subjects = ['S088', 'S092', 'S100', 'S104']
all_subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in exclude_subjects]
def process_and_save_subject_graph(subj):
    save_path_A = f"{SAVE_DIR}/{subj}_A.pt" 
    save_path_X = f"{SAVE_DIR}/{subj}_X.pt" 
    save_path_y = f"{SAVE_DIR}/{subj}_y.pt"
    if os.path.exists(save_path_A): return
        
    runs_hands = ['R04', 'R08', 'R12']
    runs_feet = ['R06', 'R10', 'R14']
    epochs_list = []
    
    try:
        # (데이터 로드 및 윈도우 분할 등 이전과 동일 부분 생략, 아래 루프가 핵심)
        for run in runs_hands:
            path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
            ep = mne.Epochs(raw, evs, {'Rest': ev_dict['T0'], 'Left Hand': ev_dict['T1'], 'Right Hand': ev_dict['T2']}, 
                            tmin=1.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            epochs_list.append(ep)

        for run in runs_feet:
            path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
            events_fixed = evs.copy()
            events_fixed[events_fixed[:, 2] == ev_dict['T2'], 2] = 4
            ep = mne.Epochs(raw, events_fixed, {'Rest': ev_dict['T0'], 'Both Feet': 4}, 
                            tmin=1.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            epochs_list.append(ep)
            
        if not epochs_list: return
        epochs_all = mne.concatenate_epochs(epochs_list, verbose=False)
        
        data = epochs_all.get_data() * 1e6  
        labels = np.array(epochs_all.events[:, 2]) - 1 
        sfreq = epochs_all.info['sfreq']

        hardy_signals = project_to_hardy_space(data, sfreq)
        envelopes = np.abs(hardy_signals)
        
        win_len = 160; stride = 80; num_windows = 5
        
        X_covs_seq = []
        for w in range(num_windows):
            start = w * stride; end = start + win_len
            env_win = envelopes[:, :, start:end]
            X_covs_seq.append(np.array([compute_spd_cov(env) for env in env_win]))
        X_covs_seq = np.stack(X_covs_seq, axis=1)

        rest_covs = X_covs_seq[labels == 0]
        p_rest = np.mean(rest_covs, axis=(0, 1)) 
        p_rest_sqrt = sqrtm(p_rest).real
        p_rest_inv_sqrt = inv(p_rest_sqrt)

        A_norm_seq = []
        X_feat_seq = []
        
        for ep_idx in range(X_covs_seq.shape[0]):
            ep_A = []; ep_X = []
            for w in range(num_windows):
                cov = X_covs_seq[ep_idx, w]
                inner = p_rest_inv_sqrt @ cov @ p_rest_inv_sqrt
                tangent = p_rest_sqrt @ logm(inner).real @ p_rest_sqrt
                
                # 💡 핵심 수정: Tangent Matrix 자체(64x64)를 노드 피처로 통째로 사용!
                node_features = tangent 
                
                A = np.abs(tangent)
                np.fill_diagonal(A, A.diagonal() + 1.0) 
                
                D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
                A_norm = D_inv_sqrt @ A @ D_inv_sqrt
                
                ep_A.append(A_norm)
                ep_X.append(node_features)
                
            A_norm_seq.append(ep_A)
            X_feat_seq.append(ep_X)
        
        A_norm_seq = np.array(A_norm_seq, dtype=np.float32) 
        X_feat_seq = np.array(X_feat_seq, dtype=np.float32) # 이제 (Epochs, 5, 64, 64)가 됨!

        scale_X = np.max(np.abs(X_feat_seq))
        if scale_X > 0: X_feat_seq = X_feat_seq / scale_X

        torch.save(torch.tensor(A_norm_seq), save_path_A)
        torch.save(torch.tensor(X_feat_seq), save_path_X)
        torch.save(torch.tensor(labels, dtype=torch.long), save_path_y)
        
    except Exception as e:
        print(f"🚨 {subj} 전처리 에러: {e}")

print("🚀 1단계: 라플라시안 그래프 텐서 변환 및 저장 시작")
for subj in tqdm(all_subjects):
    process_and_save_subject_graph(subj)

# =====================================================================
# 2. 데이터 로드 및 Dense GCN 모듈
# =====================================================================
def load_graph_data(subject_list):
    A_list, X_list, y_list = [], [], []
    for subj in subject_list:
        if os.path.exists(f"{SAVE_DIR}/{subj}_A.pt"):
            A_list.append(torch.load(f"{SAVE_DIR}/{subj}_A.pt", weights_only=True))
            X_list.append(torch.load(f"{SAVE_DIR}/{subj}_X.pt", weights_only=True))
            y_list.append(torch.load(f"{SAVE_DIR}/{subj}_y.pt", weights_only=True))
    if not A_list: return None, None, None
    return torch.cat(A_list, dim=0), torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

# 순수 PyTorch로 구현한 배치 연산 지원 Dense Graph Convolution
class DenseGCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, A_norm, X):
        # A_norm: [Batch, Nodes, Nodes], X: [Batch, Nodes, Features]
        support = torch.matmul(X, self.weight) # [Batch, 64, out_features]
        out = torch.bmm(A_norm, support)       # 그래프 위상에 따라 정보 전달!
        return out + self.bias

# =====================================================================
# 3. Riemannian Graph-SNN (RG-SNN) 모델
# =====================================================================
class RiemannianGraphSNN(nn.Module):
    def __init__(self, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # 💡 입력 피처가 1차원 -> 64차원(Tangent 행벡터)으로 확장됨!
        self.gcn1 = DenseGCNConv(64, 128) # 64 -> 128 채널로 펌핑
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.gcn2 = DenseGCNConv(128, 64)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.dropout = nn.Dropout(0.4)
        
        # 차원: 64노드 * 64특징 * 15스텝
        self.classifier = nn.Sequential(
            nn.Linear(64 * 64 * 15, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4)
        )

    def forward(self, A_norm_seq, X_feat_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        steps_per_window = self.num_steps // 5
        
        for step in range(self.num_steps):
            w_idx = step // steps_per_window
            if w_idx >= 5: w_idx = 4
            
            A_current = A_norm_seq[:, w_idx]
            X_current = X_feat_seq[:, w_idx] * 10.0 # 스파이크 발화 강제 펌핑
            
            cur1 = self.gcn1(A_current, X_current)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.gcn2(A_current, spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(torch.flatten(spk2, start_dim=1))
            
        all_time_features = torch.stack(spk2_rec).transpose(0, 1).contiguous()
        flat_spatio_temporal = self.dropout(all_time_features.view(all_time_features.size(0), -1))
        
        out = self.classifier(flat_spatio_temporal)
        return out
# =====================================================================
# 4. 학습 파이프라인 (5-Fold & SSTL)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list = []
sstl_acc_list = []

print("\n" + "="*50)
print("🧠 Riemannian Graph-SNN (RG-SNN) 5-Fold 학습 시작")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    A_train, X_train, y_train = load_graph_data(global_train_subjs)
    if A_train is None:
        raise ValueError(f"🚨 [Fold {fold+1}] 데이터 로드 실패. 전처리 로그를 확인하세요.")
        
    train_loader = DataLoader(TensorDataset(A_train, X_train, y_train), batch_size=128, shuffle=True)
    
    net = RiemannianGraphSNN(num_steps=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    net.train()
    for epoch in range(25): 
        for A_batch, X_batch, targets in train_loader:
            A_batch, X_batch, targets = A_batch.to(device), X_batch.to(device), targets.to(device)
            outputs = net(A_batch, X_batch)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 2. 진짜 Global Test Accuracy (Unseen)
    A_unseen, X_unseen, y_unseen = load_graph_data(global_test_subjs)
    unseen_loader = DataLoader(TensorDataset(A_unseen, X_unseen, y_unseen), batch_size=128, shuffle=False)
    
    net.eval()
    unseen_corr, unseen_tot = 0, 0
    with torch.no_grad():
        for A_batch, X_batch, targets in unseen_loader:
            A_batch, X_batch, targets = A_batch.to(device), X_batch.to(device), targets.to(device)
            outputs = net(A_batch, X_batch)
            _, predicted = outputs.max(1)
            unseen_tot += targets.size(0)
            unseen_corr += (predicted == targets).sum().item()
            
    true_global_acc = 100 * unseen_corr / unseen_tot
    global_acc_list.append(true_global_acc)
    print(f"🔥 Fold {fold + 1} 진짜 Global Test Acc (p0): {true_global_acc:.2f}%")
    
    # 3. SSTL
    global_model_state = net.state_dict()
    fold_sstl_accs = []
    
    for subj in global_test_subjs:
        A_sub, X_sub, y_sub = load_graph_data([subj])
        if A_sub is None: continue
        
        kf_sub = KFold(n_splits=4, shuffle=True, random_state=42)
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in kf_sub.split(A_sub):
            sstl_net = RiemannianGraphSNN(num_steps=15).to(device)
            sstl_net.load_state_dict(global_model_state)
            
            for name, param in sstl_net.named_parameters():
                if 'classifier' not in name: param.requires_grad = False
                    
            sstl_optimizer = torch.optim.Adam(sstl_net.classifier.parameters(), lr=5e-4)
            
            sub_train_loader = DataLoader(TensorDataset(A_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), batch_size=32, shuffle=True)
            sub_test_loader = DataLoader(TensorDataset(A_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), batch_size=32, shuffle=False)
            
            sstl_net.train()
            for _ in range(15): # SSTL 에폭을 15로 증가
                for A_batch, X_batch, targets in sub_train_loader:
                    A_batch, X_batch, targets = A_batch.to(device), X_batch.to(device), targets.to(device)
                    loss = criterion(sstl_net(A_batch, X_batch), targets)
                    sstl_optimizer.zero_grad()
                    loss.backward()
                    sstl_optimizer.step()
                    
            sstl_net.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for A_batch, X_batch, targets in sub_test_loader:
                    A_batch, X_batch, targets = A_batch.to(device), X_batch.to(device), targets.to(device)
                    _, predicted = sstl_net(A_batch, X_batch).max(1)
                    tot += targets.size(0)
                    corr += (predicted == targets).sum().item()
            subj_fold_accs.append(100 * corr / tot)
            
        fold_sstl_accs.append(np.mean(subj_fold_accs))
        
    avg_sstl_fold = np.mean(fold_sstl_accs)
    sstl_acc_list.append(avg_sstl_fold)
    print(f"🎯 Fold {fold + 1} SSTL 평균 정확도 (ps): {avg_sstl_fold:.2f}%")

print("\n" + "="*50)
print(f"🏆 5-Fold 최종 평균 Global Test Acc (p0): {np.mean(global_acc_list):.2f}%")
print(f"🏆 5-Fold 최종 평균 SSTL Acc (ps): {np.mean(sstl_acc_list):.2f}%")
print("="*50)