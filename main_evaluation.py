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
# 1. 기하학적 특징 및 라플라시안 그래프 전처리 (최종 완성형)
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
    save_path_L = f"{SAVE_DIR}/{subj}_L.pt" 
    save_path_X = f"{SAVE_DIR}/{subj}_X.pt" 
    save_path_y = f"{SAVE_DIR}/{subj}_y.pt"
    if os.path.exists(save_path_L): return
        
    runs_hands = ['R04', 'R08', 'R12']
    runs_feet = ['R06', 'R10', 'R14']
    epochs_list = []
    
    try:
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

        # Hardy Space 투영 후 Envelope 추출
        hardy_signals = project_to_hardy_space(data, sfreq)
        envelopes = np.abs(hardy_signals) # Shape: (Epochs, 64, 480)
        
        # 💡 Rest 상태 공분산 계산 (Reference Point)
        rest_idx = (labels == 0)
        rest_envelopes = envelopes[rest_idx]
        rest_covs = np.array([compute_spd_cov(env) for env in rest_envelopes])
        p_rest = np.mean(rest_covs, axis=0) 
        p_rest_sqrt = sqrtm(p_rest).real
        p_rest_inv_sqrt = inv(p_rest_sqrt)

        L_norm_list = []
        X_feat_list = []
        
        num_steps = 15
        chunk_size = envelopes.shape[-1] // num_steps # 480 // 15 = 32
        
        for ep_idx in range(envelopes.shape[0]):
            env = envelopes[ep_idx] # (64, 480)
            
            # 💡 1. 지도 제작: 3초 전체 데이터에서 O(N^3) 연산은 딱 한 번만 수행
            cov_full = compute_spd_cov(env)
            inner = p_rest_inv_sqrt @ cov_full @ p_rest_inv_sqrt
            tangent = p_rest_sqrt @ logm(inner).real @ p_rest_sqrt
            
            # k-NN 라플라시안 그래프 생성 (k=8)
            A = np.abs(tangent)
            np.fill_diagonal(A, 0) 
            
            k = 8 
            A_sparse = np.zeros_like(A)
            for i in range(A.shape[0]):
                idx = np.argsort(A[i])[-k:]
                A_sparse[i, idx] = A[i, idx]
            
            A_sparse = np.maximum(A_sparse, A_sparse.T) 
            
            D = np.diag(np.sum(A_sparse, axis=1))
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
            L_norm = np.eye(A.shape[0]) - (D_inv_sqrt @ A_sparse @ D_inv_sqrt)
            
            # 💡 2. 자동차 제작: 3초 궤적을 15조각으로 쪼개서 순차적 시계열 피처 생성
            X_seq = []
            for t in range(num_steps):
                chunk = env[:, t*chunk_size : (t+1)*chunk_size]
                # 각 조각의 평균 에너지를 노드 피처로 사용
                X_seq.append(np.mean(chunk, axis=1, keepdims=True)) 
            
            L_norm_list.append(L_norm)
            X_feat_list.append(np.stack(X_seq, axis=0)) # (15, 64, 1)
        
        L_norm_list = np.array(L_norm_list, dtype=np.float32) # (Epochs, 64, 64)
        X_feat_list = np.array(X_feat_list, dtype=np.float32) # (Epochs, 15, 64, 1)

        scale_X = np.max(np.abs(X_feat_list))
        if scale_X > 0: X_feat_list = X_feat_list / scale_X

        torch.save(torch.tensor(L_norm_list), save_path_L)
        torch.save(torch.tensor(X_feat_list), save_path_X)
        torch.save(torch.tensor(labels, dtype=torch.long), save_path_y)
        
    except Exception as e:
        print(f"🚨 {subj} 전처리 에러: {e}")

print("🚀 1단계: O(N^3) 최적화 전처리 시작")
for subj in tqdm(all_subjects):
    process_and_save_subject_graph(subj)

# =====================================================================
# 2. 데이터 로드 및 Laplacian GCN 모듈
# =====================================================================
def load_graph_data(subject_list):
    L_list, X_list, y_list = [], [], []
    for subj in subject_list:
        if os.path.exists(f"{SAVE_DIR}/{subj}_L.pt"):
            L_list.append(torch.load(f"{SAVE_DIR}/{subj}_L.pt", weights_only=True))
            X_list.append(torch.load(f"{SAVE_DIR}/{subj}_X.pt", weights_only=True))
            y_list.append(torch.load(f"{SAVE_DIR}/{subj}_y.pt", weights_only=True))
    if not L_list: return None, None, None
    return torch.cat(L_list, dim=0), torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

class DenseLaplacianConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, L_norm, X):
        # L_norm: [B, 64, 64], X: [B, 64, in_features]
        support = torch.matmul(X, self.weight) 
        out = torch.bmm(L_norm, support)       
        return out + self.bias

# =====================================================================
# 3. 모델 정의: 고정 지도 위에서 15개의 피처가 주행
# =====================================================================
class RiemannianGraphSNN(nn.Module):
    def __init__(self, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # 입력 에너지가 1차원(스칼라)이므로 1 -> 16 -> 32로 특징 추출
        self.gcn1 = DenseLaplacianConv(1, 16) 
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.gcn2 = DenseLaplacianConv(16, 32)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.dropout = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 15, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, L_norm, X_seq):
        # L_norm: [B, 64, 64], X_seq: [B, 15, 64, 1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        
        # 💡 3. 주행: 고정된 라플라시안 지도(L_norm) 위에서 시간에 따른 에너지 변동(X_current) 주입
        for step in range(self.num_steps):
            X_current = X_seq[:, step] * 10.0 # 강제 스파이크 발화 펌핑
            
            cur1 = self.gcn1(L_norm, X_current)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.gcn2(L_norm, spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(torch.flatten(spk2, start_dim=1))
            
        all_time_features = torch.stack(spk2_rec).transpose(0, 1).contiguous()
        flat_spatio_temporal = self.dropout(all_time_features.view(all_time_features.size(0), -1))
        
        out = self.classifier(flat_spatio_temporal)
        return out 

# =====================================================================
# 4. 학습 파이프라인
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list = []
sstl_acc_list = []

print("\n" + "="*50)
print("🧠 극비 효율화 RG-SNN (1 Graph + 15 Time Chunks) 5-Fold 학습 시작")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = load_graph_data(global_train_subjs)
    if L_train is None: continue
    train_loader = DataLoader(TensorDataset(L_train, X_train, y_train), batch_size=128, shuffle=True)
    
    net = RiemannianGraphSNN(num_steps=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    net.train()
    for epoch in range(25): 
        for L_batch, X_batch, targets in train_loader:
            L_batch, X_batch, targets = L_batch.to(device), X_batch.to(device), targets.to(device)
            outputs = net(L_batch, X_batch)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    L_unseen, X_unseen, y_unseen = load_graph_data(global_test_subjs)
    unseen_loader = DataLoader(TensorDataset(L_unseen, X_unseen, y_unseen), batch_size=128, shuffle=False)
    
    net.eval()
    unseen_corr, unseen_tot = 0, 0
    with torch.no_grad():
        for L_batch, X_batch, targets in unseen_loader:
            L_batch, X_batch, targets = L_batch.to(device), X_batch.to(device), targets.to(device)
            outputs = net(L_batch, X_batch)
            _, predicted = outputs.max(1)
            unseen_tot += targets.size(0)
            unseen_corr += (predicted == targets).sum().item()
            
    true_global_acc = 100 * unseen_corr / unseen_tot
    global_acc_list.append(true_global_acc)
    print(f"🔥 Fold {fold + 1} 진짜 Global Test Acc (p0): {true_global_acc:.2f}%")
    
    global_model_state = net.state_dict()
    fold_sstl_accs = []
    
    for subj in global_test_subjs:
        L_sub, X_sub, y_sub = load_graph_data([subj])
        if L_sub is None: continue
        
        kf_sub = KFold(n_splits=4, shuffle=True, random_state=42)
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in kf_sub.split(L_sub):
            sstl_net = RiemannianGraphSNN(num_steps=15).to(device)
            sstl_net.load_state_dict(global_model_state)
            
            for name, param in sstl_net.named_parameters():
                if 'classifier' not in name: param.requires_grad = False
                    
            sstl_optimizer = torch.optim.Adam(sstl_net.classifier.parameters(), lr=5e-4)
            
            sub_train_loader = DataLoader(TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), batch_size=32, shuffle=True)
            sub_test_loader = DataLoader(TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), batch_size=32, shuffle=False)
            
            sstl_net.train()
            for _ in range(15): 
                for L_batch, X_batch, targets in sub_train_loader:
                    L_batch, X_batch, targets = L_batch.to(device), X_batch.to(device), targets.to(device)
                    loss = criterion(sstl_net(L_batch, X_batch), targets)
                    sstl_optimizer.zero_grad()
                    loss.backward()
                    sstl_optimizer.step()
                    
            sstl_net.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for L_batch, X_batch, targets in sub_test_loader:
                    L_batch, X_batch, targets = L_batch.to(device), X_batch.to(device), targets.to(device)
                    _, predicted = sstl_net(L_batch, X_batch).max(1)
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