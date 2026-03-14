import os
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
# 1. 기하학적 특징 및 라플라시안 그래프 전처리 (너의 Novelty 완벽 유지)
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

        hardy_signals = project_to_hardy_space(data, sfreq)
        envelopes = np.abs(hardy_signals) 
        
        rest_idx = (labels == 0)
        rest_envelopes = envelopes[rest_idx]
        rest_covs = np.array([compute_spd_cov(env) for env in rest_envelopes])
        p_rest = np.mean(rest_covs, axis=0) 
        p_rest_sqrt = sqrtm(p_rest).real
        p_rest_inv_sqrt = inv(p_rest_sqrt)

        L_norm_list = []
        X_feat_list = []
        
        num_steps = 15
        chunk_size = envelopes.shape[-1] // num_steps 
        
        for ep_idx in range(envelopes.shape[0]):
            env = envelopes[ep_idx] 
            
            cov_full = compute_spd_cov(env)
            inner = p_rest_inv_sqrt @ cov_full @ p_rest_inv_sqrt
            tangent = p_rest_sqrt @ logm(inner).real @ p_rest_sqrt
            
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
            
            X_seq = []
            for t in range(num_steps):
                chunk = env[:, t*chunk_size : (t+1)*chunk_size]
                X_seq.append(np.mean(chunk, axis=1, keepdims=True)) 
            
            L_norm_list.append(L_norm)
            X_feat_list.append(np.stack(X_seq, axis=0)) 
        
        L_norm_list = np.array(L_norm_list, dtype=np.float32) 
        X_feat_list = np.array(X_feat_list, dtype=np.float32) 

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

def load_graph_data(subject_list):
    L_list, X_list, y_list = [], [], []
    for subj in subject_list:
        if os.path.exists(f"{SAVE_DIR}/{subj}_L.pt"):
            L_list.append(torch.load(f"{SAVE_DIR}/{subj}_L.pt", weights_only=True))
            X_list.append(torch.load(f"{SAVE_DIR}/{subj}_X.pt", weights_only=True))
            y_list.append(torch.load(f"{SAVE_DIR}/{subj}_y.pt", weights_only=True))
    if not L_list: return None, None, None
    return torch.cat(L_list, dim=0), torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

# =====================================================================
# 2. 모델 모듈: HR-SNN 아키텍처 결합
# =====================================================================
class DenseLaplacianConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, L_norm, X):
        support = torch.matmul(X, self.weight) 
        out = torch.bmm(L_norm, support)       
        return out + self.bias

# 💡 하이브리드 응답(HR) GCN 모듈 + 스킵 연결
class HR_GCN_Module(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.gcn = DenseLaplacianConv(in_feat, out_feat)
        self.skip = nn.Linear(in_feat, out_feat) # 펄스 소멸 방지용 스킵 브랜치
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # 3개의 서로 다른 응답 특성을 가진 뉴런 세트 구성
        self.lif_act = snn.Leaky(beta=0.8, threshold=0.8, spike_grad=spike_grad)   # 쉽게 발화, 빨리 잊음
        self.lif_neu = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=spike_grad)   # 중립
        self.lif_mem = snn.Leaky(beta=0.95, threshold=1.5, spike_grad=spike_grad)  # 어렵게 발화, 오래 기억

    def forward(self, L_norm, x, mem_act, mem_neu, mem_mem):
        cur = self.gcn(L_norm, x)
        
        spk_act, mem_act = self.lif_act(cur, mem_act)
        spk_neu, mem_neu = self.lif_neu(cur, mem_neu)
        spk_mem, mem_mem = self.lif_mem(cur, mem_mem)
        
        # 스파이크 병합 및 스킵 연결 더하기
        spk_out = spk_act + spk_neu + spk_mem + self.skip(x)
        return spk_out, mem_act, mem_neu, mem_mem

    def init_hidden(self):
        return self.lif_act.init_leaky(), self.lif_neu.init_leaky(), self.lif_mem.init_leaky()

# 💡 DP-Pooling 스파이킹 디코더
class DP_SpikingDecoder(nn.Module):
    def __init__(self, in_features, num_classes=4, n_dp=5):
        super().__init__()
        self.n_dp = n_dp # 양끝에서 평균을 낼 타임스텝 수 (15스텝 중 5스텝씩)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, potentials):
        # potentials: [B, 15, Features]
        start_pool = potentials[:, :self.n_dp, :].mean(dim=1)
        end_pool = potentials[:, -self.n_dp:, :].mean(dim=1)
        
        # 시간의 흐름에 따른 전위차(Difference of Potential) 추출
        dp_feat = end_pool - start_pool 
        
        out = self.classifier(dp_feat)
        return out

# =====================================================================
# 3. 메인 모델: 리만 그래프 + HR-SNN
# =====================================================================
class RiemannianGraph_HRSNN(nn.Module):
    def __init__(self, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        
        # 하드코딩된 *10 대신, 아날로그 에너지를 다차원 특징으로 투영하여 스파이크 생성을 유도
        self.input_proj = nn.Linear(1, 16) 
        
        # 다중 임계값 HR-GCN 모듈을 직렬로 연결
        self.hr_gcn1 = HR_GCN_Module(16, 32)
        self.hr_gcn2 = HR_GCN_Module(32, 64)
        
        # 마지막은 연속적인 전위값을 출력하는 Leaky Integrator (LI) 레이어
        self.li_layer = snn.Leaky(beta=0.9, reset_mechanism="none") 
        
        # 공간 축을 쫙 편 크기(64 노드 * 64 차원)를 DP 디코더에 전달
        self.decoder = DP_SpikingDecoder(in_features=64 * 64, num_classes=4, n_dp=5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, L_norm, X_seq):
        B = X_seq.size(0)
        
        # 모든 뉴런의 막 전위 초기화
        m1_a, m1_n, m1_m = self.hr_gcn1.init_hidden()
        m2_a, m2_n, m2_m = self.hr_gcn2.init_hidden()
        mem_li = self.li_layer.init_leaky()
        
        potentials = []
        
        for step in range(self.num_steps):
            x_t = X_seq[:, step] # [B, 64, 1]
            x_t = self.input_proj(x_t) # [B, 64, 16] (Learnable Encoding)
            
            spk1, m1_a, m1_n, m1_m = self.hr_gcn1(L_norm, x_t, m1_a, m1_n, m1_m)
            spk2, m2_a, m2_n, m2_m = self.hr_gcn2(L_norm, spk1, m2_a, m2_n, m2_m)
            
            # LI 레이어로 최종 스파이크를 연속적인 전위(Potential)로 누적 변환
            _, mem_li = self.li_layer(spk2, mem_li)
            potentials.append(torch.flatten(mem_li, start_dim=1)) # [B, 64*64]
            
        # 모든 타임스텝의 전위 기록 병합: [B, 15, 4096]
        all_time_potentials = torch.stack(potentials, dim=1) 
        all_time_potentials = self.dropout(all_time_potentials)
        
        # DP-Spiking Decoder로 전위차 기반 분류
        out = self.decoder(all_time_potentials)
        return out 

# =====================================================================
# 4. 학습 파이프라인
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list = []
sstl_acc_list = []

print("\n" + "="*50)
print("🧠 리만 기하학 + HR-SNN (DP Decoder 장착) 5-Fold 학습 시작")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = load_graph_data(global_train_subjs)
    if L_train is None: continue
    train_loader = DataLoader(TensorDataset(L_train, X_train, y_train), batch_size=128, shuffle=True)
    
    net = RiemannianGraph_HRSNN(num_steps=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    net.train()
    for epoch in range(30): # 복잡해진 모델을 위해 에폭 약간 증가
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
            sstl_net = RiemannianGraph_HRSNN(num_steps=15).to(device)
            sstl_net.load_state_dict(global_model_state)
            
            # Classifier 부분만 파인튜닝 (SSTL 원칙)
            for name, param in sstl_net.named_parameters():
                if 'decoder.classifier' not in name: param.requires_grad = False
                    
            sstl_optimizer = torch.optim.Adam(sstl_net.decoder.classifier.parameters(), lr=5e-4)
            
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