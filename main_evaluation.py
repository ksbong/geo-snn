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

# =====================================================================
# 1. 기하학적 특징 추출
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

# =====================================================================
# 2. 전처리 & 텐서 저장 (Sliding Window로 Temporal Dynamics 추출)
# =====================================================================
DATA_DIR = './raw_data/files/'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './raw_data/files'

SAVE_DIR = './processed_tensors_seq'
os.makedirs(SAVE_DIR, exist_ok=True)

exclude_subjects = ['S088', 'S092', 'S100', 'S104']
all_subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in exclude_subjects]

def process_and_save_subject(subj):
    save_path_X = f"{SAVE_DIR}/{subj}_X.pt"
    save_path_y = f"{SAVE_DIR}/{subj}_y.pt"
    if os.path.exists(save_path_X): return
        
    runs_hands = ['R04', 'R08', 'R12']
    runs_feet = ['R06', 'R10', 'R14']
    epochs_list = []
    
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
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

        # 전체 3초 신호에 대해 한 번에 Hardy Space 변환 (FFT 아티팩트 방지)
        hardy_signals = project_to_hardy_space(data, sfreq)
        envelopes = np.abs(hardy_signals)
        
        # 💡 Sliding Window 설정 (1초 길이, 0.5초 겹침 -> 총 5 시퀀스)
        win_len = 160
        stride = 80
        num_windows = 5
        
        X_covs_seq = []
        for w in range(num_windows):
            start = w * stride
            end = start + win_len
            env_win = envelopes[:, :, start:end]
            covs = np.array([compute_spd_cov(env) for env in env_win])
            X_covs_seq.append(covs)
            
        X_covs_seq = np.stack(X_covs_seq, axis=1) # Shape: (Epochs, 5, 64, 64)

        # 피험자별 Rest 상태 매핑 기준점 계산 (전체 윈도우 평균)
        rest_covs = X_covs_seq[labels == 0]
        p_rest = np.mean(rest_covs, axis=(0, 1)) 
        p_rest_sqrt = sqrtm(p_rest).real
        p_rest_inv_sqrt = inv(p_rest_sqrt)

        # Tangent Space 매핑
        features = []
        for ep_idx in range(X_covs_seq.shape[0]):
            ep_feats = []
            for w in range(num_windows):
                inner = p_rest_inv_sqrt @ X_covs_seq[ep_idx, w] @ p_rest_inv_sqrt
                tangent = p_rest_sqrt @ logm(inner).real @ p_rest_sqrt
                ep_feats.append(tangent)
            features.append(ep_feats)
        
        features = np.array(features)
        
        # 스케일링
        scale_factor = np.max(np.abs(features))
        if scale_factor > 0: features = features / scale_factor

        # SNN Spatial Conv를 위해 채널 차원 추가 -> (Epochs, 5, 1, 64, 64)
        torch.save(torch.tensor(features, dtype=torch.float32).unsqueeze(2), save_path_X)
        torch.save(torch.tensor(labels, dtype=torch.long), save_path_y)
        
    except Exception as e:
        pass

print("🚀 1단계: Temporal Dynamics 추출 및 텐서 저장 시작")
for subj in tqdm(all_subjects):
    process_and_save_subject(subj)

# =====================================================================
# 3. 데이터 로드 및 모델 구조 (Spatio-Temporal Injection)
# =====================================================================
def load_subjects_data(subject_list):
    X_list, y_list = [], []
    for subj in subject_list:
        if os.path.exists(f"{SAVE_DIR}/{subj}_X.pt"):
            X_list.append(torch.load(f"{SAVE_DIR}/{subj}_X.pt", weights_only=True))
            y_list.append(torch.load(f"{SAVE_DIR}/{subj}_y.pt", weights_only=True))
    if not X_list: return None, None
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

class RiemannianSNN(nn.Module):
    def __init__(self, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25) 
        
        # 💡 개선 1: 네트워크 용량 증대 및 Dropout 추가
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        # 뉴런의 망각률(beta)을 다양하게 분리 (Hybrid Response 흉내)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(0.3) # 피험자 간 과적합 방지
        self.pool2 = nn.MaxPool2d(2)
        
        # 💡 개선 2: Hybrid Response 메커니즘 
        # 빠른 반응 뉴런(beta=0.5)과 느린 기억 뉴런(beta=0.95)을 섞음
        self.lif2_fast = snn.Leaky(beta=0.5, spike_grad=spike_grad)
        self.lif2_slow = snn.Leaky(beta=0.95, spike_grad=spike_grad)
        
        self.flatten = nn.Flatten()
        
        # 💡 개선 3: 시간 정보를 평균 내지 않고, 통째로 분류기에 넣기 위한 차원 확장
        # 64채널 * 16 * 16 (공간) * 15 (시간) = 분류기가 시간 흐름 전체를 봄
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16 * 15, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # x shape: [Batch, 5, 1, 64, 64]
        x = x * 10.0 
        
        mem1 = self.lif1.init_leaky()
        mem2_f = self.lif2_fast.init_leaky()
        mem2_s = self.lif2_slow.init_leaky()
        
        spk2_rec = []
        steps_per_window = self.num_steps // 5
        
        for step in range(self.num_steps):
            w_idx = step // steps_per_window
            if w_idx >= 5: w_idx = 4
            
            current_x = x[:, w_idx] 
            
            # Layer 1
            cur1 = self.feature_extractor(current_x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2 (Hybrid Response)
            cur2 = self.drop2(self.pool2(self.bn2(self.conv2(spk1))))
            
            spk2_f, mem2_f = self.lif2_fast(cur2, mem2_f)
            spk2_s, mem2_s = self.lif2_slow(cur2, mem2_s)
            
            # Fast와 Slow 뉴런의 스파이크를 더해서 풍부한 시간 특징 생성
            spk2_combined = spk2_f + spk2_s 
            spk2_rec.append(self.flatten(spk2_combined))
            
        # [Time, Batch, Features] -> [Batch, Time * Features]
        # 시간축 데이터를 믹서기(mean)에 갈지 않고, 길게 쭉 펴서 공간+시간 특징을 보존함
        all_time_features = torch.stack(spk2_rec).transpose(0, 1).contiguous()
        flat_spatio_temporal = all_time_features.view(all_time_features.size(0), -1)
        
        out = self.classifier(flat_spatio_temporal)
        return out # 주의: 이제 m_mean.max(1)이 아니라 그냥 out.max(1)로 바로 CrossEntropy에 넣으면 됨
# =====================================================================
# 4. 5-Fold 교차 검증 (Global Training -> True Global Acc -> SSTL)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list = []
sstl_acc_list = []

print("\n" + "="*50)
print("🌍 본격적인 5-Fold 논문 파이프라인 가동 (시간 정보 포함)")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    # 1. Global Training
    X_train, y_train = load_subjects_data(global_train_subjs)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    
    net = RiemannianSNN(num_steps=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    net.train()
    for epoch in range(20): # 빠른 검증을 위해 20 에폭 (필요시 30으로 증가)
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            m_mean = net(data).mean(dim=0)
            loss = criterion(m_mean, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 2. 진짜 Global Test Accuracy (Unseen Subjects) 측정
    X_unseen, y_unseen = load_subjects_data(global_test_subjs)
    unseen_loader = DataLoader(TensorDataset(X_unseen, y_unseen), batch_size=128, shuffle=False)
    
    net.eval()
    unseen_correct, unseen_total = 0, 0
    with torch.no_grad():
        for data, targets in unseen_loader:
            data, targets = data.to(device), targets.to(device)
            m_mean = net(data).mean(dim=0)
            _, predicted = m_mean.max(1)
            unseen_total += targets.size(0)
            unseen_correct += (predicted == targets).sum().item()
            
    true_global_acc = 100 * unseen_correct / unseen_total
    global_acc_list.append(true_global_acc)
    print(f"🔥 Fold {fold + 1} 진짜 Global Test Acc (p0): {true_global_acc:.2f}%")
    
    # 3. SSTL 진행
    global_model_state = net.state_dict()
    fold_sstl_accs = []
    
    for subj in global_test_subjs:
        X_sub, y_sub = load_subjects_data([subj])
        if X_sub is None: continue
        
        kf_sub = KFold(n_splits=4, shuffle=True, random_state=42)
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in kf_sub.split(X_sub):
            sstl_net = RiemannianSNN(num_steps=15).to(device)
            sstl_net.load_state_dict(global_model_state)
            
            # 파인튜닝: Classifier만 열어줌
            for name, param in sstl_net.named_parameters():
                if 'classifier' not in name: param.requires_grad = False
                    
            sstl_optimizer = torch.optim.Adam(sstl_net.classifier.parameters(), lr=5e-4)
            
            sub_train_loader = DataLoader(TensorDataset(X_sub[sub_train_idx], y_sub[sub_train_idx]), batch_size=32, shuffle=True)
            sub_test_loader = DataLoader(TensorDataset(X_sub[sub_test_idx], y_sub[sub_test_idx]), batch_size=32, shuffle=False)
            
            sstl_net.train()
            for _ in range(10): # 💡 개인이 적응할 수 있도록 에폭을 5->10으로 살짝 늘림
                for data, targets in sub_train_loader:
                    data, targets = data.to(device), targets.to(device)
                    loss = criterion(sstl_net(data).mean(dim=0), targets)
                    sstl_optimizer.zero_grad()
                    loss.backward()
                    sstl_optimizer.step()
                    
            sstl_net.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for data, targets in sub_test_loader:
                    data, targets = data.to(device), targets.to(device)
                    _, predicted = sstl_net(data).mean(dim=0).max(1)
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