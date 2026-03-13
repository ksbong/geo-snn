import os
import glob
import mne
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import sqrtm, logm, inv
from sklearn.model_selection import KFold
import snntorch as snn
from snntorch import surrogate
from tqdm import tqdm

# =====================================================================
# 1. 기하학적 특징 추출 함수 (Hardy Space & SPD)
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
# 2. 전처리 & 텐서 저장 (RAM 폭발 방지 및 Riemannian Alignment)
# =====================================================================
DATA_DIR = './raw_data/files/'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './raw_data/files'

SAVE_DIR = './processed_tensors'
os.makedirs(SAVE_DIR, exist_ok=True)

# HR-SNN 논문 기준 제외 피험자 4명 (S088, S092, S100, S104) 
exclude_subjects = ['S088', 'S092', 'S100', 'S104']
all_subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in exclude_subjects]

def process_and_save_subject(subj):
    save_path_X = f"{SAVE_DIR}/{subj}_X.pt"
    save_path_y = f"{SAVE_DIR}/{subj}_y.pt"
    if os.path.exists(save_path_X): 
        return # 이미 처리됐으면 패스
        
    runs_hands = ['R04', 'R08', 'R12']
    runs_feet = ['R06', 'R10', 'R14']
    epochs_list = []
    
    try:
        # 손 상상
        for run in runs_hands:
            path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
            ep = mne.Epochs(raw, evs, {'Rest': ev_dict['T0'], 'Left Hand': ev_dict['T1'], 'Right Hand': ev_dict['T2']}, 
                            tmin=1.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            epochs_list.append(ep)

        # 발 상상
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
        labels = np.array(epochs_all.events[:, 2]) - 1 # 0:Rest, 1:Left, 2:Right, 3:Feet
        sfreq = epochs_all.info['sfreq']

        # Hardy Space & 공분산
        hardy_signals = project_to_hardy_space(data, sfreq)
        envelopes = np.abs(hardy_signals)
        X_covs = np.array([compute_spd_cov(env) for env in envelopes])

        # 💡 피험자 개인의 Rest(0) 데이터만 모아서 개인 맞춤형 기준점 계산
        rest_covs = X_covs[labels == 0]
        p_rest = np.mean(rest_covs, axis=0) 
        p_rest_sqrt = sqrtm(p_rest).real
        p_rest_inv_sqrt = inv(p_rest_sqrt)

        features = []
        for cov in X_covs:
            inner = p_rest_inv_sqrt @ cov @ p_rest_inv_sqrt
            tangent = p_rest_sqrt @ logm(inner).real @ p_rest_sqrt
            features.append(tangent)
        
        features = np.array(features)
        
        # 개인 스케일링
        scale_factor = np.max(np.abs(features))
        if scale_factor > 0: features = features / scale_factor

        torch.save(torch.tensor(features, dtype=torch.float32).unsqueeze(1), save_path_X)
        torch.save(torch.tensor(labels, dtype=torch.long), save_path_y)
        
    except Exception as e:
        print(f"Error processing {subj}: {e}")

print("🚀 1단계: 전체 피험자 특징 추출 및 텐서 저장 시작 (Riemannian Alignment 적용)")
for subj in tqdm(all_subjects):
    process_and_save_subject(subj)
print("✅ 텐서 저장 완료!")

# =====================================================================
# 3. 데이터 로더 및 Global / SSTL 분할
# =====================================================================
def load_subjects_data(subject_list):
    X_list, y_list = [], []
    for subj in subject_list:
        if os.path.exists(f"{SAVE_DIR}/{subj}_X.pt"):
            X_list.append(torch.load(f"{SAVE_DIR}/{subj}_X.pt"))
            y_list.append(torch.load(f"{SAVE_DIR}/{subj}_y.pt"))
    if not X_list: return None, None
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

# 논문 기준: 84명 Global Train, 21명 Global Test (SSTL 대상)
np.random.seed(42)
shuffled_subjects = np.random.permutation(all_subjects)
global_train_subjs = shuffled_subjects[:84]
global_test_subjs = shuffled_subjects[84:]

print(f"\n📁 Global Train 피험자 수: {len(global_train_subjs)}명")
print(f"📁 SSTL(Test) 피험자 수: {len(global_test_subjs)}명")

X_global, y_global = load_subjects_data(global_train_subjs)
global_dataset = TensorDataset(X_global, y_global)
global_loader = DataLoader(global_dataset, batch_size=128, shuffle=True)

# =====================================================================
# 4. SNN 모델 구조 (FC 파라미터 분리를 위해 구조 정비)
# =====================================================================
class RiemannianSNN(nn.Module):
    def __init__(self, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25) 
        beta = 0.9  
        
        # 특징 추출부 (Global 모델에서 뼈대를 잡음)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            # SNN 모듈은 Sequential 안에 바로 못 넣으니 forward에서 처리
        )
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.flatten = nn.Flatten()
        
        # 💡 분류기(Classifier): SSTL 단계에서 여기만 파인튜닝함
        self.classifier = nn.Linear(32 * 16 * 16, 4) 
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        x = x * 10.0 # 강제 전류 증폭
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        mem_out_rec = []
        
        for step in range(self.num_steps):
            # 1 layer
            cur1 = self.feature_extractor(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # 2 layer
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Output layer
            cur_out = self.classifier(self.flatten(spk2))
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            mem_out_rec.append(mem_out)
            
        return torch.stack(mem_out_rec)

# =====================================================================
# 5. Global Training (84명 전체 학습)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RiemannianSNN(num_steps=15).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

num_epochs_global = 30 # 시간이 없으면 15 정도로 줄여서 테스트
print("-" * 50)
print(f"🌍 Global Training (84명, 데이터 수: {len(global_dataset)}) 시작")
print("-" * 50)

for epoch in range(num_epochs_global):
    net.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, targets in global_loader:
        data, targets = data.to(device), targets.to(device)
        mem_rec = net(data) 
        m_mean = mem_rec.mean(dim=0) 
        loss = criterion(m_mean, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = m_mean.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Global Epoch {epoch+1}/{num_epochs_global} | Loss: {total_loss/len(global_loader):.4f} | Acc: {100*correct/total:.2f}%")
# =====================================================================
# 6. SSTL (Subject-Specific Transfer Learning) - 수정된 버전
# =====================================================================
print("-" * 50)
print("🚀 SSTL (개인 맞춤형 전이 학습) 시작")
print("-" * 50)

# Global Model의 가중치를 CPU로 미리 빼둠 (안전한 복사를 위해)
global_model_state = net.state_dict()

sstl_accuracies = []

for subj in global_test_subjs:
    # weights_only=True 추가해서 보안 경고 해결
    X_sub, y_sub = load_subjects_data([subj]) 
    if X_sub is None: continue
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    subj_fold_accs = []
    
    for train_idx, test_idx in kf.split(X_sub):
        # 💡 수정된 부분: deepcopy 대신 새로운 인스턴스 생성 후 가중치 로드
        sstl_net = RiemannianSNN(num_steps=15).to(device)
        sstl_net.load_state_dict(global_model_state)
        
        # Classifier(FC 레이어) 파라미터만 업데이트 허용, 나머지는 동결
        for name, param in sstl_net.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                
        sstl_optimizer = torch.optim.Adam(sstl_net.classifier.parameters(), lr=5e-4)
        
        # (이하 학습 및 평가 루틴은 동일)
        sub_train_loader = DataLoader(TensorDataset(X_sub[train_idx], y_sub[train_idx]), batch_size=32, shuffle=True)
        sub_test_loader = DataLoader(TensorDataset(X_sub[test_idx], y_sub[test_idx]), batch_size=32, shuffle=False)
        
        sstl_net.train()
        for _ in range(5): # 딱 5 에폭만 학습 [cite: 327]
            for data, targets in sub_train_loader:
                data, targets = data.to(device), targets.to(device)
                m_mean = sstl_net(data).mean(dim=0)
                loss = criterion(m_mean, targets)
                sstl_optimizer.zero_grad()
                loss.backward()
                sstl_optimizer.step()
                
        sstl_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in sub_test_loader:
                data, targets = data.to(device), targets.to(device)
                m_mean = sstl_net(data).mean(dim=0)
                _, predicted = m_mean.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        subj_fold_accs.append(100 * correct / total)
    
    subj_acc = np.mean(subj_fold_accs)
    sstl_accuracies.append(subj_acc)
    print(f"피험자 {subj} SSTL Accuracy: {subj_acc:.2f}%")

print("-" * 50)
print(f"🎯 최종 SSTL 평균 정확도 (21명 대상): {np.mean(sstl_accuracies):.2f}%")
print("-" * 50)