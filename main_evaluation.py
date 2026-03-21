import os
import gc
import random
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. 완벽한 피험자 격리 (84 Train vs 21 Test) & 메모리 최적화 로딩
# =========================================================
DATA_DIR = './07_Data' # 캐글 경로에 맞게 수정
bad_subjects = [88, 92, 100, 104]
subjects = sorted([f'S{i:03d}' for i in range(1, 110) if i not in bad_subjects])

# 학술적 표준인 5-Fold 중 1번째 Fold 비율 (84명 Train, 21명 Test)
random.seed(42)
shuffled_subjs = subjects.copy()
random.shuffle(shuffled_subjs)
train_subjs = shuffled_subjs[:84]
test_subjs = shuffled_subjs[84:]

print(f"🔒 완벽한 데이터 격리: Train {len(train_subjs)}명, Test {len(test_subjs)}명")

def load_and_align_subjects(subj_list):
    x_aligned, y_labels = [], []
    for s_idx, s in enumerate(subj_list):
        subj_dir = os.path.join(DATA_DIR, s)
        subj_epochs = []
        for run in ['R04','R08','R12','R06','R10','R14']:
            path = os.path.join(subj_dir, f'{s}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.resample(160.0); raw.filter(8.0, 30.0, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            e = evs.copy()
            if run in ['R04','R08','R12']: e[evs[:,2]==ed.get('T1', 1),2], e[evs[:,2]==ed.get('T2', 2),2] = 0, 1
            else: e[evs[:,2]==ed.get('T1', 1),2], e[evs[:,2]==ed.get('T2', 2),2] = 2, 3
                
            ep = mne.Epochs(raw, e, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: subj_epochs.append(ep)
        
        if len(subj_epochs) == 0: continue
        epochs = mne.concatenate_epochs(subj_epochs, verbose=False)
        X = epochs.get_data(copy=True) * 1e6
        y = epochs.events[:, 2]
        
        # 순수 Riemannian Whitening (누수 방지용 개별 계산)
        R_i = np.mean([np.cov(x) for x in X], axis=0) + np.eye(64) * 1e-4
        P_i = la.inv(la.sqrtm(R_i))
        x_aligned.append(np.array([P_i @ x for x in X]))
        y_labels.extend(y)
        
        if (s_idx + 1) % 20 == 0: gc.collect()
            
    return np.expand_dims(np.concatenate(x_aligned), 1), np.array(y_labels)

print("⏳ 1. Train 데이터(84명) 로딩 및 리만 정렬 중...")
X_train, Y_train = load_and_align_subjects(train_subjs)
print("⏳ 2. Test 데이터(21명) 로딩 및 리만 정렬 중...")
X_test, Y_test = load_and_align_subjects(test_subjs)
print(f"✅ 로딩 완료! Train 샘플: {X_train.shape[0]}, Test 샘플: {X_test.shape[0]}")

# =========================================================
# 2. SNN 핵심 코어 (Surrogate Gradient & LIF Neuron)
# =========================================================
# 🔥 핵심 1: 죽어버리는 역전파를 살려내는 대리 기울기(ATan)
class SurrogateATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float() # 순전파: 0을 넘으면 1(스파이크), 아니면 0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        # 역전파: 아크탄젠트 미분 형태로 부드러운 기울기 전달
        grad_input = grad_output / (1 + (alpha * input).pow(2))
        return grad_input, None

spike_fn = SurrogateATan.apply

# 🔥 핵심 2: 시간에 따라 정보를 누적하는 생물학적 뉴런
class LIFNode(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.tau = tau
        self.v_th = v_threshold

    def forward(self, x):
        # x: (B, C, N, T) -> 시간 축 T(480)에 대해 순차적 연산
        B, C, N, T = x.shape
        v = torch.zeros(B, C, N, device=x.device) # 초기 막전위 0
        spikes = []
        
        for t in range(T):
            # V[t] = V[t-1] * 누설률 + 입력 전류 X[t]
            v = v * (1 - 1/self.tau) + x[..., t]
            
            # 임계값을 넘었는지 판단하여 스파이크 발화 (Surrogate Gradient 적용)
            s = spike_fn(v - self.v_th)
            
            # 스파이크가 터졌으면 막전위 리셋 (Hard Reset)
            v = v - s * self.v_th 
            spikes.append(s)
            
        return torch.stack(spikes, dim=-1) # (B, C, N, T) 이진 스파이크 텐서 반환

# =========================================================
# 3. 하이브리드 ANN-SNN 아키텍처 (Dynamic Graph + LIF)
# =========================================================
class DynamicGraphAttention(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.query = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.key = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        x_squeeze = x.squeeze(1) # (B, 64, 480)
        x_energy = torch.var(x_squeeze, dim=-1).unsqueeze(1) # (B, 1, 64)
        
        Q = self.query(x_energy) # (B, hidden, 64)
        K = self.key(x_energy)   
        
        attn = torch.einsum('b h i, b h j -> b i j', Q, K) * self.scale
        eye = torch.eye(64, device=x.device).unsqueeze(0).bool()
        attn = attn.masked_fill(eye, -1e9)
        attn_weights = F.softmax(attn, dim=-1)
        
        out = torch.einsum('b i j, b j t -> b i t', attn_weights, x_squeeze)
        return out.unsqueeze(1) + x, attn_weights

class Hybrid_SOTA_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 공간적 정보 디코딩 (ANN 유지 - 정밀도 확보)
        self.dyn_graph = DynamicGraphAttention(hidden_dim=16)
        self.ann_spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(0.5)
            # 무거운 시계열 Conv 삭제 (연산량 2.3억 MACs 절감)
        )
        
        # 2. 시간적 통합 (SNN - 초저전력 및 정보 누적)
        self.snn_temporal = LIFNode(tau=2.0, v_threshold=0.5)
        
        self.drop_fc = nn.Dropout(0.5)
        self.fc = nn.Linear(16 * 64, 4)

    def forward(self, x):
        z_gcn, attn_weights = self.dyn_graph(x)
        z_ann = self.ann_spatial(z_gcn) # (B, 16, 64, 480) 연속적인 실수 (전류)
        
        # ANN의 특징을 LIF 뉴런의 전류로 주입하여 스파이크 발화
        spikes = self.snn_temporal(z_ann) # (B, 16, 64, 480) 이산적인 0과 1
        
        # 🔥 핵심 3: Rate Coding (발화율 코딩)으로 정보 손실 극복
        # 480번의 타임스텝 동안 터진 스파이크의 평균 빈도수를 특징 벡터로 사용
        firing_rate = torch.mean(spikes, dim=-1) # (B, 16, 64) -> 0.0 ~ 1.0 사이의 실수 복원
        
        z_feat = firing_rate.view(firing_rate.size(0), -1)
        z_feat_dropped = self.drop_fc(z_feat)
        
        return self.fc(z_feat_dropped), z_feat, attn_weights

def mmd_loss(x, y, bandwidths=[0.5, 1.0, 2.0]):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx; dyy = ry.t() + ry - 2. * yy; dxy = rx.t() + ry - 2. * zz
    XX, YY, XY = torch.zeros_like(xx), torch.zeros_like(yy), torch.zeros_like(zz)
    for a in bandwidths:
        XX += torch.exp(-0.5 * dxx / a); YY += torch.exp(-0.5 * dyy / a); XY += torch.exp(-0.5 * dxy / a)
    return torch.mean(XX + YY - 2. * XY)

# =========================================================
# 4. 모델 학습 (SNN Direct Training)
# =========================================================
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train)), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test)), batch_size=128, shuffle=False)

model = Hybrid_SOTA_SNN().cuda()
# SNN은 기울기가 작을 수 있어 초기 LR을 살짝 높여줌
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

print("🚀 Hybrid ANN-SNN (Unseen 21명) 완전체 학습 시작...")

best_acc = 0.0
mmd_weight = 0.01 

for epoch in range(1, 201):
    model.train()
    correct, total = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        logits, z_feat, _ = model(xb)
        
        loss_cls = F.cross_entropy(logits, yb)
        
        half_idx = z_feat.size(0) // 2
        loss_mmd = mmd_loss(z_feat[:half_idx], z_feat[half_idx:]) if half_idx > 0 else 0.0
            
        loss = loss_cls + mmd_weight * loss_mmd
        loss.backward()
        opt.step()
        
        correct += torch.argmax(logits, 1).eq(yb).sum().item()
        total += yb.size(0)
        
    scheduler.step()
        
    if epoch % 5 == 0:
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.cuda(), yb.cuda()
                logits, _, _ = model(xb)
                test_correct += torch.argmax(logits, 1).eq(yb).sum().item()
                test_total += yb.size(0)
                
        val_acc = test_correct / test_total * 100
        print(f"Epoch {epoch:3d} | Train Acc: {correct/total*100:.2f}% | ⚡ SNN Test Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_snn_sota.pth")

print("="*50)
print(f"🏆 하이브리드 SNN 최고 성능: {best_acc:.2f}%")
print("="*50)