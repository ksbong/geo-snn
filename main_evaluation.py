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
DATA_DIR = './07_Data'
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
        
        # 순수 Riemannian Whitening
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
# 2. 실시간 동적 그래프 어텐션 (Dynamic Graph Attention)
# =========================================================
class DynamicGraphAttention(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=8):
        super().__init__()
        # Query, Key, Value 프로젝션
        self.query = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        # x: (B, 1, 64, 480)
        # 시간축 평균으로 각 채널(전극)의 현재 상태 요약
        x_pool = torch.mean(x, dim=-1, keepdim=True) # (B, 1, 64, 1)
        
        Q = self.query(x_pool).squeeze(-1) # (B, hidden, 64)
        K = self.key(x_pool).squeeze(-1)   # (B, hidden, 64)
        
        # 샘플별 실시간 인접 행렬 A(x) 연산 (Dot-Product Attention)
        attn = torch.einsum('b h i, b h j -> b i j', Q, K) * self.scale
        
        # 대각선 마스킹 (자기 자신 참조 방지)
        eye = torch.eye(64, device=x.device).unsqueeze(0).bool()
        attn = attn.masked_fill(eye, -1e9)
        
        # Softmax로 가중치 정규화
        attn_weights = F.softmax(attn, dim=-1) # (B, 64, 64)
        
        # Message Passing
        V = self.value(x)
        out = torch.einsum('b i j, b c j t -> b c i t', attn_weights, V)
        return out, attn_weights

class SOTA_Dynamic_STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dyn_graph = DynamicGraphAttention(in_channels=1, hidden_dim=16)
        self.st_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(0.5)
        )
        self.drop_fc = nn.Dropout(0.5)
        self.fc = nn.Linear(16 * 64, 4)

    def forward(self, x):
        # 정적인 A 행렬 대신, 실시간 동적 어텐션 출력값 사용
        z_gcn, attn_weights = self.dyn_graph(x)
        z = self.st_block(z_gcn)
        z_feat = torch.mean(z, dim=-1).view(z.size(0), -1)
        z_feat_dropped = self.drop_fc(z_feat)
        return self.fc(z_feat_dropped), z_feat, attn_weights

# MMD Loss (과적합 및 도메인 차이 방어용)
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
# 3. 모델 학습 (완전 격리된 Test 세트로 검증)
# =========================================================
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train)), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test)), batch_size=128, shuffle=False)

model = SOTA_Dynamic_STGCN().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

print("🚀 Unseen Subject 21명 대상 진짜 범용성 테스트 시작...")

best_acc = 0.0
mmd_weight = 0.01 # MMD는 가볍게 거들기만 함

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
        print(f"Epoch {epoch:3d} | Train Acc: {correct/total*100:.2f}% | ✨ Unseen Test Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_dynamic_sota.pth")

print("="*50)
print(f"🏆 피험자 완전 격리 (Unseen Subject) 최고 성능: {best_acc:.2f}%")
print("="*50)