import os
import gc
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. 105명 전체 데이터 로딩 및 메모리 최적화 리만 정렬
# =========================================================
DATA_DIR = './07_Data' # 캐글 경로로 맞게 수정해
# PhysioNet 109명 중 불량 데이터 4명(88, 92, 100, 104) 제외한 105명
bad_subjects = [88, 92, 100, 104]
subjects = [f'S{i:03d}' for i in range(1, 110) if i not in bad_subjects]

all_x_aligned, all_y = [], []

print("⏳ 105명 전체 데이터 로딩 및 리만 정렬 시작 (메모리 최적화)...")
for s_idx, s in enumerate(subjects):
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
        if len(ep) > 0:
            subj_epochs.append(ep)
    
    if len(subj_epochs) == 0: continue
    
    epochs = mne.concatenate_epochs(subj_epochs, verbose=False)
    X = epochs.get_data(copy=True) * 1e6
    y = epochs.events[:, 2]
    
    # 🧠 피험자별 순수 Riemannian Whitening (개인차 영점 조절)
    R_i = np.mean([np.cov(x) for x in X], axis=0) + np.eye(64) * 1e-4
    P_i = la.inv(la.sqrtm(R_i))
    X_aligned = np.array([P_i @ x for x in X])
    
    all_x_aligned.append(X_aligned)
    all_y.extend(y)
    
    if (s_idx + 1) % 10 == 0:
        print(f"[{s_idx + 1}/{len(subjects)}] 처리 완료... RAM 정리 중")
        gc.collect()

X_final = np.expand_dims(np.concatenate(all_x_aligned), 1) # (B, 1, 64, 480)
Y_final = np.array(all_y)

print(f"✅ 데이터 로딩 완료! 총 샘플 수: {X_final.shape[0]}개")

# 마스터 인접 행렬 뼈대 추출 (메모리 위해 2000개만 샘플링해서 계산)
sample_idx = np.random.choice(len(X_final), 2000, replace=False)
A_init = torch.tensor(np.abs(np.corrcoef(np.mean(X_final[sample_idx, 0], axis=0))), dtype=torch.float32)

# =========================================================
# 2. 모델 정의 및 MMD Loss (수학적 수술 완료된 완벽본)
# =========================================================
def mmd_loss(x, y, bandwidths=[0.5, 1.0, 2.0]):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = torch.zeros_like(xx), torch.zeros_like(yy), torch.zeros_like(zz)
    for a in bandwidths:
        XX += torch.exp(-0.5 * dxx / a); YY += torch.exp(-0.5 * dyy / a); XY += torch.exp(-0.5 * dxy / a)
    return torch.mean(XX + YY - 2. * XY)

class SOTA_STGCN_SNN(nn.Module):
    def __init__(self, A_init):
        super().__init__()
        self.A_learnable = nn.Parameter(A_init.clone(), requires_grad=True)
        self.st_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(0.5) # 과적합 방지
        )
        self.drop_fc = nn.Dropout(0.5)
        self.fc = nn.Linear(16 * 64, 4)
        self.register_buffer('eye_mask', torch.eye(64).bool()) # 대각선 차단 마스크

    def forward(self, x):
        A_masked = self.A_learnable.masked_fill(self.eye_mask, 0.0)
        z = torch.einsum('vw, bfwt -> bfvt', A_masked, x)
        z = self.st_block(z)
        z_feat = torch.mean(z, dim=-1).view(z.size(0), -1)
        z_feat_dropped = self.drop_fc(z_feat)
        return self.fc(z_feat_dropped), z_feat

# =========================================================
# 3. 데이터 로더 세팅 (데이터가 많으니 Batch Size 128로 상향)
# =========================================================
idx_tr, idx_te = train_test_split(np.arange(len(X_final)), test_size=0.2, stratify=Y_final, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_final[idx_tr], dtype=torch.float32), torch.tensor(Y_final[idx_tr])), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_final[idx_te], dtype=torch.float32), torch.tensor(Y_final[idx_te])), batch_size=128, shuffle=False)

model = SOTA_STGCN_SNN(A_init).cuda() # GPU 연산 필수
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

print("🚀 SOTA 도전을 위한 105명 Global Training 시작...")

mmd_weight = 0.05 
l1_lambda = 1e-4  

best_acc = 0.0
patience = 30 # 30에폭 동안 최고점 갱신 못하면 조기 종료
trigger_times = 0

for epoch in range(1, 201):
    model.train()
    correct, total = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        logits, z_feat = model(xb)
        
        loss_cls = F.cross_entropy(logits, yb)
        
        half_idx = z_feat.size(0) // 2
        if half_idx > 0: loss_mmd = mmd_loss(z_feat[:half_idx], z_feat[half_idx:])
        else: loss_mmd = 0.0
            
        loss_l1 = l1_lambda * torch.norm(model.A_learnable, 1)
        loss = loss_cls + mmd_weight * loss_mmd + loss_l1
        
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
                logits, _ = model(xb)
                test_correct += torch.argmax(logits, 1).eq(yb).sum().item()
                test_total += yb.size(0)
                
        val_acc = test_correct / test_total * 100
        print(f"Epoch {epoch:3d} | Train Acc: {correct/total*100:.2f}% | Test Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            # SOTA 달성 시 모델 가중치 저장!
            torch.save(model.state_dict(), "best_sota_model.pth")
        else:
            trigger_times += 5
            
        if trigger_times >= patience:
            print(f"🚨 조기 종료 발동! 최고 Test Acc: {best_acc:.2f}%")
            break

print("="*50)
print(f"🏆 최종 Global Training 최고 성능: {best_acc:.2f}%")
print("="*50)