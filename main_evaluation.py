import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. Data Loading (10명, 4-Class)
# =========================================================
base = './'
DATA_DIR = os.path.join(base, '07_Data') 
subjects = [f'S{i:03d}' for i in range(1, 11)] 

# 기존 코드의 L, R, H, F 세팅 유지
runs_h, runs_f = ['R04','R08','R12'], ['R06','R10','R14']
TARGET_SFREQ = 160.0 

def load_data():
    X_list, y_list = [], []
    info = None
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        if not os.path.exists(subj_dir): continue
            
        epochs_list = []
        for run in runs_h + runs_f:
            path = os.path.join(subj_dir, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            
            try:
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
                
                # 🔥 에러 해결: 표준 10-05 몽타주(전극 위치) 씌우기
                mne.datasets.eegbci.standardize(raw) 
                montage = mne.channels.make_standard_montage('standard_1005')
                raw.set_montage(montage, match_case=False, on_missing='ignore')
                
                raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
                
                evs, ed = mne.events_from_annotations(raw, verbose=False)
                t1, t2 = ed.get('T1'), ed.get('T2')
                if t1 is None: continue
                
                e = evs.copy()
                if run in runs_h:
                    e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 0, 1 # L: 0, R: 1
                else:
                    e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 2, 3 # H: 2, F: 3
                    
                event_id = {'L':0, 'R':1} if run in runs_h else {'H':2, 'F':3}
                
                # 🔥 HR-SNN 논문 반영: tmax를 3.0으로 설정하여 앞 3초만 사용
                ep = mne.Epochs(raw, e, event_id, tmin=0.0, tmax=3.0, 
                                baseline=None, preload=True, verbose=False)
                if len(ep) > 0: 
                    epochs_list.append(ep)
                    if info is None: info = ep.info # Topomap용 info 저장
            except Exception: continue
                
        if epochs_list:
            subj_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
            X_list.append(subj_epochs.get_data() * 1e6)
            y_list.append(subj_epochs.events[:, 2])
            
    return np.concatenate(X_list), np.concatenate(y_list), info

print("⏳ 10명 데이터 로딩 중 (4-Class)...")
X, y, info = load_data()
B, C, T = X.shape
print(f"데이터 로드 완료: {B} Trials, {C} Channels, {T} Timesteps")

# =========================================================
# 2. 🧠 Feature Extraction (Graph Spectral Delta)
# =========================================================
# (A) Baseline
feat_raw = np.var(X, axis=2) # (B, C)

# (B) Proposed Feature
def extract_spectral_delta(X_batch):
    feats = []
    for x in X_batch: 
        x_centered = x - np.mean(x, axis=1, keepdims=True)
        cov = np.cov(x_centered)
        A = np.abs(cov) 
        
        D = np.diag(np.sum(A, axis=1))
        # 수치적 안정성을 위해 대각선에 작은 값 추가
        L = D - A + np.eye(len(A)) * 1e-4
        
        vals, U = np.linalg.eigh(L)
        x_hat = U.T @ x
        
        delta_x = np.diff(x_hat, axis=1) 
        feats.append(np.var(delta_x, axis=1))
    return np.array(feats)

print("⚙️ 제안된 Graph Spectral Delta Feature 추출 중...")
feat_proposed = extract_spectral_delta(X)

# =========================================================
# 3. 📊 통계적 검증 (4-Class ANOVA F-value & LDA)
# =========================================================
# 4-Class 분별력을 확인하기 위해 F-value 계산
f_val_raw, _ = f_classif(feat_raw, y)
f_val_proposed, _ = f_classif(feat_proposed, y)

# Linear Classifier Test (5-Fold CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LinearDiscriminantAnalysis()

acc_raw = cross_val_score(clf, feat_raw, y, cv=cv).mean() * 100
acc_proposed = cross_val_score(clf, feat_proposed, y, cv=cv).mean() * 100

# =========================================================
# 4. 🎨 시각화
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: LDA Accuracy
axes[0].bar(['Raw Variance', 'Graph Spectral Delta'], 
            [acc_raw, acc_proposed], color=['gray', 'blue'], alpha=0.7)
axes[0].set_ylabel('LDA Accuracy (%)')
axes[0].set_title('4-Class Linear Separability Test', fontsize=14)
axes[0].set_ylim(min(acc_raw, acc_proposed) - 5, max(acc_raw, acc_proposed) + 10)
for i, v in enumerate([acc_raw, acc_proposed]):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Plot 2: Discriminative Power Topomap (Raw)
axes[1].set_title('Discriminative Power (F-value): Raw', fontsize=14)
mne.viz.plot_topomap(f_val_raw, info, axes=axes[1], cmap='Reds', show=False, contours=0)

# Plot 3: Discriminative Power Topomap (Proposed)
axes[2].set_title('Discriminative Power (F-value): Graph Spectral Delta', fontsize=14)
mne.viz.plot_topomap(f_val_proposed, info, axes=axes[2], cmap='Reds', show=False, contours=0)

plt.tight_layout()
plt.show()
print("✅ 검증 완료!")