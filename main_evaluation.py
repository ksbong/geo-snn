import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. Data Loading (빠른 테스트를 위해 10명, L vs R만 사용)
# =========================================================
base = './'
DATA_DIR = os.path.join(base, '07_Data') 
subjects = [f'S{i:03d}' for i in range(1, 11)] # 딱 10명만
TARGET_SFREQ = 160.0 

def load_data():
    X_list, y_list = [], []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        if not os.path.exists(subj_dir): continue
            
        epochs_list = []
        for run in ['R04', 'R08', 'R12']: # Left vs Right runs
            path = os.path.join(subj_dir, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            
            try:
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
                raw.rename_channels(lambda x: x.strip('.'))
                raw.filter(l_freq=8.0, h_freq=30.0, verbose=False) # Mu/Beta band
                
                evs, ed = mne.events_from_annotations(raw, verbose=False)
                t1, t2 = ed.get('T1'), ed.get('T2')
                if t1 is None: continue
                
                e = evs.copy()
                e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 0, 1 # L: 0, R: 1
                    
                ep = mne.Epochs(raw, e, {'L':0, 'R':1}, tmin=0.0, tmax=4.0, 
                                baseline=None, preload=True, verbose=False)
                if len(ep) > 0: epochs_list.append(ep)
            except Exception: continue
                
        if epochs_list:
            subj_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
            X_list.append(subj_epochs.get_data() * 1e6)
            y_list.append(subj_epochs.events[:, 2])
            
    return np.concatenate(X_list), np.concatenate(y_list), subj_epochs.info

print("⏳ 10명 데이터 로딩 중 (Left vs Right)...")
X, y, info = load_data()
B, C, T = X.shape
print(f"데이터 로드 완료: {B} Trials, {C} Channels, {T} Timesteps")

# =========================================================
# 2. 🧠 Feature Extraction (제안한 아키텍처 구조의 수학적 구현)
# =========================================================
# (A) Baseline Feature: Raw EEG Variance (단순 분산)
feat_raw = np.var(X, axis=2) # (B, C)

# (B) Proposed Feature: Graph Spectral Delta
# 초기 상태의 가중치 학습 없이, 데이터의 경험적 공분산을 바탕으로 GFT 수행
def extract_spectral_delta(X_batch):
    feats = []
    for x in X_batch: # x shape: (C, T)
        # 1. Empirical Covariance -> Adjacency Matrix (A)
        x_centered = x - np.mean(x, axis=1, keepdims=True)
        cov = np.cov(x_centered)
        A = np.abs(cov) # 연결 강도
        
        # 2. Graph Laplacian (L = D - A)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        # 3. Eigendecomposition
        vals, U = np.linalg.eigh(L)
        
        # 4. Graph Fourier Transform
        x_hat = U.T @ x
        
        # 5. Spectral Delta (시간적 변화량)
        delta_x = np.diff(x_hat, axis=1) # (C, T-1)
        
        # Feature: 스펙트럼 변화량의 분산(Power)
        feats.append(np.var(delta_x, axis=1))
    return np.array(feats)

print("⚙️ 제안된 Graph Spectral Delta Feature 추출 중...")
feat_proposed = extract_spectral_delta(X) # (B, C)

# =========================================================
# 3. 📊 통계적 검증 및 평가 (Fisher Ratio & LDA)
# =========================================================
def calc_fisher_ratio(features, labels):
    # features: (B, C)
    idx_0, idx_1 = np.where(labels == 0)[0], np.where(labels == 1)[0]
    
    mu_0, mu_1 = np.mean(features[idx_0], axis=0), np.mean(features[idx_1], axis=0)
    var_0, var_1 = np.var(features[idx_0], axis=0), np.var(features[idx_1], axis=0)
    
    fr = (mu_0 - mu_1)**2 / (var_0 + var_1 + 1e-6)
    return fr

fr_raw = calc_fisher_ratio(feat_raw, y)
fr_proposed = calc_fisher_ratio(feat_proposed, y)

# Linear Classifier Test (5-Fold CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LinearDiscriminantAnalysis()

acc_raw = cross_val_score(clf, feat_raw, y, cv=cv).mean() * 100
acc_proposed = cross_val_score(clf, feat_proposed, y, cv=cv).mean() * 100

# =========================================================
# 4. 🎨 시각화 (논문용 Feature Test 결과)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Classifier Performance Comparison
axes[0].bar(['Raw Variance', 'Graph Spectral Delta\n(Proposed)'], 
            [acc_raw, acc_proposed], color=['gray', 'blue'], alpha=0.7)
axes[0].set_ylabel('LDA Accuracy (%)')
axes[0].set_title('Linear Separability Test', fontsize=14)
axes[0].set_ylim(40, max(acc_raw, acc_proposed) + 10)
for i, v in enumerate([acc_raw, acc_proposed]):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Plot 2: Fisher Ratio Topomap (Raw)
axes[1].set_title('Fisher Ratio: Raw EEG', fontsize=14)
mne.viz.plot_topomap(fr_raw, info, axes=axes[1], cmap='Reds', show=False, contours=0)

# Plot 3: Fisher Ratio Topomap (Proposed)
axes[2].set_title('Fisher Ratio: Graph Spectral Delta', fontsize=14)
mne.viz.plot_topomap(fr_proposed, info, axes=axes[2], cmap='Reds', show=False, contours=0)

plt.tight_layout()
plt.show()