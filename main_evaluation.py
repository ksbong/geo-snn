import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. Data Loading & Riemannian Alignment (105명 전체)
# =========================================================
base = './'
DATA_DIR = os.path.join(base, '07_Data') 

bad_subjects = ['S088', 'S089', 'S092', 'S100']
subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in bad_subjects]
runs_h, runs_f = ['R04','R08','R12'], ['R06','R10','R14']
TARGET_SFREQ = 160.0 

def process_subject(subj):
    subj_dir = os.path.join(DATA_DIR, subj)
    if not os.path.exists(subj_dir): return None
        
    epochs_list = []
    info = None
    for run in runs_h + runs_f:
        path = os.path.join(subj_dir, f'{subj}{run}.edf')
        if not os.path.exists(path): continue
        
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
            
            raw.rename_channels(lambda x: x.strip('.'))
            raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
            
            mne.datasets.eegbci.standardize(raw) 
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            t1, t2 = ed.get('T1'), ed.get('T2')
            if t1 is None: continue
            
            e = evs.copy()
            if run in runs_h:
                e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 0, 1
            else:
                e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 2, 3
                
            event_id = {'L':0, 'R':1} if run in runs_h else {'H':2, 'F':3}
            
            ep = mne.Epochs(raw, e, event_id, tmin=0.0, tmax=3.0, 
                            baseline=None, preload=True, verbose=False)
            if len(ep) > 0: 
                epochs_list.append(ep)
                if info is None: info = ep.info 
        except Exception: continue
            
    if not epochs_list: return None
    subj_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    X = subj_epochs.get_data() * 1e6
    y = subj_epochs.events[:, 2]
    
    # 🔥 핵심 Novelty: Riemannian Alignment (개인차 제거) 🔥
    covs = []
    for x in X:
        xc = x - np.mean(x, axis=1, keepdims=True)
        covs.append(np.cov(xc))
    
    # 피험자의 기하학적 기준점(Reference Matrix R) 계산
    R = np.mean(covs, axis=0) 
    R = R + np.eye(R.shape[0]) * 1e-4
    
    # R^{-1/2} 계산 (다양체의 원점으로 끌어오는 변환 행렬)
    vals, vecs = la.eigh(R)
    r_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, a_min=1e-6, a_max=None))) @ vecs.T
    
    # 모든 트라이얼을 원점으로 정렬
    X_aligned = np.zeros_like(X)
    for i in range(len(X)):
        X_aligned[i] = r_inv_sqrt @ X[i]

    return X_aligned, y, info, X # (정렬된 X, 라벨, info, 원본 X)

print(f"⏳ 105명 데이터 로딩 및 Riemannian Alignment 동시 진행 중...")
results = Parallel(n_jobs=-1)(delayed(process_subject)(subj) for subj in subjects)

valid_results = [res for res in results if res is not None]
X_aligned = np.concatenate([res[0] for res in valid_results])
y_raw = np.concatenate([res[1] for res in valid_results])
info = valid_results[0][2] 
X_raw_unaligned = np.concatenate([res[3] for res in valid_results]) # 비교를 위한 원본

B, C, T = X_aligned.shape
print(f"✅ 글로벌 데이터 로드 및 정렬 완료: {B} Trials | {C} Channels")

# =========================================================
# 2. 🧠 Feature Extraction (Aligned vs Unaligned)
# =========================================================
print("\n⚙️ 글로벌 피쳐 추출 및 계산 중...")

# (A) Baseline: 정렬되지 않은 원본 데이터의 분산
feat_raw = np.var(X_raw_unaligned, axis=2) 

# (B) Proposed: 정렬된 데이터(Aligned EEG)에 대한 Riemannian Log-Cov
def extract_riemannian_log_cov(X_batch):
    feats = []
    for x in X_batch: 
        x_centered = x - np.mean(x, axis=1, keepdims=True)
        cov = np.cov(x_centered) + np.eye(C) * 1e-4
        
        vals, vecs = la.eigh(cov)
        log_vals = np.log(np.clip(vals, a_min=1e-6, a_max=None))
        log_cov = vecs @ np.diag(log_vals) @ vecs.T
        
        feats.append(np.diag(log_cov))
    return np.array(feats)

feat_proposed = extract_riemannian_log_cov(X_aligned)

# =========================================================
# 3. 📊 통계적 검증 (F-value & LDA)
# =========================================================
print("⚙️ 글로벌 분별력 계산 및 5-Fold 교차 검증 중...")
f_val_raw, _ = f_classif(feat_raw, y_raw)
f_val_proposed, _ = f_classif(feat_proposed, y_raw)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LinearDiscriminantAnalysis()

acc_raw = cross_val_score(clf, feat_raw, y_raw, cv=cv, n_jobs=-1).mean() * 100
acc_proposed = cross_val_score(clf, feat_proposed, y_raw, cv=cv, n_jobs=-1).mean() * 100

# =========================================================
# 4. 📝 결과 터미널 출력
# =========================================================
ch_names = np.array(info.ch_names)
top_k = 15 

idx_raw = np.argsort(f_val_raw)[::-1][:top_k]
idx_prop = np.argsort(f_val_proposed)[::-1][:top_k]

print("\n" + "="*70)
print("🚀 GLOBAL Feature Test: Riemannian Alignment + Log-Cov Mapping")
print("="*70)
print("[1] Global Linear Separability (LDA 5-Fold CV Accuracy)")
print(f"  ▶ Raw Variance (Unaligned)     : {acc_raw:.2f}%")
print(f"  ▶ Aligned Riemannian Log-Cov   : {acc_proposed:.2f}%")
diff = acc_proposed - acc_raw
print(f"    (글로벌 성능 차이: {'+' if diff > 0 else ''}{diff:.2f}%p)")

print("\n" + "-"*70)
print(f"[2] Global Discriminative Power (Top {top_k} Channels by F-value)")
print(f"  {'Rank':<5} | {'Raw Variance':<20} | {'Aligned Riemannian Log-Cov'}")
print("-" * 70)

for i in range(top_k):
    raw_ch = ch_names[idx_raw[i]]
    raw_f = f_val_raw[idx_raw[i]]
    
    prop_ch = ch_names[idx_prop[i]]
    prop_f = f_val_proposed[idx_prop[i]]
    
    print(f"  {i+1:<4} | {raw_ch:<7} (F:{raw_f:>5.1f})      | {prop_ch:<7} (F:{prop_f:>5.1f})")

print("="*70)