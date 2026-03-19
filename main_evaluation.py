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
# 1. Data Loading (105명 전체, 4-Class)
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
            
            # 3초 제한
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
    return X, y, info

print(f"⏳ 105명 글로벌 데이터 로딩 시작... (멀티프로세싱 가동 중)")
results = Parallel(n_jobs=-1)(delayed(process_subject)(subj) for subj in subjects)

valid_results = [res for res in results if res is not None]
X_raw = np.concatenate([res[0] for res in valid_results])
y_raw = np.concatenate([res[1] for res in valid_results])
info = valid_results[0][2] 

B, C, T = X_raw.shape
print(f"✅ 글로벌 데이터 로드 완료: {B} Trials | {C} Channels | {T} Timesteps")

# =========================================================
# 2. 🧠 Feature Extraction 
# =========================================================
print("\n⚙️ 글로벌 피쳐 추출 및 계산 중...")

# (A) Baseline: Raw Variance
feat_raw = np.var(X_raw, axis=2) 

# (B) Proposed: Derivative-Free Riemannian Tangent Projection
def extract_derivative_free_riemannian(X_batch, window_size=64, step_size=16):
    feats = []
    num_windows = (X_batch.shape[2] - window_size) // step_size + 1
    
    for x in X_batch: 
        log_covs = []
        for w in range(num_windows):
            start = w * step_size
            end = start + window_size
            window_x = x[:, start:end]
            
            xc = window_x - np.mean(window_x, axis=1, keepdims=True)
            cov = np.cov(xc) + np.eye(C) * 1e-4
            
            vals, vecs = la.eigh(cov)
            log_vals = np.log(np.clip(vals, a_min=1e-6, a_max=None))
            log_cov = vecs @ np.diag(log_vals) @ vecs.T
            
            log_covs.append(np.diag(log_cov))
            
        log_covs = np.array(log_covs) # (W, C)
        
        # 기하학적 평균(Barycenter) 계산
        barycenter = np.mean(log_covs, axis=0, keepdims=True) 
        
        # 평균 뇌 상태로부터의 절대적 이탈 궤적 투영
        projected_trajectory = log_covs - barycenter 
        
        # 이탈 궤적의 분산을 피쳐로 사용
        feats.append(np.var(projected_trajectory, axis=0)) 
        
    return np.array(feats)

feat_proposed = extract_derivative_free_riemannian(X_raw)

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
top_k = 15 # 64채널 전체의 글로벌한 양상을 보기 위해 15개까지 출력

idx_raw = np.argsort(f_val_raw)[::-1][:top_k]
idx_prop = np.argsort(f_val_proposed)[::-1][:top_k]

print("\n" + "="*70)
print("🚀 GLOBAL Feature Test: Derivative-Free Riemannian Projection")
print("="*70)
print("[1] Global Linear Separability (LDA 5-Fold CV Accuracy)")
print(f"  ▶ Raw Variance                 : {acc_raw:.2f}%")
print(f"  ▶ Derivative-Free Riemannian   : {acc_proposed:.2f}%")
diff = acc_proposed - acc_raw
print(f"    (글로벌 성능 차이: {'+' if diff > 0 else ''}{diff:.2f}%p)")

print("\n" + "-"*70)
print(f"[2] Global Discriminative Power (Top {top_k} Channels by F-value)")
print(f"  {'Rank':<5} | {'Raw Variance':<20} | {'Derivative-Free Riemannian'}")
print("-" * 70)

for i in range(top_k):
    raw_ch = ch_names[idx_raw[i]]
    raw_f = f_val_raw[idx_raw[i]]
    
    prop_ch = ch_names[idx_prop[i]]
    prop_f = f_val_proposed[idx_prop[i]]
    
    print(f"  {i+1:<4} | {raw_ch:<7} (F:{raw_f:>5.1f})      | {prop_ch:<7} (F:{prop_f:>5.1f})")

print("="*70)