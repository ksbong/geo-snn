import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
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
                
                # 채널 이름 정리 및 필터링
                raw.rename_channels(lambda x: x.strip('.'))
                raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
                
                # 몽타주 매핑 (에러 방지용)
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
                
        if epochs_list:
            subj_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
            X_list.append(subj_epochs.get_data() * 1e6)
            y_list.append(subj_epochs.events[:, 2])
            
    return np.concatenate(X_list), np.concatenate(y_list), info

print("⏳ 10명 데이터 로딩 중 (4-Class)...")
X, y, info = load_data()
B, C, T = X.shape
print(f"✅ 데이터 로드 완료: {B} Trials | {C} Channels | {T} Timesteps")

# =========================================================
# 2. 🧠 Feature Extraction 
# =========================================================
print("\n⚙️ 피쳐 추출 및 계산 중...")

# (A) Baseline: Raw Variance
feat_raw = np.var(X, axis=2) 

# (B) Proposed: Riemannian Tangent Space (Log-Euclidean)
def extract_riemannian_log_cov(X_batch):
    feats = []
    for x in X_batch: 
        # 1. Covariance Matrix 계산
        x_centered = x - np.mean(x, axis=1, keepdims=True)
        cov = np.cov(x_centered)
        
        # 수치적 안정성을 위해 대각선에 작은 값 추가 (Regularization)
        cov = cov + np.eye(cov.shape[0]) * 1e-4
        
        # 2. Matrix Logarithm (Tangent Space Mapping)
        # 고유값 분해를 통해 logm 계산이 빠르고 정확함
        vals, vecs = la.eigh(cov)
        log_vals = np.log(np.clip(vals, a_min=1e-6, a_max=None))
        log_cov = vecs @ np.diag(log_vals) @ vecs.T
        
        # 3. 채널 본연의 기하학적 분산값 (대각성분) 추출
        # 원래 공간의 물리적 특성을 유지하기 위해 대각선(Self-variance)만 사용
        feats.append(np.diag(log_cov))
    return np.array(feats)

feat_proposed = extract_riemannian_log_cov(X)

# =========================================================
# 3. 📊 통계적 검증 (F-value & LDA)
# =========================================================
f_val_raw, _ = f_classif(feat_raw, y)
f_val_proposed, _ = f_classif(feat_proposed, y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LinearDiscriminantAnalysis()

acc_raw = cross_val_score(clf, feat_raw, y, cv=cv).mean() * 100
acc_proposed = cross_val_score(clf, feat_proposed, y, cv=cv).mean() * 100

# =========================================================
# 4. 📝 결과 터미널 출력
# =========================================================
ch_names = np.array(info.ch_names)
top_k = 10 

idx_raw = np.argsort(f_val_raw)[::-1][:top_k]
idx_prop = np.argsort(f_val_proposed)[::-1][:top_k]

print("\n" + "="*60)
print("🚀 Feature Test Results (Riemannian Tangent Mapping)")
print("="*60)
print("[1] Linear Separability (LDA 5-Fold CV Accuracy)")
print(f"  ▶ Raw Variance           : {acc_raw:.2f}%")
print(f"  ▶ Riemannian Log-Cov     : {acc_proposed:.2f}%")
diff = acc_proposed - acc_raw
print(f"    (성능 차이: {'+' if diff > 0 else ''}{diff:.2f}%p)")

print("\n" + "-"*60)
print(f"[2] Discriminative Power (Top {top_k} Channels by F-value)")
print(f"  {'Rank':<5} | {'Raw Variance':<20} | {'Riemannian Log-Cov'}")
print("-" * 60)

for i in range(top_k):
    raw_ch = ch_names[idx_raw[i]]
    raw_f = f_val_raw[idx_raw[i]]
    
    prop_ch = ch_names[idx_prop[i]]
    prop_f = f_val_proposed[idx_prop[i]]
    
    # 주요 타겟 채널(C3, C4, Cz) 시각적 강조
    rc = f"*{raw_ch}*" if raw_ch in ['C3', 'C4', 'Cz', 'Cp3', 'Cp4'] else raw_ch
    pc = f"*{prop_ch}*" if prop_ch in ['C3', 'C4', 'Cz', 'Cp3', 'Cp4'] else prop_ch
    
    print(f"  {i+1:<4} | {rc:<7} (F:{raw_f:>5.1f})      | {pc:<7} (F:{prop_f:>5.1f})")

print("="*60)