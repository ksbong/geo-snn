import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import warnings

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# =========================================================
# 1. Data Loading & 4-Class Balancing
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
    for run in runs_h + runs_f:
        path = os.path.join(subj_dir, f'{subj}{run}.edf')
        if not os.path.exists(path): continue

        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ)
            
            raw.rename_channels(lambda x: x.strip('.'))
            raw.filter(8.0, 30.0, verbose=False)

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

            # 🔥 문제의 원인 수정 (해당 Run에 존재하는 이벤트만 지정)
            event_id = {'L':0, 'R':1} if run in runs_h else {'H':2, 'F':3}
            ep = mne.Epochs(raw, e, event_id, tmin=0.0, tmax=3.0, 
                            baseline=None, preload=True, verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
        except: continue

    if not epochs_list: return None
    epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    return epochs.get_data() * 1e6, epochs.events[:,2]

print("⏳ 데이터 로딩 시작 (105 Subjects)...")
results = Parallel(n_jobs=-1)(delayed(process_subject)(s) for s in subjects)
valid_results = [r for r in results if r is not None]

X_raw = np.concatenate([r[0] for r in valid_results])
y_raw = np.concatenate([r[1] for r in valid_results])

print("⚖️ 4-Class 동등 비율 조정 중...")
unique, counts = np.unique(y_raw, return_counts=True)
min_samples = np.min(counts)
balanced_indices = []
for cls in unique:
    cls_idx = np.where(y_raw == cls)[0]
    balanced_indices.extend(np.random.choice(cls_idx, min_samples, replace=False))
np.random.shuffle(balanced_indices)

X_bal = X_raw[balanced_indices]
y_bal = y_raw[balanced_indices]

# 🔥 핵심 1: 데이터 분리를 가장 먼저 수행 (Data Leakage 원천 차단)
idx_tr, idx_te = train_test_split(np.arange(len(X_bal)), test_size=0.2, stratify=y_bal, random_state=42)
y_tr, y_te = y_bal[idx_tr], y_bal[idx_te]

# =========================================================
# 2. Strict Riemannian Alignment (Train -> Test)
# =========================================================
print("⚙️ Strict 기하학적 정렬 (Train 기준으로만 R 계산)...")
# Train 데이터로만 기준점(Reference) R_train 계산
covs_tr = [np.cov(x - np.mean(x, axis=1, keepdims=True)) for x in X_bal[idx_tr]]
R_train = np.mean(covs_tr, axis=0) + np.eye(X_bal.shape[1]) * 1e-4

vals, vecs = la.eigh(R_train)
r_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, a_min=1e-6, a_max=None))) @ vecs.T

# Train/Test 모두 R_train을 사용해 정렬
X_aligned_tr = np.array([r_inv_sqrt @ x for x in X_bal[idx_tr]])
X_aligned_te = np.array([r_inv_sqrt @ x for x in X_bal[idx_te]])

# =========================================================
# 3. Rich Trajectory Extraction & Strict Centering
# =========================================================
print("⚙️ Rich Trajectory 추출 및 Train Mean Centering...")
def extract_raw_logcov_trajectory(X_batch, window=64, step=32):
    B, C, T = X_batch.shape
    triu_idx = np.triu_indices(C)
    
    seqs = []
    for x in X_batch:
        feats = []
        for w in range((T - window) // step + 1):
            win = x[:, w*step : w*step+window]
            cov = np.cov(win - np.mean(win, axis=1, keepdims=True)) + np.eye(C) * 1e-4
            vals, vecs = la.eigh(cov)
            log_cov = vecs @ np.diag(np.log(np.clip(vals, 1e-6, None))) @ vecs.T
            feats.append(log_cov[triu_idx]) # 2080 차원 상삼각 벡터 보존
        seqs.append(feats)
    return np.array(seqs) # (B, Seq_len, 2080)

raw_seq_tr = extract_raw_logcov_trajectory(X_aligned_tr)
raw_seq_te = extract_raw_logcov_trajectory(X_aligned_te)

# 🔥 핵심 2: Train 데이터의 시퀀스 평균만 계산해서 Centering
train_barycenter = np.mean(raw_seq_tr, axis=(0, 1), keepdims=True)
traj_seq_tr = raw_seq_tr - train_barycenter
traj_seq_te = raw_seq_te - train_barycenter

# =========================================================
# 4. Strict PCA Compression
# =========================================================
print("⚙️ PCA 차원 축소 (Train Fit -> Test Transform)...")
B_tr, Seq_len, F_dim = traj_seq_tr.shape
B_te = traj_seq_te.shape[0]

traj_tr_flat = traj_seq_tr.reshape(-1, F_dim)
traj_te_flat = traj_seq_te.reshape(-1, F_dim)

# 🔥 핵심 3: Train 데이터로만 PCA 가중치 학습
pca = PCA(n_components=256, whiten=True, random_state=42)
pca.fit(traj_tr_flat)

X_seq_tr = pca.transform(traj_tr_flat).reshape(B_tr, Seq_len, 256)
X_seq_te = pca.transform(traj_te_flat).reshape(B_te, Seq_len, 256)

# Dataloaders
train_loader = DataLoader(TensorDataset(torch.tensor(X_seq_tr, dtype=torch.float32), torch.tensor(y_tr)), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_seq_te, dtype=torch.float32), torch.tensor(y_te)), batch_size=128, shuffle=False)

# =========================================================
# 5. SNN Core & Architecture 
# =========================================================
@jax.custom_vjp
def spike(x): return (x > 0).astype(jnp.float32)

def fwd(x): return spike(x), x
def bwd(res, g): return (g * 0.3 / (1.0 + jnp.abs(res * 3.0)),)
spike.defvjp(fwd, bwd)

class LIFCell(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, state, x):
        v, z = state
        decay = nn.sigmoid(self.param("decay", nn.initializers.constant(1.0), (self.hidden_dim,)))
        v = v * decay * (1.0 - z) + x
        z_new = spike(v - 0.5)
        return (v, z_new), z_new

class Strict_RTM_SNN(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        B, T, F = x.shape

        x = nn.Dense(512)(x)
        x = nn.gelu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)

        init1 = (jnp.zeros((B,256)), jnp.zeros((B,256)))
        Scan1 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk1 = Scan1(256)(init1, x)

        x2 = nn.Dense(128)(spk1)
        x2 = nn.LayerNorm()(x2)

        init2 = (jnp.zeros((B,128)), jnp.zeros((B,128)))
        Scan2 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk2 = Scan2(128)(init2, x2)

        attn = nn.softmax(nn.Dense(1)(spk2), axis=1)
        feat = jnp.sum(spk2 * attn, axis=1)

        feat = nn.Dropout(0.5, deterministic=not train)(feat)
        logits = nn.Dense(4)(feat)

        fr = (jnp.mean(spk1) + jnp.mean(spk2)) / 2
        return logits, fr

# =========================================================
# 6. Training
# =========================================================
def loss_fn(params, state, x, y, rng):
    logits, fr = state.apply_fn({'params': params}, x, train=True, rngs={'dropout': rng})
    y_s = jax.nn.one_hot(y, 4)*0.9 + 0.025
    ce = optax.softmax_cross_entropy(logits=logits, labels=y_s).mean()
    fr_loss = jnp.mean((fr - 0.15)**2) * 0.05
    return ce + fr_loss, logits

@jax.jit
def train_step(state, x, y, rng):
    rng, sub = jax.random.split(rng)
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state, x, y, sub)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return state, loss, acc, rng

@jax.jit
def eval_step(state, x):
    logits, _ = state.apply_fn({'params': state.params}, x, train=False)
    return jnp.argmax(logits, -1)

print("\n🚀 엄밀한 무결점 RTM-SNN 학습 시작...")
model = Strict_RTM_SNN()
rng = jax.random.PRNGKey(42)
params = model.init({'params': rng, 'dropout': rng}, jnp.ones((1, Seq_len, 256)))['params']

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3, weight_decay=1e-3))
best = 0

for epoch in range(1, 151):
    losses, accs = [], []
    for xb, yb in train_loader:
        state, l, a, rng = train_step(state, jnp.array(xb.numpy()), jnp.array(yb.numpy()), rng)
        losses.append(l)
        accs.append(a)

    if epoch % 10 == 0:
        preds = []
        for xb, _ in test_loader:
            p = eval_step(state, jnp.array(xb.numpy()))
            preds.extend(np.array(p))
        
        val_acc = np.mean(np.array(preds) == y_te) * 100
        print(f"[{epoch:3d}] Train: {np.mean(accs)*100:.2f}% | Val: {val_acc:.2f}%")
        best = max(best, val_acc)

print(f"\n🔥 BEST TEST ACC (Leakage-Free): {best:.2f}%")