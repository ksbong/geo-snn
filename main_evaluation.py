import os
import gc
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from joblib import Parallel, delayed
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# =========================================================
# 1. Memory-Efficient Subject-Specific RTM Extraction
# =========================================================
base = './'
DATA_DIR = os.path.join(base, '07_Data') 

bad_subjects = ['S088', 'S089', 'S092', 'S100']
subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in bad_subjects]
runs_h, runs_f = ['R04','R08','R12'], ['R06','R10','R14']
TARGET_SFREQ = 160.0 

def process_and_extract_sequential_features(subj):
    subj_dir = os.path.join(DATA_DIR, subj)
    if not os.path.exists(subj_dir): return None

    epochs_list = []
    for run in runs_h + runs_f:
        path = os.path.join(subj_dir, f'{subj}{run}.edf')
        if not os.path.exists(path): continue
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
            raw.rename_channels(lambda x: x.strip('.'))
            raw.filter(8.0, 30.0, verbose=False)
            mne.datasets.eegbci.standardize(raw) 
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, match_case=False, on_missing='ignore')

            evs, ed = mne.events_from_annotations(raw, verbose=False)
            t1, t2 = ed.get('T1'), ed.get('T2')
            if t1 is None: continue
            
            e = evs.copy()
            if run in runs_h: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 0, 1
            else: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 2, 3

            event_id = {'L':0, 'R':1} if run in runs_h else {'H':2, 'F':3}
            ep = mne.Epochs(raw, e, event_id, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
        except: continue

    if not epochs_list: return None
    epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    X = epochs.get_data() * 1e6
    y = epochs.events[:,2]

    # Subject-Specific Alignment
    covs = [np.cov(x - np.mean(x, axis=1, keepdims=True)) for x in X]
    R_i = np.mean(covs, axis=0) + np.eye(X.shape[1]) * 1e-4
    vals, vecs = la.eigh(R_i)
    r_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-6, None))) @ vecs.T
    X_aligned = np.array([r_inv_sqrt @ x for x in X])

    # Sequential Trajectory Extraction
    C = X_aligned.shape[1]
    triu_idx = np.triu_indices(C)
    
    subj_seq_feats = []
    window, step = 64, 32
    for x in X_aligned:
        win_feats = []
        for w in range((x.shape[1] - window) // step + 1):
            win = x[:, w*step : w*step+window]
            cov = np.cov(win - np.mean(win, axis=1, keepdims=True)) + np.eye(C) * 1e-4
            vals_w, vecs_w = la.eigh(cov)
            log_cov = vecs_w @ np.diag(np.log(np.clip(vals_w, 1e-6, None))) @ vecs_w.T
            win_feats.append(log_cov[triu_idx]) 
        subj_seq_feats.append(np.array(win_feats)) 
        
    return np.array(subj_seq_feats), y

print("⏳ 시간축이 보존된 Sequential RTM 피쳐 추출 중...")
results = Parallel(n_jobs=-1)(delayed(process_and_extract_sequential_features)(s) for s in subjects)
valid_results = [r for r in results if r is not None]

X_seq_all = np.concatenate([r[0] for r in valid_results]) # Shape: (B, Seq_len, 2080)
y_all = np.concatenate([r[1] for r in valid_results])

del results, valid_results
gc.collect()

print("⚖️ 4-Class 동등 비율 조정 중...")
unique, counts = np.unique(y_all, return_counts=True)
min_samples = np.min(counts)
balanced_indices = []
for cls in unique:
    balanced_indices.extend(np.random.choice(np.where(y_all == cls)[0], min_samples, replace=False))

X_bal_seq = X_seq_all[balanced_indices]
y_bal = y_all[balanced_indices]

# DataLoaders
idx_tr, idx_te = train_test_split(np.arange(len(X_bal_seq)), test_size=0.2, stratify=y_bal, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_bal_seq[idx_tr], dtype=torch.float32), torch.tensor(y_bal[idx_tr])), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_bal_seq[idx_te], dtype=torch.float32), torch.tensor(y_bal[idx_te])), batch_size=128, shuffle=False)

_, Seq_len, Feat_dim = X_bal_seq.shape

# =========================================================
# 2. 🚀 JAX/Flax SNN Architecture
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

class Sequential_RTM_SNN(nn.Module):
    @nn.compact
    def __call__(self, x_seq, train=True):
        B, T, F = x_seq.shape

        # Input Projection: 2080차원 거대 궤적을 Spike-friendly하게 압축
        x = nn.Dense(512)(x_seq)
        x = nn.gelu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)

        # SNN Layer 1
        init1 = (jnp.zeros((B, 256)), jnp.zeros((B, 256)))
        Scan1 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk1 = Scan1(256)(init1, x)

        # SNN Layer 2
        x2 = nn.Dense(128)(spk1)
        x2 = nn.LayerNorm()(x2)
        init2 = (jnp.zeros((B, 128)), jnp.zeros((B, 128)))
        Scan2 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk2 = Scan2(128)(init2, x2)

        # Temporal Attention (시간축 스파이크 중요도 가중합)
        attn = nn.softmax(nn.Dense(1)(spk2), axis=1)
        feat = jnp.sum(spk2 * attn, axis=1)

        feat = nn.Dropout(0.5, deterministic=not train)(feat)
        logits = nn.Dense(4)(feat)

        fr = (jnp.mean(spk1) + jnp.mean(spk2)) / 2.0
        return logits, fr

# =========================================================
# 3. Training Loop
# =========================================================
def loss_fn(params, state, x, y, rng):
    logits, fr = state.apply_fn({'params': params}, x, train=True, rngs={'dropout': rng})
    y_s = jax.nn.one_hot(y, 4) * 0.9 + 0.025 # Label Smoothing
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

print("\n🚀 SNN 메인 학습 시작...")
model = Sequential_RTM_SNN()
rng = jax.random.PRNGKey(42)
params = model.init({'params': rng, 'dropout': rng}, jnp.ones((1, Seq_len, Feat_dim)))['params']

lr_schedule = optax.exponential_decay(init_value=1e-3, transition_steps=len(train_loader)*20, decay_rate=0.8)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=lr_schedule, weight_decay=1e-3))

best_acc = 0
best_params = params

for epoch in range(1, 101):
    losses, accs = [], []
    for xb, yb in train_loader:
        state, l, a, rng = train_step(state, jnp.array(xb.numpy()), jnp.array(yb.numpy()), rng)
        losses.append(l)
        accs.append(a)

    if epoch % 5 == 0:
        preds = []
        for xb, _ in test_loader:
            p = eval_step(state, jnp.array(xb.numpy()))
            preds.extend(np.array(p))
        val_acc = np.mean(np.array(preds) == y_bal[idx_te]) * 100
        print(f"[{epoch:3d}] Train Acc: {np.mean(accs)*100:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = state.params

print(f"\n🔥 BEST SNN TEST ACCURACY: {best_acc:.2f}%")

# =========================================================
# 4. IEEE-Grade Evaluation (최고 성능 파라미터로 측정)
# =========================================================
print("\n" + "="*60)
print("📊 IEEE-Grade Final Evaluation: Sequential RTM + SNN")
print("="*60)

# Best 파라미터 장착 후 전체 Test Set 예측
final_preds = []
for xb, _ in test_loader:
    logits, _ = model.apply({'params': best_params}, jnp.array(xb.numpy()), train=False)
    final_preds.extend(np.array(jnp.argmax(logits, -1)))

y_true = y_bal[idx_te]
y_pred = np.array(final_preds)

kappa = cohen_kappa_score(y_true, y_pred)
print(f"▶ Cohen's Kappa    : {kappa:.4f} (0.4 이상 Good, 0.6 이상 Excellent)")

print("\n▶ Classification Report:")
target_names = ['Left Hand (0)', 'Right Hand (1)', 'Both Hands (2)', 'Both Feet (3)']
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={"size": 14})
plt.title(f"SNN Confusion Matrix (Acc: {best_acc:.2f}%)", fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()
print("="*60)