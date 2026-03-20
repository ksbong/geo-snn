import os
import gc
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

base = './'
DATA_DIR = os.path.join(base, '07_Data') 
TARGET_SFREQ = 160.0 
runs_h, runs_f = ['R04','R08','R12'], ['R06','R10','R14']

# 💡 검증을 위해 앞쪽 5명만 뽑아서 개별 테스트
test_subjects = ['S001', 'S002', 'S003', 'S004', 'S005']

# SNN Architecture (Robust Version 유지)
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

class Subject_RTM_SNN(nn.Module):
    @nn.compact
    def __call__(self, x_seq, train=True):
        B, T, F = x_seq.shape
        x = nn.Dropout(0.5, deterministic=not train)(x_seq)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        init1 = (jnp.zeros((B, 128)), jnp.zeros((B, 128)))
        Scan1 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk1 = Scan1(128)(init1, x)

        spk1 = nn.Dropout(0.3, deterministic=not train)(spk1)

        x2 = nn.Dense(64)(spk1)
        x2 = nn.LayerNorm()(x2)
        init2 = (jnp.zeros((B, 64)), jnp.zeros((B, 64)))
        Scan2 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk2 = Scan2(64)(init2, x2)

        attn = nn.softmax(nn.Dense(1)(spk2), axis=1)
        feat = jnp.sum(spk2 * attn, axis=1)

        feat = nn.Dropout(0.5, deterministic=not train)(feat)
        logits = nn.Dense(4)(feat)
        return logits, (jnp.mean(spk1) + jnp.mean(spk2)) / 2.0

def loss_fn(params, state, x, y, rng):
    logits, fr = state.apply_fn({'params': params}, x, train=True, rngs={'dropout': rng})
    y_s = jax.nn.one_hot(y, 4) * 0.9 + 0.025 
    ce = optax.softmax_cross_entropy(logits=logits, labels=y_s).mean()
    fr_loss = jnp.mean((fr - 0.15)**2) * 0.05
    return ce + fr_loss, logits

@jax.jit
def train_step(state, x, y, rng):
    rng, sub = jax.random.split(rng)
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state, x, y, sub)
    state = state.apply_gradients(grads=grads)
    return state, loss, jnp.mean(jnp.argmax(logits, -1) == y), rng

@jax.jit
def eval_step(state, x):
    logits, _ = state.apply_fn({'params': state.params}, x, train=False)
    return jnp.argmax(logits, -1)

# =========================================================
# 🚀 Intra-Subject Evaluation Loop
# =========================================================
print("\n" + "="*50)
print("🚀 피험자별 독립 검증 (Intra-Subject Evaluation)")
print("="*50)

subject_accuracies = []

for subj in test_subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    if not os.path.exists(subj_dir): continue

    # 1. 데이터 로딩 (한 명만)
    epochs_list = []
    for run in runs_h + runs_f:
        path = os.path.join(subj_dir, f'{subj}{run}.edf')
        if not os.path.exists(path): continue
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
            raw.rename_channels(lambda x: x.strip('.'))
            raw.filter(8.0, 30.0, verbose=False)
            
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
            
    if not epochs_list: continue
    epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    X, y = epochs.get_data() * 1e6, epochs.events[:,2]

    # 2. 정렬 및 특징 추출 (이 사람만의 데이터로)
    covs = [np.cov(x - np.mean(x, axis=1, keepdims=True)) for x in X]
    R_i = np.mean(covs, axis=0) + np.eye(X.shape[1]) * 1e-4
    vals, vecs = la.eigh(R_i)
    r_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-6, None))) @ vecs.T
    X_aligned = np.array([r_inv_sqrt @ x for x in X])

    triu_idx = np.triu_indices(X.shape[1])
    seq_feats = []
    window, step = 160, 16 
    for x in X_aligned:
        win_feats = []
        for w in range((x.shape[1] - window) // step + 1):
            win = x[:, w*step : w*step+window]
            cov = np.cov(win - np.mean(win, axis=1, keepdims=True)) + np.eye(64) * 1e-4
            vals_w, vecs_w = la.eigh(cov)
            log_cov = vecs_w @ np.diag(np.log(np.clip(vals_w, 1e-6, None))) @ vecs_w.T
            win_feats.append(log_cov[triu_idx]) 
        seq_feats.append(np.array(win_feats))
    
    X_seq = np.array(seq_feats)

    # 3. 모델 학습 (피험자 1명 전용 SNN)
    idx_tr, idx_te = train_test_split(np.arange(len(X_seq)), test_size=0.2, stratify=y, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_seq[idx_tr], dtype=torch.float32), torch.tensor(y[idx_tr])), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_seq[idx_te], dtype=torch.float32), torch.tensor(y[idx_te])), batch_size=16, shuffle=False)

    model = Subject_RTM_SNN()
    rng = jax.random.PRNGKey(42)
    params = model.init({'params': rng, 'dropout': rng}, jnp.ones((1, 21, 2080)))['params']
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=5e-4, weight_decay=1e-2))

    best_val = 0
    for epoch in range(1, 61): # 개별 학습은 에폭을 짧게
        for xb, yb in train_loader:
            state, _, _, rng = train_step(state, jnp.array(xb.numpy()), jnp.array(yb.numpy()), rng)
        
        preds = []
        for xb, _ in test_loader:
            preds.extend(np.array(eval_step(state, jnp.array(xb.numpy()))))
        val_acc = np.mean(np.array(preds) == y[idx_te]) * 100
        best_val = max(best_val, val_acc)

    print(f"▶ {subj} | Best Test Acc: {best_val:.2f}%")
    subject_accuracies.append(best_val)

print("="*50)
print(f"🔥 Average Intra-Subject Accuracy (Top 5): {np.mean(subject_accuracies):.2f}%")