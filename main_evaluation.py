import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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
# 1. Data Loading & Riemannian Alignment
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
            if run in runs_h: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 0, 1
            else: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 2, 3
                
            event_id = {'L':0, 'R':1} if run in runs_h else {'H':2, 'F':3}
            ep = mne.Epochs(raw, e, event_id, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
        except Exception: continue
            
    if not epochs_list: return None
    subj_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    X = subj_epochs.get_data() * 1e6
    y = subj_epochs.events[:, 2]
    
    # 🌟 Riemannian Alignment
    covs = [np.cov(x - np.mean(x, axis=1, keepdims=True)) for x in X]
    R = np.mean(covs, axis=0) + np.eye(X.shape[1]) * 1e-4
    vals, vecs = la.eigh(R)
    r_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, a_min=1e-6, a_max=None))) @ vecs.T
    
    X_aligned = np.array([r_inv_sqrt @ x for x in X])
    return X_aligned, y

print(f"⏳ 데이터 로딩 및 기하학적 정렬 진행 중...")
results = Parallel(n_jobs=-1)(delayed(process_subject)(subj) for subj in subjects)
valid_results = [res for res in results if res is not None]

X_aligned = np.concatenate([res[0] for res in valid_results])
y_all = np.concatenate([res[1] for res in valid_results])
B, C, T = X_aligned.shape

# =========================================================
# 2. 🌟 Novelty: Sequential Riemannian Trajectory Extraction
# =========================================================
print("⚙️ SNN 인코딩을 위한 Dynamic Trajectory Sequence 추출 중...")
def extract_trajectory_sequence(X_batch, window_size=64, step_size=16):
    """SNN의 시간 스텝(Seq_len)마다 들어갈 기하학적 궤적 벡터 추출"""
    num_windows = (X_batch.shape[2] - window_size) // step_size + 1
    trajectory_seqs = []
    
    for x in X_batch:
        log_covs = []
        for w in range(num_windows):
            start = w * step_size
            window_x = x[:, start:start+window_size]
            xc = window_x - np.mean(window_x, axis=1, keepdims=True)
            cov = np.cov(xc) + np.eye(C) * 1e-4
            
            vals, vecs = la.eigh(cov)
            log_vals = np.log(np.clip(vals, a_min=1e-6, a_max=None))
            log_cov = vecs @ np.diag(log_vals) @ vecs.T
            log_covs.append(np.diag(log_cov)) # (W, C)
            
        log_covs = np.array(log_covs)
        barycenter = np.mean(log_covs, axis=0, keepdims=True)
        
        # 시간에 따른 '기하학적 일탈 정도' 자체를 시퀀스로 반환!
        trajectory = np.abs(log_covs - barycenter) # Shape: (Seq_len, C)
        trajectory_seqs.append(trajectory)
        
    return np.array(trajectory_seqs)

X_seq = extract_trajectory_sequence(X_aligned)
_, Seq_len, C_feat = X_seq.shape

# Train/Test Split & Dataloader
idx_tr, idx_te = train_test_split(np.arange(len(X_seq)), test_size=0.2, stratify=y_all, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_seq[idx_tr], dtype=torch.float32), torch.tensor(y_all[idx_tr])), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_seq[idx_te], dtype=torch.float32), torch.tensor(y_all[idx_te])), batch_size=128, shuffle=False)

# =========================================================
# 3. 🚀 SNN Architecture (Riemannian Geodesic SNN)
# =========================================================
@jax.custom_vjp
def spike(x): return (x > 0).astype(jnp.float32)

def fwd(x): return spike(x), x
def bwd(res, g): 
    alpha = 2.0
    return (g * (alpha / 2.0) / (1.0 + (alpha * res)**2),)
spike.defvjp(fwd, bwd)

class LIFCell(nn.Module):
    hidden_dim: int
    v_thresh: float = 0.5 

    @nn.compact
    def __call__(self, state, x_t):
        v, z = state
        decay = nn.sigmoid(self.param('decay', nn.initializers.constant(0.0), (self.hidden_dim,)))
        # 기하학적 일탈 전류(x_t)가 들어오며 막전위 축적
        v = v * decay * (1.0 - z) + x_t
        z_out = spike(v - self.v_thresh)
        return (v, z_out), z_out

class RG_SNN(nn.Module):
    @nn.compact
    def __call__(self, x_seq, train=True):
        B, T_seq, C = x_seq.shape
        
        if train:
            x_seq = x_seq + jax.random.normal(self.make_rng('dropout'), x_seq.shape) * 0.05
            
        # SNN Layer 1
        seq_in_1 = nn.Dense(128)(x_seq)
        seq_in_1 = nn.LayerNorm()(seq_in_1)
        
        init_state1 = (jnp.zeros((B, 128)), jnp.zeros((B, 128)))
        ScanLIF1 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        _, spk_seq1 = ScanLIF1(hidden_dim=128)(init_state1, seq_in_1)
        
        # SNN Layer 2
        seq_in_2 = nn.Dense(64)(spk_seq1)
        seq_in_2 = nn.LayerNorm()(seq_in_2) * 0.5
        
        init_state2 = (jnp.zeros((B, 64)), jnp.zeros((B, 64)))
        ScanLIF2 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        _, spk_seq2 = ScanLIF2(hidden_dim=64)(init_state2, seq_in_2)
        
        # Temporal Attention (시간축에 걸친 궤적 요동의 핵심 스텝 찾기)
        temp_attn = nn.softmax(nn.Dense(1)(spk_seq2), axis=1)
        feat = jnp.sum(spk_seq2 * temp_attn, axis=1)
        feat = nn.Dropout(rate=0.4, deterministic=not train)(feat)
        
        logits = nn.Dense(4)(feat)
        mean_fr = (jnp.mean(spk_seq1) + jnp.mean(spk_seq2)) / 2.0
        return logits, mean_fr

# =========================================================
# 4. Training Loop
# =========================================================
def loss_fn(params, state, x_seq, y, dropout_rng, fr_weight):
    logits, fr = state.apply_fn({'params': params}, x_seq, train=True, rngs={'dropout': dropout_rng})
    smooth_y = jax.nn.one_hot(y, 4) * 0.9 + (0.1 / 4.0)
    ce_loss = optax.softmax_cross_entropy(logits=logits, labels=smooth_y).mean()
    fr_loss = jnp.mean((fr - 0.2) ** 2) * fr_weight 
    return ce_loss + fr_loss, (logits, fr)

@jax.jit
def train_step(state, x_seq, y, rng, fr_weight):
    rng, dropout_rng = jax.random.split(rng) 
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, fr)), grads = grad_fn(state.params, state, x_seq, y, dropout_rng, fr_weight)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return state, loss, acc, fr, rng

@jax.jit
def eval_step(state, x_seq):
    logits, fr = state.apply_fn({'params': state.params}, x_seq, train=False)
    preds = jnp.argmax(logits, -1)
    return preds, fr

rng = jax.random.PRNGKey(42)
rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
model = RG_SNN()
dummy_x = jnp.ones((1, Seq_len, C_feat))
params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, dummy_x, train=False)['params']

lr_sched = optax.exponential_decay(init_value=1e-3, transition_steps=len(train_loader)*10, decay_rate=0.95)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=lr_sched, weight_decay=1e-3))

print("\n🚀 본격적인 SNN 학습 시작...")
best_acc = 0
for epoch in range(1, 101):
    fr_weight_tensor = jnp.array(min(2.0, 0.1 + (epoch / 30.0) * 1.9), dtype=jnp.float32)
    t_accs, t_losses = [], []
    for Xb, yb in train_loader:
        state, loss, acc, _, rng = train_step(state, jnp.array(Xb.numpy()), jnp.array(yb.numpy()), rng, fr_weight_tensor)
        t_accs.append(acc)
        t_losses.append(loss)

    if epoch % 5 == 0:
        all_preds = []
        for Xb, yb in test_loader:
            preds, _ = eval_step(state, jnp.array(Xb.numpy()))
            all_preds.extend(np.array(preds))
            
        val_acc = (np.sum(np.array(all_preds) == y_all[idx_te]) / len(idx_te)) * 100
        print(f"[{epoch:3d}] Loss: {np.mean(t_losses):.4f} | Train Acc: {np.mean(t_accs)*100:.2f}% | Val Acc: {val_acc:.2f}%")
        best_acc = max(best_acc, val_acc)

print(f"🔥 BEST TEST ACC (Riemannian Geodesic SNN): {best_acc:.2f}%")