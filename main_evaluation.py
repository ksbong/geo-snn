import os
import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.fft import fft, ifft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
import gc 

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from collections import Counter

warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# =========================================================
# 1. Feature Extraction (Power Envelope)
# =========================================================
def extract_envelope_window(slice_data, sfreq):
    slice_norm = (slice_data - np.mean(slice_data, axis=-1, keepdims=True)) / (np.std(slice_data, axis=-1, keepdims=True) + 1e-8)
    
    n_times = slice_norm.shape[-1]
    freqs = fftfreq(n_times, 1 / sfreq)
    X_fft = fft(slice_norm, axis=-1)
    
    mu_mask = (np.abs(freqs) >= 8.0) & (np.abs(freqs) <= 12.0)
    beta_mask = (np.abs(freqs) >= 13.0) & (np.abs(freqs) <= 30.0)
    
    env_mu = np.abs(ifft(X_fft * mu_mask, axis=-1))
    env_beta = np.abs(ifft(X_fft * beta_mask, axis=-1))
    
    env_combined = np.stack([env_mu, env_beta], axis=-1).transpose(1, 0, 2)
    
    T, C, F = env_combined.shape
    scaler = MinMaxScaler()
    env_scaled = scaler.fit_transform(env_combined.reshape(-1, F)).reshape(T, C, F)
    
    return env_scaled.astype(np.float32)

def augment_trial(trial_data, sfreq, win_sec=2.0, step_sec=0.5):
    win_size = int(win_sec * sfreq)
    step_size = int(step_sec * sfreq)
    n_times = trial_data.shape[-1]
    
    feats = []
    for start in range(0, n_times - win_size + 1, step_size):
        end = start + win_size
        slice_data = trial_data[:, start:end]
        feats.append(extract_envelope_window(slice_data, sfreq))
        
    return np.array(feats)

# =========================================================
# 2. Multi-Subject Data Loading (🔥 Resampling 안전장치 추가)
# =========================================================
base = './'
DATA_DIR = os.path.join(base, '07_Data') 

subjects = [f'S{i:03d}' for i in range(1, 110)]
runs_h, runs_f = ['R04','R08','R12'], ['R06','R10','R14']
epochs_list = []
TARGET_SFREQ = 160.0 # 🔥 통일할 타겟 주파수

print("⏳ 109명 전체 데이터 로딩 시작... (Resampling 및 지뢰 패스 진행 중)")
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    if not os.path.exists(subj_dir): 
        continue
        
    for run in runs_h + runs_f:
        path = os.path.join(subj_dir, f'{subj}{run}.edf')
        if not os.path.exists(path): continue
        
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            
            # 🔥 주파수 불일치 에러 방지용 강제 리샘플링
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ)
                
            raw.rename_channels(lambda x: x.strip('.')) 
            
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            t1, t2 = ed.get('T1'), ed.get('T2')
            if t1 is None: continue
            
            e = evs.copy()
            if run in runs_h: 
                e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 1, 2 
            else: 
                e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 3, 4 
                
            ep = mne.Epochs(raw, e, {'L':1,'R':2} if run in runs_h else {'H':3,'F':4},
                            tmin=0.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
            
            del raw, evs, e
            gc.collect()
            
        except Exception as err:
            print(f"⚠️ {subj}{run} 로드 실패 (패스): {err}")
            continue

if not epochs_list:
    raise ValueError("🚨 데이터를 못 불러왔어! 경로를 다시 확인해.")

epochs = mne.concatenate_epochs(epochs_list)
X_raw = epochs.get_data() * 1e6 
y_raw = epochs.events[:, 2] - 1
sfreq = epochs.info['sfreq']

del epochs_list, epochs
gc.collect()

# 4-Class 완벽 밸런싱
idx_L, idx_R, idx_H, idx_F = (np.where(y_raw == i)[0] for i in range(4))
min_trials = min(len(idx_L), len(idx_R), len(idx_H), len(idx_F))

np.random.seed(42)
balanced_idx = np.concatenate([
    np.random.choice(idx_L, min_trials, replace=False),
    np.random.choice(idx_R, min_trials, replace=False),
    np.random.choice(idx_H, min_trials, replace=False),
    np.random.choice(idx_F, min_trials, replace=False)
])
np.random.shuffle(balanced_idx)

X_raw = X_raw[balanced_idx]
y_raw = y_raw[balanced_idx]
print(f"📊 100명대 통합 밸런싱 완료: 각 클래스당 {min_trials}개 (총 {len(y_raw)}개 트라이얼)")

BATCH_SIZE = 128
idx_tr, idx_te = train_test_split(np.arange(len(X_raw)), test_size=0.2, stratify=y_raw, random_state=42)

def create_dataset_with_ids(indices):
    X_list, y_list, trial_ids = [], [], []
    for idx in indices:
        fs = augment_trial(X_raw[idx], sfreq)
        X_list.append(fs)
        y_list.append(np.full(len(fs), y_raw[idx]))
        trial_ids.append(np.full(len(fs), idx)) 
    return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(trial_ids)

Xtr, ytr, _ = create_dataset_with_ids(idx_tr)
Xte, yte, id_te = create_dataset_with_ids(idx_te)

print(f"🚀 증강 후 윈도우 수 - Train: {len(Xtr)}개, Test: {len(Xte)}개")

train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)), batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 3. SNN Model 
# =========================================================
@jax.custom_vjp
def spike(x): return (x > 0).astype(jnp.float32)

def fwd(x): return spike(x), x
def bwd(res, g): return (g * 1.0 / (1.0 + jnp.abs(res)),)
spike.defvjp(fwd, bwd)

class LIF(nn.Module):
    @nn.compact
    def __call__(self, mem, x):
        mem = mem * 0.9 + x  
        spk = spike(mem - 0.5) 
        mem = mem - spk * 0.5
        return mem, spk

class SNNLayer(nn.Module):
    @nn.compact
    def __call__(self, mem, x):
        mem, spk = LIF()(mem, x)
        return mem, spk

class Model(nn.Module):
    @nn.compact
    def __call__(self, X, train):
        if train:
            noise = jax.random.normal(self.make_rng('dropout'), X.shape) * 0.05
            X = X + noise

        X_conv = nn.Conv(features=32, kernel_size=(5, 1), strides=(2, 1), padding='SAME')(X)
        X_conv = nn.relu(X_conv)
        X_conv = nn.LayerNorm()(X_conv)
        
        adj_base = self.param('adj_matrix', jax.nn.initializers.normal(stddev=0.1), (64, 64))
        adj_sym = (adj_base + adj_base.T) / 2.0
        adj_sym = adj_sym - jnp.diag(jnp.diag(adj_sym))
        
        channel_attn = nn.Dense(64)(jnp.mean(X_conv, axis=(1, 3))) 
        channel_attn = nn.sigmoid(channel_attn)
        channel_attn = jnp.expand_dims(channel_attn, axis=(1, 3)) 
        
        adj_sym = nn.Dropout(rate=0.5, deterministic=not train)(adj_sym)
        X_graph = jnp.einsum('nm,btmf->btnf', adj_sym, X_conv)
        X_graph = X_graph * channel_attn 
        
        X_res = X_graph + X_conv
        
        B, T_new, C, F = X_res.shape
        X_flat = X_res.reshape(B, T_new, -1)
        
        X_mixed = nn.Dense(128)(X_flat)
        X_mixed = nn.relu(X_mixed)
        X_mixed = nn.LayerNorm()(X_mixed)
        X_mixed = nn.Dropout(rate=0.4, deterministic=not train)(X_mixed) 
        
        init_mem = jnp.zeros((B, 128)) 
        scan_layer = nn.scan(SNNLayer, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk_seq = scan_layer()(init_mem, X_mixed)

        feat = jnp.mean(spk_seq, axis=1) 
        
        feat = nn.Dropout(rate=0.3, deterministic=not train)(feat)
        feat = nn.Dense(128)(feat)
        feat = nn.relu(feat)
        logits = nn.Dense(4)(feat)
        
        return logits, jnp.mean(spk_seq)

# =========================================================
# 4. Training (FR 페널티 유지)
# =========================================================
def loss_fn(params, state, X, y, dropout_rng):
    logits, fr = state.apply_fn({'params': params}, X, train=True, rngs={'dropout': dropout_rng})
    
    one_hot_y = jax.nn.one_hot(y, 4)
    smooth_y = one_hot_y * 0.8 + (0.2 / 4.0)
    ce_loss = optax.softmax_cross_entropy(logits=logits, labels=smooth_y).mean()
    
    target_fr = 0.25
    fr_loss = jnp.mean((fr - target_fr) ** 2) * 5.0 
    
    return ce_loss + fr_loss, (logits, fr)

@jax.jit
def train_step(state, X, y, rng):
    rng, dropout_rng = jax.random.split(rng) 
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, fr)), grads = grad_fn(state.params, state, X, y, dropout_rng)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return state, loss, acc, fr, rng

@jax.jit
def eval_step_preds(state, X):
    logits, fr = state.apply_fn({'params': state.params}, X, train=False)
    preds = jnp.argmax(logits, -1)
    return preds, fr

rng = jax.random.PRNGKey(42)
rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
model = Model()
win_size = int(2.0 * TARGET_SFREQ) 

params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, 
                    jnp.ones((1, win_size, 64, 2)), train=False)['params']

steps_per_epoch = len(train_loader)
lr_sched = optax.exponential_decay(init_value=1e-3, transition_steps=steps_per_epoch * 10, decay_rate=0.95)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=lr_sched, weight_decay=1e-3))

# =========================================================
# 5. Training Loop
# =========================================================
best_acc = 0

for epoch in range(1, 301): 
    t_accs, t_losses = [], []
    for Xb, yb in train_loader:
        state, loss, acc, _, rng = train_step(state, jnp.array(Xb.numpy()), jnp.array(yb.numpy()), rng)
        t_accs.append(acc)
        t_losses.append(loss)

    if epoch % 10 == 0:
        all_preds = []
        all_frs = []
        
        for Xb, yb in test_loader:
            preds, vf = eval_step_preds(state, jnp.array(Xb.numpy()))
            all_preds.extend(np.array(preds))
            all_frs.append(vf)
            
        trial_preds = {}
        trial_truths = {}
        for i, tid in enumerate(id_te):
            if tid not in trial_preds:
                trial_preds[tid] = []
                trial_truths[tid] = yte[i]
            trial_preds[tid].append(all_preds[i])
            
        correct = 0
        total = len(trial_preds)
        for tid, preds in trial_preds.items():
            most_common = Counter(preds).most_common(1)[0][0]
            if most_common == trial_truths[tid]:
                correct += 1
                
        avg_val_acc = (correct / total) * 100
        avg_train_acc = np.mean(t_accs) * 100
        avg_loss = np.mean(t_losses)
        
        print(f"[{epoch:3d}] Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc(Voting): {avg_val_acc:.2f}% | FR: {np.mean(all_frs):.4f}")
        best_acc = max(best_acc, avg_val_acc)

print(f"🔥 BEST TEST ACC (109 Subjects + Resampled): {best_acc:.2f}%")