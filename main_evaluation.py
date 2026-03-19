import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from collections import Counter

warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# =========================================================
# 1. Data Loading (Raw/Filtered 뇌파 보존)
# =========================================================
# 🚨 Riemannian 기하학을 위해 진폭/파워가 아닌 필터링된 Raw 신호 그 자체를 사용함
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
            raw.filter(l_freq=8.0, h_freq=30.0, verbose=False) # Mu + Beta 대역
            
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            t1, t2 = ed.get('T1'), ed.get('T2')
            if t1 is None: continue
            
            e = evs.copy()
            if run in runs_h: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 1, 2 
            else: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 3, 4 
                
            ep = mne.Epochs(raw, e, {'L':1,'R':2} if run in runs_h else {'H':3,'F':4},
                            tmin=0.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
        except Exception: continue
            
    if not epochs_list: return None
    subj_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    X = subj_epochs.get_data() * 1e6
    y = subj_epochs.events[:, 2] - 1
    return X, y

print(f"⏳ 105명 데이터 로딩 시작...")
results = Parallel(n_jobs=-1)(delayed(process_subject)(subj) for subj in subjects)
results = [res for res in results if res is not None]

X_raw = np.concatenate([res[0] for res in results])
y_raw = np.concatenate([res[1] for res in results])

# 밸런싱 및 분할
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

X_raw, y_raw = X_raw[balanced_idx], y_raw[balanced_idx]
idx_tr, idx_te = train_test_split(np.arange(len(X_raw)), test_size=0.2, stratify=y_raw, random_state=42)

Xtr, ytr = X_raw[idx_tr], y_raw[idx_tr]
Xte, yte = X_raw[idx_te], y_raw[idx_te]

# 피험자 전체를 아우르는 Global Scaling (형태 보존)
B_tr, C_tr, T_tr = Xtr.shape
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr.transpose(0, 2, 1).reshape(-1, C_tr)).reshape(B_tr, T_tr, C_tr).transpose(0, 2, 1)

B_te, C_te, T_te = Xte.shape
Xte_scaled = scaler.transform(Xte.transpose(0, 2, 1).reshape(-1, C_te)).reshape(B_te, T_te, C_te).transpose(0, 2, 1)

BATCH_SIZE = 128
train_loader = DataLoader(TensorDataset(torch.tensor(Xtr_scaled), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(Xte_scaled), torch.tensor(yte)), batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 2. 🚀 아키텍처: Riemannian TCN-GNN driven SNN
# =========================================================
@jax.custom_vjp
def spike(x): return (x > 0).astype(jnp.float32)

def fwd(x): return spike(x), x
def bwd(res, g): return (g * 1.0 / (1.0 + jnp.abs(res)),)
spike.defvjp(fwd, bwd)

class TCNBlock(nn.Module):
    features: int
    dilation: int
    @nn.compact
    def __call__(self, x):
        # x shape: (B, T, C)
        res = x
        x = nn.Conv(self.features, kernel_size=(3,), kernel_dilation=(self.dilation,), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        if res.shape[-1] != x.shape[-1]:
            res = nn.Dense(self.features)(res)
        return x + res

class GeoTempDrivenLIF(nn.Module):
    @nn.compact
    def __call__(self, mem, x, geo_temp_feat):
        # 🔥 Novelty: 시공간-기하학적 피처가 LIF 뉴런의 감쇠율(decay)을 실시간으로 결정
        decay = nn.sigmoid(nn.Dense(x.shape[-1])(geo_temp_feat)) 
        mem = mem * decay + x  
        spk = spike(mem - 0.5) 
        mem = mem - spk * 0.5
        return mem, spk

class Model(nn.Module):
    @nn.compact
    def __call__(self, X, train):
        # X shape: (B, C, T)
        if train:
            X = X + jax.random.normal(self.make_rng('dropout'), X.shape) * 0.05
            
        B, C, T = X.shape
        
        # ---------------------------------------------------------
        # Step 1: Riemannian Manifold Mapping (Log-Euclidean)
        # ---------------------------------------------------------
        # 공분산 행렬 계산 및 안정성(Regularization) 확보
        mean_X = jnp.mean(X, axis=-1, keepdims=True)
        X_centered = X - mean_X
        cov = jnp.matmul(X_centered, jnp.transpose(X_centered, (0, 2, 1))) / (T - 1)
        cov = cov + jnp.eye(C) * 1e-4 
        
        # Tangent Space 투영 (Eigen Decomposition)
        vals, vecs = jnp.linalg.eigh(cov)
        log_vals = jnp.log(jnp.clip(vals, a_min=1e-6))
        # log_cov를 GNN의 기하학적 인접 행렬로 사용! (B, C, C)
        log_cov = jnp.matmul(vecs * jnp.expand_dims(log_vals, 1), jnp.transpose(vecs, (0, 2, 1)))
        
        # ---------------------------------------------------------
        # Step 2: TCN (Temporal Dynamics)
        # ---------------------------------------------------------
        X_t = jnp.transpose(X, (0, 2, 1)) # (B, T, C)
        tcn_out = TCNBlock(features=64, dilation=1)(X_t)
        tcn_out = TCNBlock(features=64, dilation=2)(tcn_out)
        tcn_out = TCNBlock(features=64, dilation=4)(tcn_out) # (B, T, 64)
        
        # ---------------------------------------------------------
        # Step 3: GNN 융합 (Geometric-Temporal Features)
        # ---------------------------------------------------------
        # TCN 피처를 노드(채널) 차원으로 변환: (B, C, T) * (B, T, 64) -> (B, C, 64)
        node_feats = jnp.matmul(X, tcn_out) 
        
        # 기하학적 인접 행렬(log_cov)을 이용한 GNN 연산
        adj = nn.softmax(log_cov, axis=-1)
        adj = nn.Dropout(rate=0.3, deterministic=not train)(adj)
        geo_temp_feat = jnp.matmul(adj, node_feats) # (B, C, 64)
        geo_temp_feat = nn.LayerNorm()(geo_temp_feat)
        geo_temp_feat = nn.relu(geo_temp_feat)
        
        # ---------------------------------------------------------
        # Step 4: Geometric-Temporal driven SNN
        # ---------------------------------------------------------
        X_mixed = nn.Dense(128)(geo_temp_feat.reshape(B, -1)) # (B, C*64) -> (B, 128)
        X_mixed = nn.Dropout(rate=0.4, deterministic=not train)(X_mixed)
        
        # 시계열 스텝 설정 (TCN 출력을 순차적으로 SNN에 공급)
        seq_len = 10 
        init_mem = jnp.zeros((B, 128))
        
        # LIF를 위해 TCN 피처를 활용하여 sequence 생성
        seq_input = nn.Dense(128)(tcn_out) # (B, T, 128)
        # 풀링하여 고정된 길이(seq_len)의 스텝으로 변환
        seq_input = seq_input.reshape(B, seq_len, -1, 128).mean(axis=2) # (B, seq_len, 128)
        
        def scan_fn(mem, x_t):
            # geo_temp_feat를 컨텍스트로 함께 전달
            mem, spk = GeoTempDrivenLIF()(mem, x_t, geo_temp_feat.reshape(B, -1))
            return mem, spk

        scan_layer = nn.scan(scan_fn, variable_broadcast='params', split_rngs={'params':False}, in_axes=(1), out_axes=1)
        _, spk_seq = scan_layer(init_mem, seq_input)

        # 시간축 어텐션
        temp_attn = nn.softmax(nn.Dense(1)(spk_seq), axis=1)
        feat = jnp.sum(spk_seq * temp_attn, axis=1)
        
        feat = nn.Dropout(rate=0.3, deterministic=not train)(feat)
        feat = nn.Dense(64)(feat)
        feat = nn.relu(feat)
        logits = nn.Dense(4)(feat)
        
        return logits, jnp.mean(spk_seq)

# =========================================================
# 3. Training Loop (이전과 동일한 구조 적용)
# =========================================================
def loss_fn(params, state, X, y, dropout_rng, fr_weight):
    logits, fr = state.apply_fn({'params': params}, X, train=True, rngs={'dropout': dropout_rng})
    one_hot_y = jax.nn.one_hot(y, 4)
    smooth_y = one_hot_y * 0.8 + (0.2 / 4.0)
    ce_loss = optax.softmax_cross_entropy(logits=logits, labels=smooth_y).mean()
    fr_loss = jnp.mean((fr - 0.25) ** 2) * fr_weight 
    return ce_loss + fr_loss, (logits, fr)

@jax.jit
def train_step(state, X, y, rng, fr_weight):
    rng, dropout_rng = jax.random.split(rng) 
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, fr)), grads = grad_fn(state.params, state, X, y, dropout_rng, fr_weight)
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

# 더미 데이터로 초기화 (B, C, T)
dummy_x = jnp.ones((1, 64, int(4.0 * TARGET_SFREQ)))
params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, dummy_x, train=False)['params']

steps_per_epoch = len(train_loader)
lr_sched = optax.exponential_decay(init_value=1e-3, transition_steps=steps_per_epoch * 10, decay_rate=0.95)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=lr_sched, weight_decay=1e-3))

best_acc = 0

for epoch in range(1, 301): 
    current_fr_weight = min(5.0, 0.1 + (epoch / 50.0) * 4.9) if epoch <= 50 else 5.0
    fr_weight_tensor = jnp.array(current_fr_weight, dtype=jnp.float32)
    
    t_accs, t_losses = [], []
    for Xb, yb in train_loader:
        state, loss, acc, _, rng = train_step(state, jnp.array(Xb.numpy()), jnp.array(yb.numpy()), rng, fr_weight_tensor)
        t_accs.append(acc)
        t_losses.append(loss)

    if epoch % 10 == 0:
        correct = 0
        total = 0
        all_frs = []
        
        for Xb, yb in test_loader:
            preds, vf = eval_step_preds(state, jnp.array(Xb.numpy()))
            correct += np.sum(np.array(preds) == yb.numpy())
            total += len(yb)
            all_frs.append(vf)
                
        avg_val_acc = (correct / total) * 100
        avg_train_acc = np.mean(t_accs) * 100
        avg_loss = np.mean(t_losses)
        
        print(f"[{epoch:3d}] Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | FR: {np.mean(all_frs):.4f}")
        best_acc = max(best_acc, avg_val_acc)

print(f"🔥 BEST TEST ACC (Riemannian-TCN-GNN-SNN): {best_acc:.2f}%")