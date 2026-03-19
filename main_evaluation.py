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

B_tr, C_tr, T_tr = Xtr.shape
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr.transpose(0, 2, 1).reshape(-1, C_tr)).reshape(B_tr, T_tr, C_tr).transpose(0, 2, 1)

B_te, C_te, T_te = Xte.shape
Xte_scaled = scaler.transform(Xte.transpose(0, 2, 1).reshape(-1, C_te)).reshape(B_te, T_te, C_te).transpose(0, 2, 1)

# Riemannian Backbone 계산 (Train 셋 전체에 대해 딱 한 번)
mean_X_tr = np.mean(Xtr_scaled, axis=-1, keepdims=True)
X_centered_tr = Xtr_scaled - mean_X_tr
cov_tr = np.matmul(X_centered_tr, X_centered_tr.transpose(0, 2, 1)) / (T_tr - 1)
# Frechet Mean 대신 단순 평균으로 Riemannian Mean 근사 (연산량 타협)
mean_cov_tr = np.mean(cov_tr, axis=0)
mean_cov_tr = mean_cov_tr + np.eye(C_tr) * 1e-4

# Tangent Space Mapping (Log-Euclidean)
vals, vecs = np.linalg.eigh(mean_cov_tr)
log_vals = np.log(np.clip(vals, a_min=1e-6))
log_cov_tr = np.matmul(vecs * log_vals, vecs.T)

BATCH_SIZE = 128
train_loader = DataLoader(TensorDataset(torch.tensor(Xtr_scaled), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(Xte_scaled), torch.tensor(yte)), batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 2. 🚀 아키텍처: Decoupled Riemannian TCN-GNN driven SNN
# =========================================================
@jax.custom_vjp
def spike(x): return (x > 0).astype(jnp.float32)

def fwd(x): return spike(x), x
def bwd(res, g): return (g * 1.0 / (1.0 + jnp.abs(res)),)
spike.defvjp(fwd, bwd)

# 🔥 수정 완료: 채널 독립적인 TCN 블록 (Node-wise Temporal Dynamics)
class NodeWiseTCNBlock(nn.Module):
    features: int
    dilation: int
    @nn.compact
    def __call__(self, x):
        # x shape: (B, T, C)
        res = x
        # 채널(C)을 유지하면서 시간축(T)에 대해서만 Conv 연산 수행
        x = nn.Conv(self.features * x.shape[-1], kernel_size=(3,), kernel_dilation=(self.dilation,), padding='SAME', feature_group_count=x.shape[-1])(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        if res.shape[-1] != x.shape[-1]:
            res = nn.Dense(x.shape[-1])(res)
        return x + res

class GeoTempDrivenLIF(nn.Module):
    @nn.compact
    def __call__(self, mem, inputs):
        x_t, geo_temp_t = inputs
        decay = nn.sigmoid(nn.Dense(x_t.shape[-1])(geo_temp_t)) 
        mem = mem * decay + x_t  
        spk = spike(mem - 0.5) 
        mem = mem - spk * 0.5
        return mem, spk

class Model(nn.Module):
    log_cov_backbone: jnp.ndarray # Riemannian Backbone을 입력으로 받음

    @nn.compact
    def __call__(self, X, train):
        if train:
            X = X + jax.random.normal(self.make_rng('dropout'), X.shape) * 0.05
            
        B, C, T = X.shape
        
        # ---------------------------------------------------------
        # Step 1: Riemannian Backbone (Pre-computed) -> GNN의 Adjacency Matrix
        # ---------------------------------------------------------
        adj = nn.softmax(self.log_cov_backbone, axis=-1)
        adj = nn.Dropout(rate=0.3, deterministic=not train)(adj)
        
        # ---------------------------------------------------------
        # Step 2: Node-wise TCN (Temporal Dynamics)
        # ---------------------------------------------------------
        X_t = jnp.transpose(X, (0, 2, 1)) # (B, T, C)
        # 🔥 수정 완료: 채널 독립적인 TCN으로 시간적 맥락 추출
        tcn_out = NodeWiseTCNBlock(features=16, dilation=1)(X_t)
        tcn_out = NodeWiseTCNBlock(features=16, dilation=2)(tcn_out)
        tcn_out = NodeWiseTCNBlock(features=16, dilation=4)(tcn_out) # (B, T, C*16)
        tcn_out = tcn_out.reshape(B, T, C, -1) # (B, T, C, 16)
        
        # ---------------------------------------------------------
        # Step 3: Riemannian-Spatio-Temporal GNN 융합
        # ---------------------------------------------------------
        # 🔥 수정 완료: Riemannian Adjacency Matrix(adj) 위에서 TCN 피처(tcn_out)가 흐름
        # (C, C) * (B, T, C, 16) -> (B, T, C, 16)
        # GNN 연산을 시간축(T)에 대해 병렬로 수행 (연산 효율 극대화)
        gnn_out = jnp.einsum('cc,btcf->btcf', adj, tcn_out) 
        gnn_out = nn.LayerNorm()(gnn_out)
        gnn_out = nn.relu(gnn_out)
        
        # ---------------------------------------------------------
        # Step 4: Geometric-Temporal driven SNN (Time-step Sequential Processing)
        # ---------------------------------------------------------
        seq_len = 10 # 시간 스텝을 10개로 pooling
        init_mem = jnp.zeros((B, 128))
        
        # SNN 입력을 위해 GNN 출력 pooling
        # (B, T, C, 16) -> (B, T, C*16)
        seq_input = gnn_out.reshape(B, T, -1)
        seq_input = nn.Dense(128)(seq_input)
        # 자투리 절삭 (이전 코드의 수정 사항 적용)
        valid_T = (T // seq_len) * seq_len
        seq_input = seq_input[:, :valid_T, :]
        # (B, valid_T, 128) -> (B, seq_len, 128) 로 pooling
        seq_input = seq_input.reshape(B, seq_len, -1, 128).mean(axis=2)
        
        # geo_temp_feat를 시간축으로 확장해서 nn.scan에 던짐 (이전 코드의 수정 사항 적용)
        # gnn_out을 시간축으로 pooling해서 SNN의 decay 제어 피처로 사용
        geo_temp_feat = gnn_out.reshape(B, T, -1)
        geo_temp_feat = geo_temp_feat[:, :valid_T, :]
        geo_temp_feat = geo_temp_feat.reshape(B, seq_len, -1, geo_temp_feat.shape[-1]).mean(axis=2)
        geo_temp_feat = nn.Dense(128)(geo_temp_feat) # (B, seq_len, 128)

        ScanLIF = nn.scan(
            GeoTempDrivenLIF,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1
        )
        # 🔥 수정 완료: 시간축이 살아있는 seq_input과 geo_temp_feat를 SNN에 흘려보냄
        _, spk_seq = ScanLIF()(init_mem, (seq_input, geo_temp_feat))

        # 시간축 어텐션
        temp_attn = nn.softmax(nn.Dense(1)(spk_seq), axis=1)
        feat = jnp.sum(spk_seq * temp_attn, axis=1)
        
        feat = nn.Dropout(rate=0.3, deterministic=not train)(feat)
        feat = nn.Dense(64)(feat)
        feat = nn.relu(feat)
        logits = nn.Dense(4)(feat)
        
        return logits, jnp.mean(spk_seq)

# =========================================================
# 3. Training Loop
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
# 🔥 수정 완료: Riemannian Backbone을 모델 초기화 시점에 전달
model = Model(log_cov_backbone=jnp.array(log_cov_tr))

# 실제 데이터 차원(C_tr, T_tr)을 가져와서 더미 데이터 생성
dummy_x = jnp.ones((1, C_tr, T_tr))
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
        all_preds = []
        all_frs = []
        
        for Xb, yb in test_loader:
            preds, vf = eval_step_preds(state, jnp.array(Xb.numpy()))
            all_preds.extend(np.array(preds))
            all_frs.append(vf)
            
        correct = np.sum(np.array(all_preds) == yte)
        total = len(yte)
                
        avg_val_acc = (correct / total) * 100
        avg_train_acc = np.mean(t_accs) * 100
        avg_loss = np.mean(t_losses)
        
        print(f"[{epoch:3d}] Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | FR: {np.mean(all_frs):.4f}")
        best_acc = max(best_acc, avg_val_acc)

print(f"🔥 BEST TEST ACC (Riemannian-TCN-GNN-SNN): {best_acc:.2f}%")