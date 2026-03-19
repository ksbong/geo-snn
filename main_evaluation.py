import os
import mne
mne.set_log_level('ERROR')
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
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

BATCH_SIZE = 128
train_loader = DataLoader(TensorDataset(torch.tensor(Xtr_scaled), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(Xte_scaled), torch.tensor(yte)), batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 2. 🚀 아키텍처: Graph Spectral Neuromorphic Model 
# =========================================================
@jax.custom_vjp
def spike(x): 
    return (x > 0).astype(jnp.float32)

def fwd(x): 
    return spike(x), x

def bwd(res, g): 
    alpha = 2.0
    return (g * (alpha / 2.0) / (1.0 + (alpha * res)**2),)

spike.defvjp(fwd, bwd)

class LIFCell(nn.Module):
    hidden_dim: int
    v_thresh: float = 0.3 # 🔥 임계값 완화로 데드 뉴런 방지

    @nn.compact
    def __call__(self, state, x_t):
        v, z = state
        decay = nn.sigmoid(self.param('decay', nn.initializers.constant(0.0), (self.hidden_dim,)))
        v = v * decay * (1.0 - z) + x_t
        z_out = spike(v - self.v_thresh)
        return (v, z_out), z_out

class Model(nn.Module):
    @nn.compact
    def __call__(self, X, train):
        if train:
            X = X + jax.random.normal(self.make_rng('dropout'), X.shape) * 0.05
            
        B, C, T = X.shape
        
        # Step 1: Dynamic Spatial Graphing
        d_model = 32
        W_q = self.param('W_q', nn.initializers.glorot_normal(), (T, d_model))
        W_k = self.param('W_k', nn.initializers.glorot_normal(), (T, d_model))
        
        Q = jnp.matmul(X, W_q)
        K = jnp.matmul(X, W_k)
        
        attn = jnp.matmul(Q, jnp.transpose(K, (0, 2, 1))) / (jnp.sqrt(d_model) * 2.0)
        A = nn.softmax(attn, axis=-1)
        A = (A + jnp.transpose(A, (0, 2, 1))) / 2.0
        
        D = jnp.eye(C) * jnp.sum(A, axis=-1, keepdims=True)
        L = D - A + jnp.eye(C) * 1e-4
        vals, U = jax.vmap(jnp.linalg.eigh)(L)
        
        # Step 2: Graph Fourier Transform (GFT)
        U_T = jnp.transpose(U, (0, 2, 1))
        X_hat = jnp.matmul(U_T, X)
        X_hat_seq = jnp.transpose(X_hat, (0, 2, 1))
        
        # Step 3: Spectral Delta Encoding & Normalization
        delta_X = jnp.concatenate([
            jnp.zeros((B, 1, C)), 
            X_hat_seq[:, 1:, :] - X_hat_seq[:, :-1, :]
        ], axis=1)
        delta_X = nn.LayerNorm()(delta_X) 
        
        # Step 4: Deep LIF SNN
        seq_in_1 = nn.Dense(128)(delta_X)
        seq_in_1 = nn.LayerNorm()(seq_in_1)
        
        init_v1 = jnp.zeros((B, 128))
        init_z1 = jnp.zeros((B, 128))
        
        ScanLIF1 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        _, spk_seq1 = ScanLIF1(hidden_dim=128)((init_v1, init_z1), seq_in_1)
        
        seq_in_2 = nn.Dense(64)(spk_seq1)
        seq_in_2 = nn.LayerNorm()(seq_in_2) * 0.5 
        
        init_v2 = jnp.zeros((B, 64))
        init_z2 = jnp.zeros((B, 64))
        
        ScanLIF2 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        _, spk_seq2 = ScanLIF2(hidden_dim=64)((init_v2, init_z2), seq_in_2)
        
        # Step 5: Attention Readout over Spikes
        temp_attn = nn.Dense(1)(spk_seq2)
        temp_attn = nn.softmax(temp_attn, axis=1)
        
        feat = jnp.sum(spk_seq2 * temp_attn, axis=1)
        feat_drop = nn.Dropout(rate=0.4, deterministic=not train)(feat)
        
        logits = nn.Dense(4)(feat_drop)
        mean_fr = (jnp.mean(spk_seq1) + jnp.mean(spk_seq2)) / 2.0
        
        # 🔥 Feature 시각화를 위해 딕셔너리로 추출할 변수들 같이 리턴
        intermediates = {
            'A': A,
            'spk_seq2': spk_seq2,
            'temp_attn': temp_attn,
            'feat': feat 
        }
        
        return logits, mean_fr, intermediates

# =========================================================
# 3. Training Loop
# =========================================================
def loss_fn(params, state, X, y, dropout_rng, fr_weight):
    logits, fr, _ = state.apply_fn({'params': params}, X, train=True, rngs={'dropout': dropout_rng})
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
    logits, fr, intermediates = state.apply_fn({'params': state.params}, X, train=False)
    preds = jnp.argmax(logits, -1)
    return preds, fr, intermediates

rng = jax.random.PRNGKey(42)
rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
model = Model()

dummy_x = jnp.ones((1, C_tr, T_tr))
params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, dummy_x, train=False)['params']

steps_per_epoch = len(train_loader)
lr_sched = optax.exponential_decay(init_value=1e-3, transition_steps=steps_per_epoch * 10, decay_rate=0.95)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=lr_sched, weight_decay=1e-3))

best_acc = 0

print("🚀 본격적인 학습 시작...")
for epoch in range(1, 101): # 빠른 확인을 위해 100 epoch로 세팅
    current_fr_weight = min(5.0, 0.1 + (epoch / 50.0) * 4.9) if epoch <= 50 else 5.0
    fr_weight_tensor = jnp.array(current_fr_weight, dtype=jnp.float32)
    
    t_accs, t_losses = [], []
    for Xb, yb in train_loader:
        state, loss, acc, _, rng = train_step(state, jnp.array(Xb.numpy()), jnp.array(yb.numpy()), rng, fr_weight_tensor)
        t_accs.append(acc)
        t_losses.append(loss)

    if epoch % 10 == 0:
        all_preds, all_frs = [], []
        
        for Xb, yb in test_loader:
            preds, vf, _ = eval_step_preds(state, jnp.array(Xb.numpy()))
            all_preds.extend(np.array(preds))
            all_frs.append(vf)
            
        correct = np.sum(np.array(all_preds) == yte)
        total = len(yte)
                
        avg_val_acc = (correct / total) * 100
        avg_train_acc = np.mean(t_accs) * 100
        avg_loss = np.mean(t_losses)
        
        print(f"[{epoch:3d}] Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | FR: {np.mean(all_frs):.4f}")
        best_acc = max(best_acc, avg_val_acc)

print(f"🔥 BEST TEST ACC: {best_acc:.2f}%")

# =========================================================
# 4. 📊 시각화: 논문용 피겨(Figure) 생성 (학습 종료 후)
# =========================================================
print("🎨 데이터 시각화 추출 중...")

# 테스트 셋 전체에 대해 피쳐 추출
all_feats, all_labels = [], []
sample_A, sample_spk, sample_attn = None, None, None

for Xb, yb in test_loader:
    _, _, inter = eval_step_preds(state, jnp.array(Xb.numpy()))
    all_feats.append(np.array(inter['feat']))
    all_labels.append(yb.numpy())
    
    # 시각화용 샘플 하나만 저장 (첫 번째 배치 기준)
    if sample_A is None:
        sample_A = np.array(inter['A'][0])           # (C, C)
        sample_spk = np.array(inter['spk_seq2'][0])  # (T, 64)
        sample_attn = np.array(inter['temp_attn'][0])# (T, 1)

all_feats = np.concatenate(all_feats, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
class_names = ['Left', 'Right', 'Hands', 'Feet']

fig = plt.figure(figsize=(20, 10))

# 1. t-SNE (Latent Space)
ax1 = plt.subplot(2, 2, 1)
tsne = TSNE(n_components=2, random_state=42)
feats_2d = tsne.fit_transform(all_feats)
scatter = ax1.scatter(feats_2d[:, 0], feats_2d[:, 1], c=all_labels, cmap='viridis', alpha=0.7)
ax1.set_title("t-SNE of Latent Space (Feature Separation)", fontsize=14)
legend1 = ax1.legend(*scatter.legend_elements(), title="Classes")
for i, text in enumerate(legend1.get_texts()):
    text.set_text(class_names[i])

# 2. Learned Adjacency Matrix (Heatmap)
ax2 = plt.subplot(2, 2, 2)
sns.heatmap(sample_A, cmap='magma', ax=ax2)
ax2.set_title("Dynamic Learned Adjacency Matrix (A)", fontsize=14)
ax2.set_xlabel("Channels")
ax2.set_ylabel("Channels")

# 3. SNN Raster Plot
ax3 = plt.subplot(2, 2, 3)
time_steps, neuron_idx = np.where(sample_spk > 0)
ax3.scatter(time_steps, neuron_idx, s=5, c='black', marker='|')
ax3.set_title("SNN Spike Raster Plot (Layer 2)", fontsize=14)
ax3.set_xlim(0, sample_spk.shape[0])
ax3.set_ylim(0, 64)
ax3.set_xlabel("Time Step")
ax3.set_ylabel("Neuron Index")

# 4. Temporal Attention Weights
ax4 = plt.subplot(2, 2, 4)
ax4.plot(sample_attn, color='blue', linewidth=2)
ax4.set_title("Temporal Attention Weights over Time", fontsize=14)
ax4.set_xlim(0, len(sample_attn))
ax4.set_xlabel("Time Step")
ax4.set_ylabel("Attention Score")

plt.tight_layout()
plt.show()
print("✅ 완료! 창을 확인해봐.")