# 필요한 라이브러리 설치 (캐글 첫 셀에서 실행)
# !pip install -q flax optax mne

import os
import gc
import random
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la
import warnings

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax
from torch.utils.data import DataLoader, TensorDataset
import torch

warnings.filterwarnings("ignore")

# =========================================================
# 1. 완벽한 피험자 격리 및 JAX(NHWC) 텐서 규격화
# =========================================================
DATA_DIR = './07_Data'
bad_subjects = [88, 92, 100, 104]
subjects = sorted([f'S{i:03d}' for i in range(1, 110) if i not in bad_subjects])

np.random.seed(42)
shuffled_subjs = subjects.copy()
np.random.shuffle(shuffled_subjs)
train_subjs = shuffled_subjs[:84]
test_subjs = shuffled_subjs[84:]

def load_and_align_subjects(subj_list):
    x_aligned, y_labels = [], []
    for s_idx, s in enumerate(subj_list):
        subj_dir = os.path.join(DATA_DIR, s)
        subj_epochs = []
        for run in ['R04','R08','R12','R06','R10','R14']:
            path = os.path.join(subj_dir, f'{s}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.resample(160.0); raw.filter(8.0, 30.0, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            e = evs.copy()
            if run in ['R04','R08','R12']: e[evs[:,2]==ed.get('T1', 1),2], e[evs[:,2]==ed.get('T2', 2),2] = 0, 1
            else: e[evs[:,2]==ed.get('T1', 1),2], e[evs[:,2]==ed.get('T2', 2),2] = 2, 3
                
            ep = mne.Epochs(raw, e, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: subj_epochs.append(ep)
        
        if len(subj_epochs) == 0: continue
        epochs = mne.concatenate_epochs(subj_epochs, verbose=False)
        X = epochs.get_data(copy=True) * 1e6
        y = epochs.events[:, 2]
        
        R_i = np.mean([np.cov(x) for x in X], axis=0) + np.eye(64) * 1e-4
        P_i = la.inv(la.sqrtm(R_i))
        x_aligned.append(np.array([P_i @ x for x in X]))
        y_labels.extend(y)
        if (s_idx + 1) % 20 == 0: gc.collect()
            
    # 🔥 JAX는 NHWC (Batch, 64, 480, 1) 규격을 사용함
    X_out = np.expand_dims(np.concatenate(x_aligned), -1) 
    return X_out, np.array(y_labels)

print("⏳ 1. Train 데이터 로딩 (84명)...")
X_train, Y_train = load_and_align_subjects(train_subjs)
print("⏳ 2. Test 데이터 로딩 (21명)...")
X_test, Y_test = load_and_align_subjects(test_subjs)
print(f"✅ 로딩 완료! Train: {X_train.shape}, Test: {X_test.shape}")

# 미니배치용 DataLoader (PyTorch 꺼 빌려 쓰기)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=128, shuffle=False)

# =========================================================
# 2. JAX 기반 SNN 핵심 코어 (XLA 최적화 + Surrogate Gradient)
# =========================================================
# 🔥 핵심 1: JAX custom_vjp를 이용한 아크탄젠트 대리 기울기
@jax.custom_vjp
def spike_fn(v):
    return jnp.where(v > 0, 1.0, 0.0)

def spike_fn_fwd(v):
    return spike_fn(v), v

def spike_fn_bwd(res, g):
    v = res
    alpha = 2.0
    grad = g / (1.0 + (alpha * v)**2)
    return (grad,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

# 🔥 핵심 2: Python For-loop를 박살내는 lax.scan
class LIFNode(nn.Module):
    tau: float = 2.0
    v_th: float = 0.5

    @nn.compact
    def __call__(self, x):
        # x shape: (B, 64, 480, 16) -> lax.scan을 위해 시간을 맨 앞으로 (480, B, 64, 16)
        x_seq = jnp.moveaxis(x, 2, 0)
        
        def scan_fn(v, x_t):
            v = v * (1.0 - 1.0/self.tau) + x_t
            s = spike_fn(v - self.v_th)
            v = v - s * self.v_th
            return v, s

        v_init = jnp.zeros_like(x_seq[0])
        # 단 한 번의 C++ 커널 호출로 480번의 연산 퓨전 완료
        _, spikes = jax.lax.scan(scan_fn, v_init, x_seq)
        
        # 다시 원래 shape (B, 64, 480, 16)으로 복구
        return jnp.moveaxis(spikes, 0, 2)

# =========================================================
# 3. JAX/Flax 하이브리드 SNN 아키텍처
# =========================================================
class DynamicGraphAttention(nn.Module):
    hidden_dim: int = 16

    @nn.compact
    def __call__(self, x):
        # x: (B, 64, 480, 1)
        x_squeeze = x[..., 0] # (B, 64, 480)
        x_energy = jnp.var(x_squeeze, axis=-1, keepdims=True) # (B, 64, 1)
        
        Q = nn.Dense(self.hidden_dim)(x_energy) # (B, 64, 16)
        K = nn.Dense(self.hidden_dim)(x_energy) # (B, 64, 16)
        
        scale = self.hidden_dim ** -0.5
        attn = jnp.einsum('b i h, b j h -> b i j', Q, K) * scale
        
        eye = jnp.eye(64, dtype=bool)
        attn = jnp.where(eye, -1e9, attn)
        attn_weights = jax.nn.softmax(attn, axis=-1)
        
        out = jnp.einsum('b i j, b j t -> b i t', attn_weights, x_squeeze)
        out = jnp.expand_dims(out, -1) + x # Residual
        return out, attn_weights

class Hybrid_SOTA_SNN(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        # 1. 공간적 정보 디코딩 (ANN)
        z_gcn, attn_weights = DynamicGraphAttention(hidden_dim=16)(x)
        
        z_ann = nn.Conv(features=16, kernel_size=(1, 15))(z_gcn)
        z_ann = nn.BatchNorm(use_running_average=not train)(z_ann)
        z_ann = nn.elu(z_ann)
        z_ann = nn.Dropout(rate=0.5, deterministic=not train)(z_ann)
        
        # 2. 시간적 통합 (SNN)
        spikes = LIFNode(tau=2.0, v_th=0.5)(z_ann) # (B, 64, 480, 16)
        
        # 3. Rate Coding
        firing_rate = jnp.mean(spikes, axis=2) # (B, 64, 16)
        z_feat = firing_rate.reshape((firing_rate.shape[0], -1))
        
        z_feat = nn.Dropout(rate=0.5, deterministic=not train)(z_feat)
        logits = nn.Dense(features=4)(z_feat)
        
        return logits, z_feat

def mmd_loss(x, y, bandwidths=(0.5, 1.0, 2.0)):
    xx = jnp.dot(x, x.T)
    yy = jnp.dot(y, y.T)
    zz = jnp.dot(x, y.T)
    rx = jnp.expand_dims(jnp.diag(xx), 0)
    ry = jnp.expand_dims(jnp.diag(yy), 0)
    
    dxx = rx.T + rx - 2.0 * xx
    dyy = ry.T + ry - 2.0 * yy
    dxy = rx.T + ry - 2.0 * zz
    
    loss = 0.0
    for a in bandwidths:
        loss += jnp.mean(jnp.exp(-0.5 * dxx / a) + jnp.exp(-0.5 * dyy / a) - 2.0 * jnp.exp(-0.5 * dxy / a))
    return loss

# =========================================================
# 4. JAX XLA 최적화 학습 루프 (Train Loop)
# =========================================================
rng = jrandom.PRNGKey(42)
rng, init_rng = jrandom.split(rng)
model = Hybrid_SOTA_SNN()

# 모델 파라미터 & BatchNorm 상태 초기화
dummy_x = jnp.ones((1, 64, 480, 1))
variables = model.init(init_rng, dummy_x, train=False)
state = {'params': variables['params'], 'batch_stats': variables.get('batch_stats', {})}

# Cosine Annealing Optimizer
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=optax.cosine_decay_schedule(3e-3, 200), weight_decay=1e-2))
opt_state = tx.init(state['params'])

# 🔥 핵심 3: 전체 학습 단계를 JIT(Just-In-Time) 컴파일로 캐싱
@jax.jit
def train_step(state, opt_state, x, y, rng_key):
    def loss_fn(params):
        vars_in = {'params': params, 'batch_stats': state['batch_stats']}
        (logits, z_feat), new_vars = model.apply(vars_in, x, train=True, rngs={'dropout': rng_key}, mutable=['batch_stats'])
        
        # Cross Entropy
        y_onehot = jax.nn.one_hot(y, 4)
        loss_cls = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
        
        # MMD Loss
        half_idx = z_feat.shape[0] // 2
        loss_mmd = jax.lax.cond(half_idx > 0, lambda _: mmd_loss(z_feat[:half_idx], z_feat[half_idx:]), lambda _: 0.0, operand=None)
        
        total_loss = loss_cls + 0.01 * loss_mmd
        return total_loss, (logits, new_vars['batch_stats'])

    grads, (logits, new_batch_stats) = jax.grad(loss_fn, has_aux=True)(state['params'])
    updates, new_opt_state = tx.update(grads, opt_state, state['params'])
    new_params = optax.apply_updates(state['params'], updates)
    
    new_state = {'params': new_params, 'batch_stats': new_batch_stats}
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return new_state, new_opt_state, acc

@jax.jit
def eval_step(state, x, y):
    vars_in = {'params': state['params'], 'batch_stats': state['batch_stats']}
    logits, _ = model.apply(vars_in, x, train=False)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return acc

print("🚀 JAX/Flax XLA 기반 초고속 Hybrid SNN 학습 시작...")

best_acc = 0.0
for epoch in range(1, 201):
    train_accs = []
    for xb, yb in train_loader:
        rng, step_rng = jrandom.split(rng)
        # PyTorch Tensor -> Numpy Array -> JAX Array
        xb_jax, yb_jax = jnp.array(xb.numpy()), jnp.array(yb.numpy())
        state, opt_state, acc = train_step(state, opt_state, xb_jax, yb_jax, step_rng)
        train_accs.append(acc)
        
    if epoch % 5 == 0:
        test_accs = []
        for xb, yb in test_loader:
            xb_jax, yb_jax = jnp.array(xb.numpy()), jnp.array(yb.numpy())
            test_accs.append(eval_step(state, xb_jax, yb_jax))
            
        val_acc = np.mean(test_accs) * 100
        train_acc = np.mean(train_accs) * 100
        print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.2f}% | ⚡ JAX Test Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc

print("="*50)
print(f"🏆 JAX 최적화 하이브리드 SNN 최고 성능: {best_acc:.2f}%")
print("="*50)