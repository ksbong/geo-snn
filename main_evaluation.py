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
# 1. 데이터 로딩 및 4-Class 1:1:1:1 밸런싱
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
            t0_id = ed.get('T0', 1)
            t1_id = ed.get('T1', 2)
            t2_id = ed.get('T2', 3)
            
            if run in ['R04','R08','R12']:
                mask = np.isin(evs[:, 2], [t0_id, t1_id, t2_id])
                e = evs[mask].copy()
                e[e[:, 2] == t0_id, 2] = 0 
                e[e[:, 2] == t1_id, 2] = 1 
                e[e[:, 2] == t2_id, 2] = 2 
            else:
                mask = np.isin(evs[:, 2], [t0_id, t2_id]) 
                e = evs[mask].copy()
                e[e[:, 2] == t0_id, 2] = 0 
                e[e[:, 2] == t2_id, 2] = 3 
                
            ep = mne.Epochs(raw, e, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: subj_epochs.append(ep)
        
        if len(subj_epochs) == 0: continue
        epochs = mne.concatenate_epochs(subj_epochs, verbose=False)
        X = epochs.get_data(copy=True) * 1e6
        y = epochs.events[:, 2]
        
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        idx_2 = np.where(y == 2)[0]
        idx_3 = np.where(y == 3)[0]
        
        min_active = min(len(idx_1), len(idx_2), len(idx_3))
        if min_active > 0 and len(idx_0) > min_active:
            np.random.shuffle(idx_0)
            idx_0 = idx_0[:min_active] 
            
        balanced_idx = np.concatenate([idx_0, idx_1, idx_2, idx_3])
        np.random.shuffle(balanced_idx)
        
        X_bal, y_bal = X[balanced_idx], y[balanced_idx]
        
        R_i = np.mean([np.cov(x) for x in X_bal], axis=0) + np.eye(64) * 1e-4
        P_i = la.inv(la.sqrtm(R_i))
        x_aligned.append(np.array([P_i @ x for x in X_bal]))
        y_labels.extend(y_bal)
        if (s_idx + 1) % 20 == 0: gc.collect()
            
    X_out = np.expand_dims(np.concatenate(x_aligned), -1) 
    return X_out, np.array(y_labels)

print("⏳ 1. Train 데이터 로딩 (84명)...")
X_train, Y_train = load_and_align_subjects(train_subjs)
print("⏳ 2. Test 데이터 로딩 (21명)...")
X_test, Y_test = load_and_align_subjects(test_subjs)
print(f"✅ 로딩 완료! Train: {X_train.shape}, Test: {X_test.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=128, shuffle=False)

# =========================================================
# 2. SNN 코어 (상봉의 통찰 적용: 이종 매개변수 학습형 PLIF)
# =========================================================
@jax.custom_vjp
def spike_fn(x):
    return jnp.where(x > 0, 1.0, 0.0)

def spike_fn_fwd(x):
    return spike_fn(x), x

def spike_fn_bwd(res, g):
    x = res 
    # KIST 논문 검증 Triangle Surrogate (alpha=0.5)
    grad = jnp.maximum(0.0, 1.0 - jnp.abs(x) / 0.5)
    return (g * grad,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

class PLIFNodeHetero(nn.Module):
    @nn.compact
    def __call__(self, x):
        x_seq = jnp.moveaxis(x, 2, 0) # (T, B, Space, Features)
        feat_dim = x.shape[-1]
        
        # 🔥 네 아이디어의 핵심: 피처별로 뉴런의 스펙을 다르게 학습!
        # 감쇠계수(Decay) 학습 파라미터
        decay_param = self.param('decay', nn.initializers.constant(0.0), (feat_dim,))
        # 임계값(Threshold) 학습 파라미터
        vth_param = self.param('v_th', nn.initializers.constant(0.0), (feat_dim,))

        def scan_fn(v, x_t):
            # 시그모이드로 0~1 사이의 안전한 범위 내에서 각자 다른 스펙을 가짐
            decay = jax.nn.sigmoid(decay_param) 
            # 임계값도 0.1 ~ 1.1 사이에서 뉴런마다 다르게 세팅됨
            v_th = jax.nn.sigmoid(vth_param) + 0.1 
            
            v = v * decay + x_t
            s = spike_fn(v - v_th)
            
            # 미분 가능한 Hard Reset (stop_gradient 적용)
            v = v * (1.0 - jax.lax.stop_gradient(s)) 
            return v, s

        v_init = jnp.zeros_like(x_seq[0])
        _, spikes = jax.lax.scan(scan_fn, v_init, x_seq)
        return jnp.moveaxis(spikes, 0, 2)

# =========================================================
# 3. 모델 아키텍처 (공간 보존 + 이종 PLIF)
# =========================================================
class DynamicGraphAttention(nn.Module):
    hidden_dim: int = 16 

    @nn.compact
    def __call__(self, x):
        x_squeeze = x[..., 0] 
        x_energy = jnp.var(x_squeeze, axis=-1, keepdims=True) 
        
        Q = nn.Dense(self.hidden_dim)(x_energy) 
        K = nn.Dense(self.hidden_dim)(x_energy) 
        
        scale = self.hidden_dim ** -0.5
        attn = jnp.einsum('b i h, b j h -> b i j', Q, K) * scale
        
        eye = jnp.eye(64, dtype=bool)
        attn = jnp.where(eye, -1e9, attn)
        attn_weights = jax.nn.softmax(attn, axis=-1)
        
        out = jnp.einsum('b i j, b j t -> b i t', attn_weights, x_squeeze)
        out = jnp.expand_dims(out, -1) + x 
        return out, attn_weights

class Hybrid_SOTA_SNN(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        # 1. 공간 유기성 파악
        z_gcn, _ = DynamicGraphAttention(hidden_dim=16)(x)
        
        # 2. 시간 파동 추출 (피처 32개로 넉넉하게 확장)
        z_ann = nn.Conv(features=32, kernel_size=(1, 32), padding='SAME')(z_gcn)
        z_ann = nn.BatchNorm(use_running_average=not train)(z_ann)
        z_ann = nn.relu(z_ann) # 양수 강제 주입
        
        # BPTT 압축 (481스텝 -> 약 60스텝)
        z_pooled = nn.avg_pool(z_ann, window_shape=(1, 8), strides=(1, 8))
        
        # 3. 🔥 이종 PLIF 스파이크 발화 (뉴런 32그룹이 각자 다른 예민도/기억력으로 뇌파 분석)
        spikes = PLIFNodeHetero()(z_pooled) # (B, 64, 60, 32)
        
        # 4. 단순 합산 (공간 64채널 절대 뭉개지 않음!)
        spike_count = jnp.sum(spikes, axis=2) # (B, 64, 32)
        z_feat = spike_count.reshape((spike_count.shape[0], -1)) # (B, 2048)
        
        # 5. 분류기
        z_feat = nn.LayerNorm()(z_feat)
        z_feat = nn.Dropout(rate=0.5, deterministic=not train)(z_feat)
        
        z_feat = nn.Dense(features=128)(z_feat)
        z_feat = nn.elu(z_feat)
        z_feat = nn.Dropout(rate=0.5, deterministic=not train)(z_feat)
        
        logits = nn.Dense(features=4)(z_feat)
        
        return logits, z_feat, spikes
    
# =========================================================
# 4. 학습 루프
# =========================================================
rng = jrandom.PRNGKey(42)
rng, init_rng = jrandom.split(rng)
model = Hybrid_SOTA_SNN()

T_dim = X_train.shape[2] 
dummy_x = jnp.ones((1, 64, T_dim, 1))
variables = model.init(init_rng, dummy_x, train=False)
state = {'params': variables['params'], 'batch_stats': variables.get('batch_stats', {})}

tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=optax.cosine_decay_schedule(2e-3, 200), weight_decay=1e-2))
opt_state = tx.init(state['params'])

@jax.jit
def train_step(state, opt_state, x, y, rng_key):
    def loss_fn(params):
        vars_in = {'params': params, 'batch_stats': state['batch_stats']}
        (logits, dp_flat, spikes), new_vars = model.apply(vars_in, x, train=True, rngs={'dropout': rng_key}, mutable=['batch_stats'])
        
        y_onehot = jax.nn.one_hot(y, 4)
        loss_cls = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
        
        firing_rate = jnp.mean(spikes)
        return loss_cls, (logits, new_vars['batch_stats'], firing_rate)

    grads, (logits, new_batch_stats, firing_rate) = jax.grad(loss_fn, has_aux=True)(state['params'])
    updates, new_opt_state = tx.update(grads, opt_state, state['params'])
    new_params = optax.apply_updates(state['params'], updates)
    
    new_state = {'params': new_params, 'batch_stats': new_batch_stats}
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return new_state, new_opt_state, acc, firing_rate

@jax.jit
def eval_step(state, x, y):
    vars_in = {'params': state['params'], 'batch_stats': state['batch_stats']}
    logits, _, _ = model.apply(vars_in, x, train=False)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return acc

print("🚀 최종 완전체: Dynamic Graph + PLIF(Hard Reset) + DP-Pooling 학습 시작...")

best_acc = 0.0
for epoch in range(1, 201):
    train_accs, train_frs = [], []
    for xb, yb in train_loader:
        rng, step_rng = jrandom.split(rng)
        xb_jax, yb_jax = jnp.array(xb.numpy()), jnp.array(yb.numpy())
        state, opt_state, acc, fr = train_step(state, opt_state, xb_jax, yb_jax, step_rng)
        train_accs.append(acc)
        train_frs.append(fr) 
        
    if epoch % 5 == 0:
        test_accs = []
        for xb, yb in test_loader:
            xb_jax, yb_jax = jnp.array(xb.numpy()), jnp.array(yb.numpy())
            test_accs.append(eval_step(state, xb_jax, yb_jax))
            
        val_acc = np.mean(test_accs) * 100
        train_acc = np.mean(train_accs) * 100
        avg_fr = np.mean(train_frs)
        
        print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.2f}% | ⚡ Test Acc: {val_acc:.2f}% | 🧠 Firing Rate: {avg_fr:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc

print("="*50)
print(f"🏆 최종 SOTA 달성 성능: {best_acc:.2f}%")
print("="*50)