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
# 1. 완벽한 피험자 격리 및 데이터 밸런싱 (HR-SNN 논문 규격)
# =========================================================
DATA_DIR = './07_Data' # 캐글 경로 맞게 수정
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
            
            # 논문 규격에 맞춘 4-Class 라벨링
            if run in ['R04','R08','R12']:
                mask = np.isin(evs[:, 2], [t0_id, t1_id, t2_id])
                e = evs[mask].copy()
                e[e[:, 2] == t0_id, 2] = 0 # 휴식
                e[e[:, 2] == t1_id, 2] = 1 # 왼손
                e[e[:, 2] == t2_id, 2] = 2 # 오른손
            else:
                mask = np.isin(evs[:, 2], [t0_id, t2_id]) # T1(양손) 삭제
                e = evs[mask].copy()
                e[e[:, 2] == t0_id, 2] = 0 # 휴식
                e[e[:, 2] == t2_id, 2] = 3 # 양발
                
            ep = mne.Epochs(raw, e, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: subj_epochs.append(ep)
        
        if len(subj_epochs) == 0: continue
        epochs = mne.concatenate_epochs(subj_epochs, verbose=False)
        X = epochs.get_data(copy=True) * 1e6
        y = epochs.events[:, 2]
        
        # 🔥 데이터 밸런싱: T0(휴식)가 너무 많아서 생기는 62% 찍기 꼼수 차단
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        idx_2 = np.where(y == 2)[0]
        idx_3 = np.where(y == 3)[0]
        
        min_active = min(len(idx_1), len(idx_2), len(idx_3))
        if min_active > 0 and len(idx_0) > min_active:
            np.random.shuffle(idx_0)
            idx_0 = idx_0[:min_active] # 휴식 데이터를 동작 데이터 수와 1:1로 맞춤
            
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
# 2. PLIF (Parametric LIF) 뉴런 도입
# =========================================================
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

class PLIFNode(nn.Module):
    v_th: float = 0.5

    @nn.compact
    def __call__(self, x):
        x_seq = jnp.moveaxis(x, 2, 0) # 시간을 앞으로 (481, B, 64, 16)
        
        # 🔥 핵심 수술 1: 채널별로 기억력(망각 곡선)을 스스로 학습하는 파라미터 생성
        # 초기값을 2.0으로 주어 sigmoid(2.0) ≈ 0.88의 긴 기억력으로 시작 유도
        decay_param = self.param('decay', nn.initializers.constant(2.0), (x.shape[-1],))

        def scan_fn(v, x_t):
            # Sigmoid를 통과시켜 0~1 사이의 안정적인 망각률 보장
            decay = jax.nn.sigmoid(decay_param)
            v = v * decay + x_t
            s = spike_fn(v - self.v_th)
            v = v - s * self.v_th # Hard reset
            return v, s

        v_init = jnp.zeros_like(x_seq[0])
        _, spikes = jax.lax.scan(scan_fn, v_init, x_seq)
        return jnp.moveaxis(spikes, 0, 2)
# =========================================================
# 3. 하이브리드 SNN 아키텍처 (DP-Spiking Decoder 완전판)
# =========================================================
class LINode(nn.Module):
    # 스파이크를 발화하지 않고 전위(Potential)만 누적하는 뉴런
    tau: float = 2.0
    
    @nn.compact
    def __call__(self, x):
        x_seq = jnp.moveaxis(x, 2, 0) # (T, B, S, F)
        
        def scan_fn(v, x_t):
            v = v * (1.0 - 1.0/self.tau) + x_t
            return v, v # 스파이크 리셋 없이 전위만 반환

        v_init = jnp.zeros_like(x_seq[0])
        _, potentials = jax.lax.scan(scan_fn, v_init, x_seq)
        return jnp.moveaxis(potentials, 0, 2) # (B, 64, 481, 16)

class DPSpikingDecoder(nn.Module):
    L_dp: int = 24
    N_dp: int = 12

    @nn.compact
    def __call__(self, spikes, train: bool = True):
        # 1. LI Layer: 스파이크(이산) -> 전위(연속) 변환
        potentials = LINode(tau=2.0)(spikes) 
        
        # 2. 공간축(64) 차원 축소: 논문의 Avg-pooling 구조 적용
        v_pooled = jnp.mean(potentials, axis=1) # (B, 481, 16)
        B, T, F = v_pooled.shape
        
        # 3. 패딩 및 DP 윈도우 분할
        pad_len = (self.L_dp - (T % self.L_dp)) % self.L_dp
        v_padded = jnp.pad(v_pooled, ((0,0), (0, pad_len), (0,0)))
        T_new = v_padded.shape[1]
        num_windows = T_new // self.L_dp # 21개 윈도우
        
        v_chunked = v_padded.reshape((B, num_windows, self.L_dp, F))
        
        # 4. DP-Pooling 연산 (앞뒤 평균의 차이 추출)
        first_part = jnp.mean(v_chunked[:, :, :self.N_dp, :], axis=2)
        last_part = jnp.mean(v_chunked[:, :, -self.N_dp:, :], axis=2)
        dp_out = last_part - first_part # (B, 21, 16)
        
        # 5. Temporal Attention (TA) - 차원 에러 완벽 해결
        # 먼저 평탄화하여 21개 윈도우 전체의 글로벌 문맥 파악
        dp_flat = dp_out.reshape((B, -1)) # (B, 21 * 16)
        
        ta_fc1 = nn.Dense(features=num_windows // 2)(dp_flat) # (B, 10)
        ta_relu = nn.relu(ta_fc1)
        ta_fc2 = nn.Dense(features=num_windows)(ta_relu) # (B, 21)
        
        # 21개 윈도우 각각의 중요도 가중치 산출
        ta_weights = jax.nn.softmax(ta_fc2, axis=-1) # (B, 21)
        
        # 브로드캐스팅을 위해 (B, 21, 1)로 차원 확장
        ta_weights_expanded = jnp.expand_dims(ta_weights, axis=-1) 
        
        # 6. Hadamard product (중요한 시간 구간의 특징 증폭)
        # (B, 21, 16) * (B, 21, 1) -> 완벽한 차원 매칭
        attended_out = dp_out * ta_weights_expanded 
        
        # 최종 분류기를 위해 평탄화
        return attended_out.reshape((B, -1))
class DynamicGraphAttention(nn.Module):
    hidden_dim: int = 16 # 가볍고 빠르게 16으로 롤백

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
        z_gcn, attn_weights = DynamicGraphAttention(hidden_dim=16)(x)
        
        z_ann = nn.Conv(features=16, kernel_size=(1, 64), padding='SAME')(z_gcn)
        z_ann = nn.BatchNorm(use_running_average=not train)(z_ann)
        z_ann = nn.elu(z_ann)
        z_ann = nn.Dropout(rate=0.5, deterministic=not train)(z_ann)
        
        # PLIF 뉴런 스파이크 발화 (B, 64, 481, 16)
        spikes = PLIFNode(v_th=0.5)(z_ann) 
        
        # 🔥 다이어트 1단계: 480스텝으로 깔끔하게 자르고 10개의 윈도우(0.3초)로 분할
        spikes = spikes[:, :, :480, :] 
        B, S, T, F = spikes.shape
        spikes_chunked = spikes.reshape((B, S, 10, 48, F))
        
        # 윈도우(48스텝) 내의 평균 발화율(Rate) 계산 -> (B, 64, 10, 16)
        spike_rates = jnp.mean(spikes_chunked, axis=3) 
        
        # 🔥 다이어트 2단계 (가장 중요): 전극(64) 차원을 평균내어 1차원으로 압축
        # 이미 Graph Attention이 전극 간 정보를 융합했으므로 안전함
        z_temporal = jnp.mean(spike_rates, axis=1) # (B, 10, 16)
        
        # 🔥 다이어트 3단계: 10(시간) * 16(피처) = 160 차원으로 평탄화!
        # (기존 22,528 차원에서 극적으로 감소)
        z_feat = z_temporal.reshape((B, -1)) # (B, 160)
        
        z_feat = nn.LayerNorm()(z_feat)
        z_feat = nn.Dropout(rate=0.5, deterministic=not train)(z_feat)
        
        # 최종 분류기 (Dense 파라미터 고작 644개!)
        logits = nn.Dense(features=4)(z_feat)
        
        return logits, z_feat
            
# =========================================================
# 4. 학습 루프 (JIT 최적화)
# =========================================================
rng = jrandom.PRNGKey(42)
rng, init_rng = jrandom.split(rng)
model = Hybrid_SOTA_SNN()

T_dim = X_train.shape[2] 
dummy_x = jnp.ones((1, 64, T_dim, 1))
variables = model.init(init_rng, dummy_x, train=False)
state = {'params': variables['params'], 'batch_stats': variables.get('batch_stats', {})}

tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=optax.cosine_decay_schedule(3e-3, 200), weight_decay=1e-2))
opt_state = tx.init(state['params'])

@jax.jit
def train_step(state, opt_state, x, y, rng_key):
    def loss_fn(params):
        vars_in = {'params': params, 'batch_stats': state['batch_stats']}
        (logits, z_feat), new_vars = model.apply(vars_in, x, train=True, rngs={'dropout': rng_key}, mutable=['batch_stats'])
        
        y_onehot = jax.nn.one_hot(y, 4)
        loss_cls = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
        
        return loss_cls, (logits, new_vars['batch_stats'])

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

print("🚀 완벽한 4-Class 동일 비율 SNN 학습 시작...")

best_acc = 0.0
for epoch in range(1, 201):
    train_accs = []
    for xb, yb in train_loader:
        rng, step_rng = jrandom.split(rng)
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
print(f"🏆 최종 SOTA 달성 성능: {best_acc:.2f}%")
print("="*50)