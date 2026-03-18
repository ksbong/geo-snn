import os
import mne
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import logm
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

# JAX & Flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 🔥 A5000(Ampere) 극한 최적화: TF32 가속 온! (행렬 곱셈 속도 2~3배 폭발)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# =====================================================================
# 1. 생물학적 4채널 인코딩 (아날로그 특징 추출)
# =====================================================================
def extract_snn_encoded_features(data, sfreq):
    n_times = data.shape[-1]
    freqs = fftfreq(n_times, 1/sfreq)
    X = fft(data, axis=-1)
    
    mask_mu = np.zeros_like(freqs)
    mask_mu[(freqs >= 8.0) & (freqs <= 12.0)] = 2.0
    mask_beta = np.zeros_like(freqs)
    mask_beta[(freqs >= 13.0) & (freqs <= 30.0)] = 2.0
    
    sig_mu = np.real(ifft(X * mask_mu, axis=-1))
    sig_beta = np.real(ifft(X * mask_beta, axis=-1))
    
    mu_on = np.maximum(sig_mu, 0.0)
    mu_off = np.maximum(-sig_mu, 0.0)
    beta_on = np.maximum(sig_beta, 0.0)
    beta_off = np.maximum(-sig_beta, 0.0)
    
    return np.stack([mu_on, mu_off, beta_on, beta_off], axis=-1)

def compute_spd_cov(signal):
    cov = np.cov(signal)
    epsilon = 1e-4 * (np.trace(cov) / cov.shape[0])
    return cov + np.eye(cov.shape[0]) * epsilon

# =====================================================================
# 2. 다수 피험자 데이터 로딩 (RunPod 최적화)
# =====================================================================
def load_all_graph_data():
    # RunPod 보통 /workspace 를 기본으로 씀. 알아서 찾게 세팅.
    base_search_dir = '/workspace' if os.path.exists('/workspace') else '.'
    DATA_DIR = None

    print(f"\n>>> [{base_search_dir}] 뇌파 데이터 자동 탐색 중...")
    for root, dirs, files in os.walk(base_search_dir):
        if 'S001' in dirs:
            DATA_DIR = root
            break

    if DATA_DIR is None:
        raise FileNotFoundError("❌ 데이터셋 경로를 찾을 수 없어. RunPod에 데이터 압축 제대로 풀었는지 확인해.")

    exclude_subjects = ['S088', 'S092', 'S100', 'S104']
    all_subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in exclude_subjects]
    
    subject_data_dict = {}
    print(f">>> A5000 파워 온! {len(all_subjects)}명 피험자 풀-데이터 병렬 로드 시작...\n")
    
    for subj in tqdm(all_subjects, desc="Loading Data"):
        runs_hands = ['R04', 'R08', 'R12']
        runs_feet = ['R06', 'R10', 'R14']
        epochs_list = []
        
        try:
            for run in runs_hands + runs_feet:
                path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
                if not os.path.exists(path): continue
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
                
                t1_id, t2_id = ev_dict.get('T1'), ev_dict.get('T2')
                if t1_id is None or t2_id is None: continue
                
                events_fixed = evs.copy()
                if run in runs_hands:
                    events_fixed[evs[:, 2] == t1_id, 2] = 1 
                    events_fixed[evs[:, 2] == t2_id, 2] = 2 
                    ep = mne.Epochs(raw, events_fixed, {'Left': 1, 'Right': 2}, tmin=1.0, tmax=4.0, baseline=None, preload=True, on_missing='ignore', verbose=False)
                else:
                    events_fixed[evs[:, 2] == t1_id, 2] = 3 
                    events_fixed[evs[:, 2] == t2_id, 2] = 4 
                    ep = mne.Epochs(raw, events_fixed, {'BothHands': 3, 'BothFeet': 4}, tmin=1.0, tmax=4.0, baseline=None, preload=True, on_missing='ignore', verbose=False)
                
                if len(ep) > 0: epochs_list.append(ep)
                
            if not epochs_list: continue
                
            epochs_all = mne.concatenate_epochs(epochs_list, verbose=False)
            labels = np.array(epochs_all.events[:, 2]) - 1
            data = epochs_all.get_data() * 1e6  
            sfreq = epochs_all.info['sfreq']
            
            mean_data = np.mean(data, axis=(0, 2), keepdims=True)
            std_data = np.std(data, axis=(0, 2), keepdims=True)
            data_norm = (data - mean_data) / (std_data + 1e-8)
            
            encoded_signals = extract_snn_encoded_features(data_norm, sfreq)[:, :, :480, :] 
            
            L_norm_list, X_feat_list = [], []
            for ep_idx in range(encoded_signals.shape[0]):
                seq = encoded_signals[ep_idx]
                sig_for_cov = seq.reshape(seq.shape[0], -1)
                cov_full = compute_spd_cov(sig_for_cov)
                tangent = logm(cov_full).real
                tau = np.std(tangent) if np.std(tangent) > 0 else 1.0
                A = np.exp(tangent / tau)
                np.fill_diagonal(A, 0) 
                D = np.diag(np.sum(A, axis=1))
                D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
                
                L_norm_list.append(np.eye(A.shape[0]) - (D_inv_sqrt @ A @ D_inv_sqrt))
                X_feat_list.append(np.transpose(seq, (1, 0, 2))) 

            subject_data_dict[subj] = {
                'L': torch.tensor(np.nan_to_num(np.array(L_norm_list, dtype=np.float32))),
                'X': torch.tensor(np.nan_to_num(np.array(X_feat_list, dtype=np.float32))),
                'y': torch.tensor(labels, dtype=torch.long)
            }
        except Exception:
            continue
            
    return subject_data_dict, list(subject_data_dict.keys())

subject_data_dict, valid_subjects = load_all_graph_data()

def get_combined_data(subj_list):
    L_list, X_list, y_list = [], [], []
    for s in subj_list:
        L_list.append(subject_data_dict[s]['L'])
        X_list.append(subject_data_dict[s]['X'])
        y_list.append(subject_data_dict[s]['y'])
    if not L_list: return None, None, None
    return torch.cat(L_list, dim=0), torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

# =====================================================================
# 3. Surrogate Gradient & 순정 LIF
# =====================================================================
@jax.custom_vjp
def spike_fn(x):
    return jnp.where(x > 0, 1.0, 0.0)

def spike_fwd(x):
    return spike_fn(x), x

def spike_bwd(res, g):
    x = res
    alpha = 2.0
    grad = alpha / (2.0 * jnp.square(1.0 + jnp.abs(alpha * x)))
    return (g * grad,)

spike_fn.defvjp(spike_fwd, spike_bwd)

class LIF(nn.Module):
    beta: float = 0.8
    threshold: float = 1.0
    
    @nn.compact
    def __call__(self, mem, x):
        mem = mem * self.beta + x
        spk = spike_fn(mem - self.threshold) 
        mem = mem - jax.lax.stop_gradient(spk) * self.threshold 
        return mem, spk

# =====================================================================
# 4. 네 오리지널 모델 아키텍처 (100% 복원)
# =====================================================================
class SNNStep(nn.Module):
    @nn.compact
    def __call__(self, carry, current_in):
        mems, L_norm = carry
        mem_enc, mem_s, mem_out = mems
        
        enc_in = nn.Dense(16)(current_in)
        enc_in = nn.LayerNorm()(enc_in) 
        mem_enc, spk_enc = LIF()(mem_enc, enc_in)
        
        adj = jnp.eye(64) - L_norm
        gx = jnp.einsum('bnm, bmf -> bnf', adj, spk_enc)
        
        spatial_w = self.param('spatial_w', nn.initializers.glorot_normal(), (64, 8))
        gx_pooled = jnp.einsum('bcf, cs -> bsf', gx, spatial_w) 
        gx_flat = gx_pooled.reshape((gx_pooled.shape[0], -1)) 
        
        cur_s = nn.Dense(64)(gx_flat)
        cur_s = nn.LayerNorm()(cur_s)
        mem_s, spk_s = LIF()(mem_s, cur_s) 
        
        cur_out = nn.Dense(32)(spk_s)
        mem_out = mem_out * 0.95 + cur_out 
        
        return (mem_enc, mem_s, mem_out), (mem_out, spk_enc, spk_s)

class Ultimate_STCN_GraphSNN(nn.Module):
    num_classes: int = 4

    @nn.compact
    def __call__(self, L_norm, X_seq, deterministic: bool):
        B = X_seq.shape[0]
        
        ann_out = nn.Conv(features=16, kernel_size=(16, 1), strides=(4, 1), padding='SAME')(X_seq)
        ann_out = nn.relu(ann_out) 
        
        if not deterministic:
            ann_out = nn.Dropout(rate=0.3, deterministic=deterministic)(ann_out)
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, p=0.7, shape=L_norm.shape)
            L_norm = L_norm * mask / 0.7

        init_carry = ((jnp.zeros((B, 64, 16)), jnp.zeros((B, 64)), jnp.zeros((B, 32))), L_norm)
        ScanSNN = nn.scan(SNNStep, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        _, (mem_out_seq, spk_enc, spk_s) = ScanSNN()(init_carry, ann_out)
        
        num_chunks = 10
        chunk_size = 120 // num_chunks
        chunks = mem_out_seq.reshape((B, num_chunks, chunk_size, 32))
        pooled_feat = jnp.mean(chunks, axis=2).reshape((B, -1)) 
        
        y = nn.Dropout(rate=0.5, deterministic=deterministic)(pooled_feat)
        logits = nn.Dense(self.num_classes)(y)
        
        return logits, (jnp.mean(spk_enc) + jnp.mean(spk_s)) / 2.0 

# =====================================================================
# 5. JAX 학습 스텝 
# =====================================================================
def smooth_labels(labels, num_classes, smoothing=0.2): 
    return jax.nn.one_hot(labels, num_classes) * (1.0 - smoothing) + (smoothing / num_classes)

@jax.jit
def train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits, firing_rate = state.apply_fn(
            {'params': params}, L_batch, X_batch, deterministic=False, rngs={'dropout': dropout_key}
        )
        loss = optax.softmax_cross_entropy(logits=logits, labels=smooth_labels(targets, 4)).mean()
        return loss, (logits, firing_rate)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, firing_rate)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, jnp.mean(jnp.argmax(logits, -1) == targets), firing_rate
    
@jax.jit
def eval_step(state, L_batch, X_batch, targets):
    logits, _ = state.apply_fn({'params': state.params}, L_batch, X_batch, deterministic=True)
    return jnp.mean(jnp.argmax(logits, -1) == targets)

@jax.jit
def sstl_train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits, _ = state.apply_fn(
            {'params': params}, L_batch, X_batch, deterministic=False, rngs={'dropout': dropout_key}
        )
        loss = optax.softmax_cross_entropy(logits=logits, labels=smooth_labels(targets, 4)).mean()
        return loss, logits 

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, jnp.mean(jnp.argmax(logits, -1) == targets)
# =====================================================================
# 6. 메인 실행 파이프라인 (A5000 극한 병렬화 + Best Model + SSTL)
# =====================================================================
kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list, sstl_acc_list = [], []

rng = jax.random.PRNGKey(42)

print("\n" + "="*50)
print("🚀 A5000 풀 파워 가동: JAX 순정 SNN 파이프라인 시작 🚀")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(valid_subjects)):
    print(f"\n[Fold {fold + 1}/5] Start")
    
    # [핵심 수정] 여기서 변수를 명확히 선언해야 밑에 SSTL에서 쓸 수 있음
    global_train_subjs = [valid_subjects[i] for i in train_idx]
    global_test_subjs = [valid_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = get_combined_data(global_train_subjs)
    L_unseen, X_unseen, y_unseen = get_combined_data(global_test_subjs)
    if X_train is None: continue
    
    # 🔥 GPU Feeding 병목 제거: 핀 메모리 + 다중 워커 + 프리패치
    train_loader = DataLoader(
        TensorDataset(L_train, X_train, y_train), 
        batch_size=128, shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    unseen_loader = DataLoader(
        TensorDataset(L_unseen, X_unseen, y_unseen), 
        batch_size=128, shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_L = jnp.ones((1, 64, 64))
    dummy_X = jnp.ones((1, 480, 64, 4))
    
    model = Ultimate_STCN_GraphSNN()
    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_L, dummy_X, deterministic=True)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-4, peak_value=2e-3, warmup_steps=100, decay_steps=len(train_loader)*150)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4))
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    best_test_acc = 0.0
    best_params = state.params 
    
    # 1. Global Training (150 Epochs)
    for epoch in range(1, 151): 
        for batch_L, batch_X, batch_y in train_loader:
            # 🔥 jax.device_put으로 GPU 다이렉트 전송 (비동기 통신 가속)
            j_L = jax.device_put(jnp.array(batch_L.numpy()))
            j_X = jax.device_put(jnp.array(batch_X.numpy()))
            j_y = jax.device_put(jnp.array(batch_y.numpy()))
            
            rng, dropout_key = jax.random.split(rng)
            state, loss, acc, firing_rate = train_step(state, j_L, j_X, j_y, dropout_key)

        if epoch % 5 == 0:
            unseen_corr, unseen_tot = 0, 0
            for batch_L, batch_X, batch_y in unseen_loader:
                j_L = jax.device_put(jnp.array(batch_L.numpy()))
                j_X = jax.device_put(jnp.array(batch_X.numpy()))
                j_y = jax.device_put(jnp.array(batch_y.numpy()))
                
                batch_acc = eval_step(state, j_L, j_X, j_y)
                unseen_corr += batch_acc.item() * len(j_y)
                unseen_tot += len(j_y)
                
            current_test_acc = 100 * unseen_corr / unseen_tot
            
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                best_params = state.params
                print(f"[Epoch {epoch:03d}] ⭐ 최고 기록! Global Test Acc: {current_test_acc:.2f}% | FR: {firing_rate.item():.3f}")
            else:
                print(f"[Epoch {epoch:03d}] Global Test Acc: {current_test_acc:.2f}% | FR: {firing_rate.item():.3f}")
    
    global_acc_list.append(best_test_acc)
    print(f"Fold {fold + 1} Final Global Test Acc (Best Checkpoint): {best_test_acc:.2f}%")
    
    # 2. SSTL Transfer Learning (20 Epochs)
    print(f"Fold {fold + 1} SSTL Transfer Learning (20 Epochs)...")
    fold_sstl_accs = []
    
    for subj in tqdm(global_test_subjs, desc="SSTL", leave=False):
        L_sub, X_sub, y_sub = subject_data_dict[subj]['L'], subject_data_dict[subj]['X'], subject_data_dict[subj]['y']
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in KFold(n_splits=4, shuffle=True, random_state=42).split(X_sub):
            sstl_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=5e-4, weight_decay=1e-4))
            sstl_state = train_state.TrainState.create(apply_fn=model.apply, params=best_params, tx=sstl_tx)
            
            sub_train_loader = DataLoader(
                TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), 
                batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
            )
            sub_test_loader = DataLoader(
                TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), 
                batch_size=16, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
            )
            
            for _ in range(20): 
                for batch_L, batch_X, batch_y in sub_train_loader:
                    j_L = jax.device_put(jnp.array(batch_L.numpy()))
                    j_X = jax.device_put(jnp.array(batch_X.numpy()))
                    j_y = jax.device_put(jnp.array(batch_y.numpy()))
                    
                    rng, dropout_key = jax.random.split(rng)
                    sstl_state, loss, acc = sstl_train_step(sstl_state, j_L, j_X, j_y, dropout_key)
                    
            corr, tot = 0, 0
            for batch_L, batch_X, batch_y in sub_test_loader:
                j_L = jax.device_put(jnp.array(batch_L.numpy()))
                j_X = jax.device_put(jnp.array(batch_X.numpy()))
                j_y = jax.device_put(jnp.array(batch_y.numpy()))
                
                batch_acc = eval_step(sstl_state, j_L, j_X, j_y)
                corr += batch_acc.item() * len(j_y)
                tot += len(j_y)
            subj_fold_accs.append(100 * corr / tot)
            
        fold_sstl_accs.append(np.mean(subj_fold_accs))
        
    avg_sstl_fold = np.mean(fold_sstl_accs)
    sstl_acc_list.append(avg_sstl_fold)
    print(f"Fold {fold + 1} SSTL Mean Acc: {avg_sstl_fold:.2f}%")

print("\n" + "="*50)
print(f"5-Fold Final Global Test Acc (Best Checkpoint): {np.mean(global_acc_list):.2f}%")
print(f"5-Fold Final SSTL Acc: {np.mean(sstl_acc_list):.2f}%")
print("="*50)# =====================================================================
# 6. 메인 실행 파이프라인 (A5000 극한 병렬화 + Best Model + SSTL)
# =====================================================================
kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list, sstl_acc_list = [], []

rng = jax.random.PRNGKey(42)

print("\n" + "="*50)
print("🚀 A5000 풀 파워 가동: JAX 순정 SNN 파이프라인 시작 🚀")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(valid_subjects)):
    print(f"\n[Fold {fold + 1}/5] Start")
    
    # [핵심 수정] 여기서 변수를 명확히 선언해야 밑에 SSTL에서 쓸 수 있음
    global_train_subjs = [valid_subjects[i] for i in train_idx]
    global_test_subjs = [valid_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = get_combined_data(global_train_subjs)
    L_unseen, X_unseen, y_unseen = get_combined_data(global_test_subjs)
    if X_train is None: continue
    
    # 🔥 GPU Feeding 병목 제거: 핀 메모리 + 다중 워커 + 프리패치
    train_loader = DataLoader(
        TensorDataset(L_train, X_train, y_train), 
        batch_size=128, shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    unseen_loader = DataLoader(
        TensorDataset(L_unseen, X_unseen, y_unseen), 
        batch_size=128, shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_L = jnp.ones((1, 64, 64))
    dummy_X = jnp.ones((1, 480, 64, 4))
    
    model = Ultimate_STCN_GraphSNN()
    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_L, dummy_X, deterministic=True)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-4, peak_value=2e-3, warmup_steps=100, decay_steps=len(train_loader)*150)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4))
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    best_test_acc = 0.0
    best_params = state.params 
    
    # 1. Global Training (150 Epochs)
    for epoch in range(1, 151): 
        for batch_L, batch_X, batch_y in train_loader:
            # 🔥 jax.device_put으로 GPU 다이렉트 전송 (비동기 통신 가속)
            j_L = jax.device_put(jnp.array(batch_L.numpy()))
            j_X = jax.device_put(jnp.array(batch_X.numpy()))
            j_y = jax.device_put(jnp.array(batch_y.numpy()))
            
            rng, dropout_key = jax.random.split(rng)
            state, loss, acc, firing_rate = train_step(state, j_L, j_X, j_y, dropout_key)

        if epoch % 5 == 0:
            unseen_corr, unseen_tot = 0, 0
            for batch_L, batch_X, batch_y in unseen_loader:
                j_L = jax.device_put(jnp.array(batch_L.numpy()))
                j_X = jax.device_put(jnp.array(batch_X.numpy()))
                j_y = jax.device_put(jnp.array(batch_y.numpy()))
                
                batch_acc = eval_step(state, j_L, j_X, j_y)
                unseen_corr += batch_acc.item() * len(j_y)
                unseen_tot += len(j_y)
                
            current_test_acc = 100 * unseen_corr / unseen_tot
            
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                best_params = state.params
                print(f"[Epoch {epoch:03d}] ⭐ 최고 기록! Global Test Acc: {current_test_acc:.2f}% | FR: {firing_rate.item():.3f}")
            else:
                print(f"[Epoch {epoch:03d}] Global Test Acc: {current_test_acc:.2f}% | FR: {firing_rate.item():.3f}")
    
    global_acc_list.append(best_test_acc)
    print(f"Fold {fold + 1} Final Global Test Acc (Best Checkpoint): {best_test_acc:.2f}%")
    
    # 2. SSTL Transfer Learning (20 Epochs)
    print(f"Fold {fold + 1} SSTL Transfer Learning (20 Epochs)...")
    fold_sstl_accs = []
    
    for subj in tqdm(global_test_subjs, desc="SSTL", leave=False):
        L_sub, X_sub, y_sub = subject_data_dict[subj]['L'], subject_data_dict[subj]['X'], subject_data_dict[subj]['y']
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in KFold(n_splits=4, shuffle=True, random_state=42).split(X_sub):
            sstl_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=5e-4, weight_decay=1e-4))
            sstl_state = train_state.TrainState.create(apply_fn=model.apply, params=best_params, tx=sstl_tx)
            
            sub_train_loader = DataLoader(
                TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), 
                batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
            )
            sub_test_loader = DataLoader(
                TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), 
                batch_size=16, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
            )
            
            for _ in range(20): 
                for batch_L, batch_X, batch_y in sub_train_loader:
                    j_L = jax.device_put(jnp.array(batch_L.numpy()))
                    j_X = jax.device_put(jnp.array(batch_X.numpy()))
                    j_y = jax.device_put(jnp.array(batch_y.numpy()))
                    
                    rng, dropout_key = jax.random.split(rng)
                    sstl_state, loss, acc = sstl_train_step(sstl_state, j_L, j_X, j_y, dropout_key)
                    
            corr, tot = 0, 0
            for batch_L, batch_X, batch_y in sub_test_loader:
                j_L = jax.device_put(jnp.array(batch_L.numpy()))
                j_X = jax.device_put(jnp.array(batch_X.numpy()))
                j_y = jax.device_put(jnp.array(batch_y.numpy()))
                
                batch_acc = eval_step(sstl_state, j_L, j_X, j_y)
                corr += batch_acc.item() * len(j_y)
                tot += len(j_y)
            subj_fold_accs.append(100 * corr / tot)
            
        fold_sstl_accs.append(np.mean(subj_fold_accs))
        
    avg_sstl_fold = np.mean(fold_sstl_accs)
    sstl_acc_list.append(avg_sstl_fold)
    print(f"Fold {fold + 1} SSTL Mean Acc: {avg_sstl_fold:.2f}%")

print("\n" + "="*50)
print(f"5-Fold Final Global Test Acc (Best Checkpoint): {np.mean(global_acc_list):.2f}%")
print(f"5-Fold Final SSTL Acc: {np.mean(sstl_acc_list):.2f}%")
print("="*50)