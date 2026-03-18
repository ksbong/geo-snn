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
import functools

# JAX & Flax 생태계 임포트
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import flax.traverse_util

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =====================================================================
# 1. 생물학적 4채널 인코딩 & 전처리 (캐시 유지)
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
    
    encoded_features = np.stack([mu_on, mu_off, beta_on, beta_off], axis=-1)
    return encoded_features

def compute_spd_cov(signal):
    cov = np.cov(signal)
    epsilon = 1e-4 * (np.trace(cov) / cov.shape[0])
    return cov + np.eye(cov.shape[0]) * epsilon

DATA_DIR = './07_Data'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './07_Data' 

SAVE_DIR = './processed_graph_tensors_fixed_v20_Riemannian_Weighted_SNN'
os.makedirs(SAVE_DIR, exist_ok=True)

exclude_subjects = ['S088', 'S092', 'S100', 'S104']
all_subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in exclude_subjects]

def process_and_save_subject_graph(subj):
    save_path_L = f"{SAVE_DIR}/{subj}_L.pt" 
    save_path_X = f"{SAVE_DIR}/{subj}_X.pt" 
    save_path_y = f"{SAVE_DIR}/{subj}_y.pt"
    if os.path.exists(save_path_L): return None
        
    runs_hands = ['R04', 'R08', 'R12']
    runs_feet = ['R06', 'R10', 'R14']
    epochs_list = []
    
    try:
        for run in runs_hands:
            path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
            events_fixed = evs.copy()
            events_fixed[evs[:, 2] == ev_dict.get('T1', -1), 2] = 1 
            events_fixed[evs[:, 2] == ev_dict.get('T2', -1), 2] = 2 
            ep = mne.Epochs(raw, events_fixed, {'Left': 1, 'Right': 2}, tmin=1.0, tmax=4.0, baseline=None, preload=True, on_missing='ignore', verbose=False)
            if len(ep) > 0: epochs_list.append(ep)

        for run in runs_feet:
            path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
            events_fixed = evs.copy()
            events_fixed[evs[:, 2] == ev_dict.get('T1', -1), 2] = 3 
            events_fixed[evs[:, 2] == ev_dict.get('T2', -1), 2] = 4 
            ep = mne.Epochs(raw, events_fixed, {'BothHands': 3, 'BothFeet': 4}, tmin=1.0, tmax=4.0, baseline=None, preload=True, on_missing='ignore', verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
            
        if not epochs_list: return None
        epochs_all = mne.concatenate_epochs(epochs_list, verbose=False)
        labels = np.array(epochs_all.events[:, 2]) - 1
        data = epochs_all.get_data() * 1e6  
        sfreq = epochs_all.info['sfreq']
        
        mean_data = np.mean(data, axis=(0, 2), keepdims=True)
        std_data = np.std(data, axis=(0, 2), keepdims=True)
        data = (data - mean_data) / (std_data + 1e-8)
        
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]
        idx_2 = np.where(labels == 2)[0]
        idx_3 = np.where(labels == 3)[0]
        min_count = min(len(idx_0), len(idx_1), len(idx_2), len(idx_3))
        if min_count == 0: return None 
        
        np.random.seed(42)
        balanced_idx = np.concatenate([
            np.random.choice(idx_0, min_count, replace=False),
            np.random.choice(idx_1, min_count, replace=False),
            np.random.choice(idx_2, min_count, replace=False),
            np.random.choice(idx_3, min_count, replace=False)
        ])
        np.random.shuffle(balanced_idx)
        
        balanced_data = data[balanced_idx]
        balanced_labels = labels[balanced_idx]

        encoded_signals = extract_snn_encoded_features(balanced_data, sfreq)
        encoded_signals = encoded_signals[:, :, :480, :] 
        
        L_norm_list = []
        X_feat_list = []
        
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
            L_norm = np.eye(A.shape[0]) - (D_inv_sqrt @ A @ D_inv_sqrt)
            
            X_seq = np.transpose(seq, (1, 0, 2)) 
            L_norm_list.append(L_norm)
            X_feat_list.append(X_seq) 
        
        L_norm_list = np.nan_to_num(np.array(L_norm_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_feat_list = np.nan_to_num(np.array(X_feat_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        torch.save(torch.tensor(L_norm_list), save_path_L)
        torch.save(torch.tensor(X_feat_list), save_path_X)
        torch.save(torch.tensor(balanced_labels, dtype=torch.long), save_path_y)
        return None 
        
    except Exception as e:
        return f"🚨 {subj} 전처리 에러: {e}"

print(f"🚀 1단계: 캐시된 전처리 데이터 확인 중 (대상 기기: A5000 1 GPU)")
for subj in tqdm(all_subjects, desc="Data Preprocessing", leave=True):
    process_and_save_subject_graph(subj)

def load_graph_data(subject_list):
    L_list, X_list, y_list = [], [], []
    for subj in subject_list:
        if os.path.exists(f"{SAVE_DIR}/{subj}_L.pt"):
            L_list.append(torch.load(f"{SAVE_DIR}/{subj}_L.pt", weights_only=True))
            X_list.append(torch.load(f"{SAVE_DIR}/{subj}_X.pt", weights_only=True))
            y_list.append(torch.load(f"{SAVE_DIR}/{subj}_y.pt", weights_only=True))
    if not L_list: return None, None, None
    return torch.cat(L_list, dim=0), torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


# =====================================================================
# 2. Surrogate Gradient (Fast Sigmoid) & LIF 뉴런
# =====================================================================
@jax.custom_vjp
def spike_fn(x):
    return jnp.where(x > 0, 1.0, 0.0)

def spike_fwd(x):
    return spike_fn(x), x

def spike_bwd(res, g):
    x = res
    alpha = 2.0  # 🔥 절대 기울기가 0으로 죽지 않는 Fast Sigmoid 미분 적용
    grad = 1.0 / jnp.square(1.0 + jnp.abs(alpha * x))
    return (g * grad,)

spike_fn.defvjp(spike_fwd, spike_bwd)

class LIF(nn.Module):
    beta: float = 0.8
    threshold: float = 0.5 
    @nn.compact
    def __call__(self, mem, x):
        mem = mem * self.beta + x
        # SNN 내부 Dense 제거로 인해, 값을 0 이상으로 자연스럽게 유지
        spk = spike_fn(mem - self.threshold) 
        mem = mem - spk * self.threshold 
        return mem, spk


# =====================================================================
# 3. 모델 아키텍처: 구조 단순화 및 공간-시간 철저한 분리 처리
# =====================================================================
class SNN_Core(nn.Module):
    """
    내부에 Dense/LayerNorm이 전혀 없는 순수 시계열 동역학 엔진
    """
    @nn.compact
    def __call__(self, carry, x_t):
        mem1, mem2, mem_out = carry

        mem1, spk1 = LIF(beta=0.8, threshold=0.5)(mem1, x_t)
        mem2, spk2 = LIF(beta=0.9, threshold=0.5)(mem2, spk1)
        
        # Leaky Integrator: 스파이크를 전위로 부드럽게 누적하여 디코더로 패스
        mem_out = mem_out * 0.95 + spk2
        
        return (mem1, mem2, mem_out), (mem_out, spk1, spk2)

class Ultimate_STCN_GraphSNN(nn.Module):
    num_classes: int = 4

    @nn.compact
    def __call__(self, L_norm, X_seq, deterministic: bool):
        B, T, C, F = X_seq.shape
        
        # 1. Temporal Conv (Float 연산)
        ann_out = nn.Conv(features=8, kernel_size=(32, 1), padding='SAME', use_bias=False)(X_seq)
        ann_out = nn.LayerNorm()(ann_out)
        ann_out = nn.relu(ann_out)
        if not deterministic:
            ann_out = nn.Dropout(rate=0.3, deterministic=deterministic)(ann_out)

        # 🔥 2. SNN 루프 밖에서 Riemannian 공간 위상 반영 
        # (시간 축을 포함한 전체 텐서에 한 번에 L_norm 곱셈 수행)
        gx = jnp.einsum('bnm, btmf -> btnf', L_norm, ann_out) # (B, 240, 64, 8)

        # 🔥 3. SNN 루프 밖에서 CSP 공간 압축 진행 
        # 64채널을 4개의 보편적 소스로 강제 압축 (과적합 원천 차단)
        spatial_w = self.param('spatial_w', nn.initializers.glorot_uniform(), (64, 4))
        gx_csp = jnp.einsum('btcf, cs -> btsf', gx, spatial_w) # (B, 240, 4, 8)
        
        # SNN이 처리하기 좋게 채널과 피처를 납작하게 폄 (B, 240, 32)
        snn_in = gx_csp.reshape((B, T, -1)) 

        # 4. 순수 SNN 시계열 역학 
        mem_init = jnp.zeros((B, 32))
        init_carry = (mem_init, mem_init, mem_init)
        
        ScanSNN = nn.scan(
            SNN_Core, variable_broadcast='params',
            split_rngs={'params': False}, in_axes=1, out_axes=1
        )
        
        _, (all_time_potentials, spk1, spk2) = ScanSNN()(init_carry, snn_in)

        # 5. DP-Pooling 디코더 
        L_DP = 24
        N_DP = 12
        num_chunks = T // L_DP
        chunks = all_time_potentials.reshape((B, num_chunks, L_DP, 32))
        
        start_mean = jnp.mean(chunks[:, :, :N_DP, :], axis=2)
        end_mean = jnp.mean(chunks[:, :, -N_DP:, :], axis=2)
        
        # Concat으로 ERD/ERS 보존
        dp_feat = jnp.concatenate([start_mean, end_mean], axis=-1) # (B, 10, 64)
        flat_feat = dp_feat.reshape((B, -1)) # (B, 640)
        
        # 6. Classifier
        y = nn.Dropout(rate=0.5, deterministic=deterministic)(flat_feat)
        y = nn.Dense(32)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=0.5, deterministic=deterministic)(y)
        logits = nn.Dense(self.num_classes)(y)
        
        firing_rate = (jnp.mean(spk1) + jnp.mean(spk2)) / 2.0
        return logits, firing_rate


# =====================================================================
# 4. JAX Single-GPU (jit) 학습 스텝
# =====================================================================
def smooth_labels(labels, num_classes, smoothing=0.2): 
    one_hot = jax.nn.one_hot(labels, num_classes)
    return one_hot * (1.0 - smoothing) + (smoothing / num_classes)

@jax.jit
def train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits, firing_rate = state.apply_fn(
            {'params': params}, L_batch, X_batch, 
            deterministic=False, rngs={'dropout': dropout_key}
        )
        smoothed_targets = smooth_labels(targets, 4, smoothing=0.2)
        ce_loss = optax.softmax_cross_entropy(logits=logits, labels=smoothed_targets).mean()
        
        target_rate = 0.15 
        # FR 페널티 비중을 낮춰서 CE Loss 학습에 집중하게 함
        fr_lambda = 1.0   
        fr_loss = fr_lambda * jnp.square((firing_rate + 1e-8) - target_rate)
        
        loss = ce_loss + fr_loss
        return loss, (logits, firing_rate, ce_loss, fr_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, firing_rate, ce_loss, fr_loss)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    
    return state, loss, acc, firing_rate, fr_loss
    
@jax.jit
def eval_step(state, L_batch, X_batch, targets):
    logits, firing_rate = state.apply_fn(
        {'params': state.params}, L_batch, X_batch, 
        deterministic=True
    )
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return acc

@jax.jit
def sstl_train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits, firing_rate = state.apply_fn(
            {'params': params}, L_batch, X_batch, 
            deterministic=False, rngs={'dropout': dropout_key}
        )
        smoothed_targets = smooth_labels(targets, 4, smoothing=0.2)
        loss = optax.softmax_cross_entropy(logits=logits, labels=smoothed_targets).mean()
        return loss, logits 

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return state, loss, acc

# =====================================================================
# 5. 메인 실행 파이프라인
# =====================================================================
kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list = []
sstl_acc_list = []

rng = jax.random.PRNGKey(42)

print("\n" + "="*55)
print(f"⚡ JAX Single-GPU (A5000 Optimized): Real Riemannian SNN")
print("="*55)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = load_graph_data(global_train_subjs)
    if X_train is None: continue
    
    train_loader = DataLoader(
        TensorDataset(L_train, X_train, y_train), 
        batch_size=128,        
        shuffle=True, 
        drop_last=True,
        num_workers=8,         
        pin_memory=True,       
        persistent_workers=True 
    )
    
    L_unseen, X_unseen, y_unseen = load_graph_data(global_test_subjs)
    unseen_loader = DataLoader(
        TensorDataset(L_unseen, X_unseen, y_unseen), 
        batch_size=128, 
        shuffle=False, 
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_L = jnp.ones((1, 64, 64))
    dummy_X = jnp.ones((1, 240, 64, 4))
    
    model = Ultimate_STCN_GraphSNN()
    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_L, dummy_X, deterministic=True)
    
    total_train_steps = len(train_loader) * 200
    warmup_steps = len(train_loader) * 10 
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, 
        peak_value=2e-3, 
        warmup_steps=warmup_steps, 
        decay_steps=total_train_steps
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4) 
    )
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    for epoch in range(200): 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/200", leave=False)
        for batch_L, batch_X, batch_y in pbar:
            j_L = jnp.array(batch_L.numpy())
            # 다운샘플링 
            j_X = jnp.array(batch_X.numpy()[:, ::2, :, :]) 
            j_y = jnp.array(batch_y.numpy())
            
            rng, dropout_key = jax.random.split(rng)
            state, loss, acc, firing_rate, fr_loss = train_step(state, j_L, j_X, j_y, dropout_key)
                
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{acc.item():.4f}",
                'FR': f"{firing_rate.item():.3f}"
            })

        if (epoch + 1) % 10 == 0:
            unseen_corr, unseen_tot = 0, 0
            for batch_L, batch_X, batch_y in unseen_loader:
                j_L = jnp.array(batch_L.numpy())
                j_X = jnp.array(batch_X.numpy()[:, ::2, :, :])
                j_y = jnp.array(batch_y.numpy())
                
                batch_acc = eval_step(state, j_L, j_X, j_y)
                unseen_corr += batch_acc.item() * j_L.shape[0]
                unseen_tot += j_L.shape[0]
                
            current_test_acc = 100 * unseen_corr / unseen_tot
            print(f"⏳ [Epoch {epoch+1:03d}] 실시간 Global Test Acc (Unseen): {current_test_acc:.2f}%")

    unseen_corr, unseen_tot = 0, 0
    for batch_L, batch_X, batch_y in unseen_loader:
        j_L = jnp.array(batch_L.numpy())
        j_X = jnp.array(batch_X.numpy()[:, ::2, :, :])
        j_y = jnp.array(batch_y.numpy())
        
        batch_acc = eval_step(state, j_L, j_X, j_y)
        unseen_corr += batch_acc.item() * j_L.shape[0]
        unseen_tot += j_L.shape[0]
            
    true_global_acc = 100 * unseen_corr / unseen_tot
    global_acc_list.append(true_global_acc)
    print(f"🔥 Fold {fold + 1} 최종 Global Test Acc (p0): {true_global_acc:.2f}%")
    
    global_params_backup = state.params
    fold_sstl_accs = []
    
    print(f"🎯 Fold {fold + 1} SSTL 진행 중...")
    for subj in tqdm(global_test_subjs, desc="SSTL", leave=False):
        L_sub, X_sub, y_sub = load_graph_data([subj])
        if X_sub is None: continue
        
        kf_sub = KFold(n_splits=4, shuffle=True, random_state=42)
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in kf_sub.split(X_sub):
            sstl_tx = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=5e-4, weight_decay=1e-4) 
            )
            sstl_state = train_state.TrainState.create(apply_fn=model.apply, params=global_params_backup, tx=sstl_tx)
            
            sub_train_loader = DataLoader(
                TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), 
                batch_size=16, shuffle=True, drop_last=True,
                num_workers=8, pin_memory=True, persistent_workers=True
            )
            sub_test_loader = DataLoader(
                TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), 
                batch_size=16, shuffle=False, drop_last=True,
                num_workers=8, pin_memory=True, persistent_workers=True
            )
            
            for _ in range(5): 
                for batch_L, batch_X, batch_y in sub_train_loader:
                    j_L = jnp.array(batch_L.numpy())
                    j_X = jnp.array(batch_X.numpy()[:, ::2, :, :]) 
                    j_y = jnp.array(batch_y.numpy())
                    
                    rng, dropout_key = jax.random.split(rng)
                    sstl_state, loss, acc = sstl_train_step(sstl_state, j_L, j_X, j_y, dropout_key)
                    
            corr, tot = 0, 0
            for batch_L, batch_X, batch_y in sub_test_loader:
                j_L = jnp.array(batch_L.numpy())
                j_X = jnp.array(batch_X.numpy()[:, ::2, :, :]) 
                j_y = jnp.array(batch_y.numpy())
                
                batch_acc = eval_step(sstl_state, j_L, j_X, j_y)
                corr += batch_acc.item() * j_X.shape[0]
                tot += j_X.shape[0]
            subj_fold_accs.append(100 * corr / tot)
            
        fold_sstl_accs.append(np.mean(subj_fold_accs))
        
    avg_sstl_fold = np.mean(fold_sstl_accs)
    sstl_acc_list.append(avg_sstl_fold)
    print(f"🎯 Fold {fold + 1} SSTL 평균 정확도 (ps): {avg_sstl_fold:.2f}%")

print("\n" + "="*50)
print(f"🏆 5-Fold 최종 평균 Global Test Acc (p0): {np.mean(global_acc_list):.2f}%")
print(f"🏆 5-Fold 최종 평균 SSTL Acc (ps): {np.mean(sstl_acc_list):.2f}%")
print("="*50)