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

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =====================================================================
# 1. 생물학적 4채널 인코딩 & 전처리 (V20 캐시 재사용)
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

# RunPod 환경에 맞게 경로 확인 필수
DATA_DIR = '/workspace/full-physionet/07_Data'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './07_Data' # 현재 디렉토리에 데이터가 있을 경우

# 캐시 저장 경로
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
# 2. Surrogate Gradient (🚀 가속화 세팅) & LIF 뉴런
# =====================================================================
@jax.custom_vjp
def multi_gaussian_spike(x):
    return jnp.where(x > 0, 1.0, 0.0)

def mg_fwd(x):
    return multi_gaussian_spike(x), x

def mg_bwd(res, g):
    x = res
    h, s, sigma = 0.2, 6.0, 0.6 
    def gaussian(val, mu, std):
        return (1.0 / (jnp.sqrt(2 * jnp.pi) * std)) * jnp.exp(-0.5 * jnp.square((val - mu) / std))
    grad = (1 + h) * gaussian(x, 0.0, sigma) - h * gaussian(x, sigma, s * sigma) - h * gaussian(x, -sigma, s * sigma)
    return (g * grad,)

multi_gaussian_spike.defvjp(mg_fwd, mg_bwd)

class LIF(nn.Module):
    beta: float = 0.9
    threshold: float = 0.5 
    @nn.compact
    def __call__(self, mem, x):
        mem = mem * self.beta + x
        mem = jnp.maximum(mem, -2.0) 
        spk = multi_gaussian_spike(mem - self.threshold) 
        mem = mem - spk * self.threshold 
        return mem, spk


# =====================================================================
# 3. 모델 아키텍처 V22: SNN-Main 구조 + Trainable Skip Connection
# =====================================================================
class MultiBetaLIF(nn.Module):
    @nn.compact
    def __call__(self, mems, x):
        mem1, spk1 = LIF(beta=0.8, threshold=0.5)(mems[0], x)
        mem2, spk2 = LIF(beta=0.9, threshold=0.3)(mems[1], x)
        mem3, spk3 = LIF(beta=0.95, threshold=0.2)(mems[2], x)
        spk_out = jnp.concatenate([spk1, spk2, spk3], axis=-1) 
        return [mem1, mem2, mem3], spk_out

class DPPoolingDecoder(nn.Module):
    num_classes: int = 4
    L_DP: int = 24
    N_DP: int = 12

    @nn.compact
    def __call__(self, x_seq, deterministic: bool):
        B, T, F = x_seq.shape 
        num_chunks = T // self.L_DP
        chunks = x_seq.reshape((B, num_chunks, self.L_DP, F))
        
        start_mean = jnp.mean(chunks[:, :, :self.N_DP, :], axis=2)
        end_mean = jnp.mean(chunks[:, :, -self.N_DP:, :], axis=2)
        dp_feat = end_mean - start_mean 
        
        x = dp_feat.reshape((B, -1)) 
        
        x = nn.Dense(32)(x)
        x = nn.LayerNorm(epsilon=1e-5)(x) 
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.5, deterministic=deterministic)(x)
        x = nn.Dense(self.num_classes)(x)
        return x

class SNNStep(nn.Module):
    @nn.compact
    def __call__(self, carry, current_in):
        mems, L_norm = carry
        mem_enc, mem_s, mem_hr1, mem_hr2, mem_hr3, mem_out = mems
        
        mem_enc, spk_enc = LIF(beta=0.8, threshold=0.5)(mem_enc, current_in)
        
        gx = jnp.einsum('bnm, bmf -> bnf', L_norm, spk_enc)
        gx_flat = gx.reshape((gx.shape[0], -1))
        
        cur_s = nn.Dense(16, use_bias=False)(gx_flat)
        mem_s, spk_s = LIF(beta=0.85, threshold=0.3)(mem_s, cur_s) 
        
        mems_hr_input = [mem_hr1, mem_hr2, mem_hr3]
        mems_hr_output, spk_hr = MultiBetaLIF()(mems_hr_input, spk_s) 
        mem_hr1, mem_hr2, mem_hr3 = mems_hr_output
        
        # Trainable Skip Connection
        skip_gx = nn.Dense(8, use_bias=False)(gx_flat)
        skip_s  = nn.Dense(8, use_bias=False)(spk_s)
        
        cur_out = nn.Dense(8, use_bias=False)(spk_hr) 
        mem_out, spk_out = LIF(beta=0.9, threshold=0.3)(mem_out, cur_out + skip_gx + skip_s)
        
        new_mems = (mem_enc, mem_s, mem_hr1, mem_hr2, mem_hr3, mem_out)
        return (new_mems, L_norm), (mem_out, spk_enc, spk_s, spk_hr, spk_out)
        
class Ultimate_STCN_GraphSNN(nn.Module):
    num_steps: int = 240 

    @nn.compact
    def __call__(self, L_norm, X_seq, deterministic: bool):
        B = X_seq.shape[0]
        
        ann_out = nn.Conv(features=8, kernel_size=(15, 1), padding='SAME', use_bias=True)(X_seq)
        ann_out = nn.LayerNorm(epsilon=1e-5)(ann_out)
        ann_out = nn.relu(ann_out) 
        
        if not deterministic:
            ann_out = nn.Dropout(rate=0.3, deterministic=deterministic)(ann_out)

        mem_enc = jnp.zeros((B, 64, 8))
        mem_s = jnp.zeros((B, 16))
        mem_hr1 = jnp.zeros((B, 16))
        mem_hr2 = jnp.zeros((B, 16))
        mem_hr3 = jnp.zeros((B, 16))
        mem_out = jnp.zeros((B, 8))
        
        mems = (mem_enc, mem_s, mem_hr1, mem_hr2, mem_hr3, mem_out)
        init_carry = (mems, L_norm)
        
        ScanSNN = nn.scan(
            SNNStep, variable_broadcast='params',
            split_rngs={'params': False}, in_axes=1, out_axes=1
        )
        
        _, (all_time_potentials, spk_enc, spk_s, spk_hr, spk_out) = ScanSNN()(init_carry, ann_out)
        
        out = DPPoolingDecoder()(all_time_potentials, deterministic=deterministic)
        
        fr_enc = jnp.mean(spk_enc)
        fr_s = jnp.mean(spk_s)
        fr_hr = jnp.mean(spk_hr)
        fr_out = jnp.mean(spk_out)
        mean_firing_rate = (fr_enc + fr_s + fr_hr + fr_out) / 4.0
        
        return out, mean_firing_rate


# =====================================================================
# 4. JAX Single-GPU (jit) 학습 스텝 최적화 🔥
# =====================================================================
def smooth_labels(labels, num_classes, smoothing=0.2): 
    one_hot = jax.nn.one_hot(labels, num_classes)
    return one_hot * (1.0 - smoothing) + (smoothing / num_classes)

# 💡 pmap 제거, 순수 jit 컴파일로 오버헤드 0% 달성
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
        fr_lambda = 3.0   
        fr_loss = fr_lambda * jnp.square((firing_rate + 1e-8) - target_rate)
        
        loss = ce_loss + fr_loss
        return loss, (logits, firing_rate, ce_loss, fr_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, firing_rate, ce_loss, fr_loss)), grads = grad_fn(state.params)
    
    # 단일 GPU이므로 pmean(그래디언트 평균) 과정 불필요
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
    
    flat_grads = flax.traverse_util.flatten_dict(grads)
    flat_grads = {k: (v if 'DPPoolingDecoder' in k[0] else jnp.zeros_like(v)) for k, v in flat_grads.items()}
    grads = flax.traverse_util.unflatten_dict(flat_grads)
    
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
print(f"⚡ JAX Single-GPU (A5000 Optimized): Fast Convergence SNN")
print("="*55)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = load_graph_data(global_train_subjs)
    if X_train is None: continue
    
    train_loader = DataLoader(
        TensorDataset(L_train, X_train, y_train), 
        batch_size=32, shuffle=True, drop_last=True 
    )
    
    L_unseen, X_unseen, y_unseen = load_graph_data(global_test_subjs)
    unseen_loader = DataLoader(TensorDataset(L_unseen, X_unseen, y_unseen), batch_size=32, shuffle=False, drop_last=True)
    
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_L = jnp.ones((1, 64, 64))
    dummy_X = jnp.ones((1, 240, 64, 4))
    
    model = Ultimate_STCN_GraphSNN(num_steps=240)
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
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-5) 
    )
    
    # 단일 GPU이므로 state.replicate 불필요
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    for epoch in range(200): 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/200", leave=False)
        for batch_L, batch_X, batch_y in pbar:
            j_L = jnp.array(batch_L.numpy())
            j_X = jnp.array(batch_X.numpy()) 
            j_y = jnp.array(batch_y.numpy())
            
            # 단일 GPU 최적화: RNG split 간소화
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
                j_X = jnp.array(batch_X.numpy())
                j_y = jnp.array(batch_y.numpy())
                
                batch_acc = eval_step(state, j_L, j_X, j_y)
                
                unseen_corr += batch_acc.item() * j_L.shape[0]
                unseen_tot += j_L.shape[0]
                
            current_test_acc = 100 * unseen_corr / unseen_tot
            print(f"⏳ [Epoch {epoch+1:03d}] 실시간 Global Test Acc (Unseen): {current_test_acc:.2f}%")

    unseen_corr, unseen_tot = 0, 0
    for batch_L, batch_X, batch_y in unseen_loader:
        j_L = jnp.array(batch_L.numpy())
        j_X = jnp.array(batch_X.numpy())
        j_y = jnp.array(batch_y.numpy())
        
        batch_acc = eval_step(state, j_L, j_X, j_y)
        
        unseen_corr += batch_acc.item() * j_L.shape[0]
        unseen_tot += j_L.shape[0]
            
    true_global_acc = 100 * unseen_corr / unseen_tot
    global_acc_list.append(true_global_acc)
    print(f"🔥 Fold {fold + 1} 최종 Global Test Acc (p0): {true_global_acc:.2f}%")
    
    # 단일 GPU이므로 state.unreplicate 불필요
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
            
            sub_train_loader = DataLoader(TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), batch_size=16, shuffle=True, drop_last=True)
            sub_test_loader = DataLoader(TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), batch_size=16, shuffle=False, drop_last=True)
            
            for _ in range(5): 
                for batch_L, batch_X, batch_y in sub_train_loader:
                    j_L = jnp.array(batch_L.numpy())
                    j_X = jnp.array(batch_X.numpy()) 
                    j_y = jnp.array(batch_y.numpy())
                    
                    rng, dropout_key = jax.random.split(rng)
                    sstl_state, loss, acc = sstl_train_step(sstl_state, j_L, j_X, j_y, dropout_key)
                    
            corr, tot = 0, 0
            for batch_L, batch_X, batch_y in sub_test_loader:
                j_L = jnp.array(batch_L.numpy())
                j_X = jnp.array(batch_X.numpy()) 
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