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

# =====================================================================
# 1. 아날로그 특징 인코딩 함수 
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

# =====================================================================
# 2. 다수 피험자 데이터 로딩 (Kaggle 최적화)
# =====================================================================
def load_all_graph_data(num_subjects=10):
    base_search_dir = '/kaggle/input' if os.path.exists('/kaggle/input') else '.'
    DATA_DIR = None

    print(f">>> [{base_search_dir}] 뇌파 데이터 탐색 중...")
    for root, dirs, files in os.walk(base_search_dir):
        if 'S001' in dirs:
            DATA_DIR = root
            break

    if DATA_DIR is None:
        raise FileNotFoundError(f"❌ 데이터셋 경로를 찾을 수 없다. Kaggle 우측 Data 탭 확인해.")

    exclude_subjects = ['S088', 'S092', 'S100', 'S104']
    all_subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in exclude_subjects]
    
    # Kaggle CPU 테스트용으로 10명만 먼저 컷
    all_subjects = all_subjects[:num_subjects]
    
    subject_data_dict = {}
    print(f">>> {len(all_subjects)}명 피험자 데이터 전처리 및 로드 중...")
    
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
                
            if not epochs_list:
                continue
                
            epochs_all = mne.concatenate_epochs(epochs_list, verbose=False)
            labels = np.array(epochs_all.events[:, 2]) - 1
            data = epochs_all.get_data() * 1e6  
            sfreq = epochs_all.info['sfreq']
            
            mean_data = np.mean(data, axis=(0, 2), keepdims=True)
            std_data = np.std(data, axis=(0, 2), keepdims=True)
            data_norm = (data - mean_data) / (std_data + 1e-8)
            
            encoded_signals = extract_snn_encoded_features(data_norm, sfreq)
            encoded_signals = encoded_signals[:, :, :480, :] 
            
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
                L_norm = np.eye(A.shape[0]) - (D_inv_sqrt @ A @ D_inv_sqrt)
                X_seq = np.transpose(seq, (1, 0, 2)) 
                L_norm_list.append(L_norm)
                X_feat_list.append(X_seq) 

            subject_data_dict[subj] = {
                'L': torch.tensor(np.nan_to_num(np.array(L_norm_list, dtype=np.float32))),
                'X': torch.tensor(np.nan_to_num(np.array(X_feat_list, dtype=np.float32))),
                'y': torch.tensor(labels, dtype=torch.long)
            }
        except Exception as e:
            continue
            
    return subject_data_dict, list(subject_data_dict.keys())

subject_data_dict, valid_subjects = load_all_graph_data(num_subjects=10)

def get_combined_data(subj_list):
    L_list, X_list, y_list = [], [], []
    for s in subj_list:
        L_list.append(subject_data_dict[s]['L'])
        X_list.append(subject_data_dict[s]['X'])
        y_list.append(subject_data_dict[s]['y'])
    if not L_list: return None, None, None
    return torch.cat(L_list, dim=0), torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

# =====================================================================
# 3. SNN 코어 및 모델 아키텍처 (검증된 다이렉트 인코딩)
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
    beta: float 
    threshold: float 
    
    @nn.compact
    def __call__(self, mem, x):
        mem = mem * self.beta + x
        spk = spike_fn(mem - self.threshold) 
        mem = mem - jax.lax.stop_gradient(spk) * self.threshold 
        return mem, spk

class SNNStep(nn.Module):
    @nn.compact
    def __call__(self, mems, current_in):
        mem_enc, mem_s1, mem_s2, mem_s3, mem_out = mems
        
        gain = self.param('enc_gain', nn.initializers.ones, current_in.shape[-2:])
        bias = self.param('enc_bias', nn.initializers.zeros, current_in.shape[-2:])
        enc_current = current_in * gain + bias
        
        enc_current = nn.Dense(32)(enc_current)
        mem_enc, spk_enc = LIF(beta=0.5, threshold=0.5)(mem_enc, enc_current)
        
        spatial_w = self.param('spatial_w', nn.initializers.glorot_normal(), (64, 16))
        gx_pooled = jnp.einsum('bcf, cs -> bsf', spk_enc, spatial_w) 
        
        c1 = nn.Dense(32)(gx_pooled)
        mem_s1, spk1 = LIF(beta=0.8, threshold=0.6)(mem_s1, c1) 
        c2 = nn.Dense(32)(gx_pooled)
        mem_s2, spk2 = LIF(beta=0.9, threshold=1.0)(mem_s2, c2) 
        c3 = nn.Dense(32)(gx_pooled)
        mem_s3, spk3 = LIF(beta=0.95, threshold=1.4)(mem_s3, c3) 
        
        spk_cat = jnp.concatenate([spk1, spk2, spk3], axis=-1)
        cur_out = nn.Dense(32)(spk_cat)
        mem_out = mem_out * 0.8 + cur_out
        
        new_mems = (mem_enc, mem_s1, mem_s2, mem_s3, mem_out)
        return new_mems, (mem_out, spk_enc, spk_cat)

class Ultimate_STCN_GraphSNN(nn.Module):
    num_classes: int = 4

    @nn.compact
    def __call__(self, L_norm, X_seq, deterministic: bool):
        B, T, C, F = X_seq.shape
        gx_lap = jnp.einsum('bnm, btmf -> btnf', L_norm, X_seq)
        X_seq_filtered = X_seq - 0.5 * gx_lap
        
        ann_out = nn.Conv(features=16, kernel_size=(4, 1), strides=(4, 1), padding='SAME')(X_seq_filtered)
        ann_out = nn.relu(ann_out) 
        
        if not deterministic:
            ann_out = nn.Dropout(rate=0.4, deterministic=deterministic)(ann_out)
        
        mem_enc = jnp.zeros((B, 64, 32))
        mem_s1 = jnp.zeros((B, 16, 32))
        mem_s2 = jnp.zeros((B, 16, 32))
        mem_s3 = jnp.zeros((B, 16, 32))
        mem_out = jnp.zeros((B, 16, 32))
        
        init_mems = (mem_enc, mem_s1, mem_s2, mem_s3, mem_out)
        
        ScanSNN = nn.scan(
            SNNStep, variable_broadcast='params',
            split_rngs={'params': False}, in_axes=1, out_axes=1
        )
        
        _, (mem_out_seq, spk_enc, spk_s) = ScanSNN()(init_mems, ann_out)
        
        pooled_feat = jnp.mean(mem_out_seq, axis=1) 
        flat_feat = pooled_feat.reshape((B, -1)) 
        flat_feat = nn.LayerNorm()(flat_feat) 
        
        y = nn.Dropout(rate=0.5, deterministic=deterministic)(flat_feat)
        logits = nn.Dense(self.num_classes)(y)
        firing_rate = (jnp.mean(spk_enc) + jnp.mean(spk_s)) / 2.0 
        
        return logits, firing_rate

# =====================================================================
# 4. JAX 학습 및 평가 스텝
# =====================================================================
def smooth_labels(labels, num_classes, smoothing=0.2): 
    one_hot = jax.nn.one_hot(labels, num_classes)
    return one_hot * (1.0 - smoothing) + (smoothing / num_classes)

@jax.jit
def train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits, firing_rate = state.apply_fn(
            {'params': params}, L_batch, X_batch, deterministic=False, rngs={'dropout': dropout_key}
        )
        smoothed_targets = smooth_labels(targets, 4, smoothing=0.2)
        loss = optax.softmax_cross_entropy(logits=logits, labels=smoothed_targets).mean()
        return loss, (logits, firing_rate)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, firing_rate)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return state, loss, acc, firing_rate
    
@jax.jit
def eval_step(state, L_batch, X_batch, targets):
    logits, firing_rate = state.apply_fn({'params': state.params}, L_batch, X_batch, deterministic=True)
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return acc

@jax.jit
def sstl_train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits, firing_rate = state.apply_fn(
            {'params': params}, L_batch, X_batch, deterministic=False, rngs={'dropout': dropout_key}
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
# 5. 메인 파이프라인 (5-Fold Global Test + SSTL)
# =====================================================================
kf_global = KFold(n_splits=5, shuffle=True, random_state=42)
global_acc_list = []
sstl_acc_list = []

rng = jax.random.PRNGKey(42)

print("\n" + "="*50)
print("🚀 SNN 5-Fold Cross Validation + SSTL 파이프라인 시작 🚀")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(valid_subjects)):
    print(f"\n[Fold {fold + 1}/5] Start")
    global_train_subjs = [valid_subjects[i] for i in train_idx]
    global_test_subjs = [valid_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = get_combined_data(global_train_subjs)
    if X_train is None: continue
    
    train_loader = DataLoader(TensorDataset(L_train, X_train, y_train), batch_size=128, shuffle=True, drop_last=True)
    
    L_unseen, X_unseen, y_unseen = get_combined_data(global_test_subjs)
    unseen_loader = DataLoader(TensorDataset(L_unseen, X_unseen, y_unseen), batch_size=128, shuffle=False, drop_last=False)
    
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_L = jnp.ones((1, 64, 64))
    dummy_X = jnp.ones((1, 480, 64, 4))
    
    model = Ultimate_STCN_GraphSNN()
    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_L, dummy_X, deterministic=True)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-4, peak_value=2e-3, warmup_steps=100, decay_steps=len(train_loader)*100)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=1e-3))
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # 1. Global Training
    for epoch in range(1, 101): # 100 에포크로 일단 압축
        for batch_L, batch_X, batch_y in train_loader:
            j_L = jnp.array(batch_L.numpy())
            j_X = jnp.array(batch_X.numpy()) 
            j_y = jnp.array(batch_y.numpy())
            
            rng, dropout_key = jax.random.split(rng)
            state, loss, acc, firing_rate = train_step(state, j_L, j_X, j_y, dropout_key)
            
        if epoch % 10 == 0:
            unseen_corr, unseen_tot = 0, 0
            for batch_L, batch_X, batch_y in unseen_loader:
                j_L = jnp.array(batch_L.numpy())
                j_X = jnp.array(batch_X.numpy())
                j_y = jnp.array(batch_y.numpy())
                
                batch_acc = eval_step(state, j_L, j_X, j_y)
                unseen_corr += batch_acc.item() * len(j_y)
                unseen_tot += len(j_y)
                
            print(f"[Epoch {epoch:03d}] Global Test Acc: {100 * unseen_corr / unseen_tot:.2f}% | FR: {firing_rate:.3f}")
    
    unseen_corr, unseen_tot = 0, 0
    for batch_L, batch_X, batch_y in unseen_loader:
        j_L = jnp.array(batch_L.numpy())
        j_X = jnp.array(batch_X.numpy())
        j_y = jnp.array(batch_y.numpy())
        batch_acc = eval_step(state, j_L, j_X, j_y)
        unseen_corr += batch_acc.item() * len(j_y)
        unseen_tot += len(j_y)
            
    true_global_acc = 100 * unseen_corr / unseen_tot
    global_acc_list.append(true_global_acc)
    print(f"Fold {fold + 1} Final Global Test Acc: {true_global_acc:.2f}%")
    
    global_params_backup = state.params
    fold_sstl_accs = []
    
    # 2. SSTL (Subject-Specific Transfer Learning)
    print(f"Fold {fold + 1} SSTL Transfer Learning...")
    for subj in tqdm(global_test_subjs, desc="SSTL", leave=False):
        L_sub = subject_data_dict[subj]['L']
        X_sub = subject_data_dict[subj]['X']
        y_sub = subject_data_dict[subj]['y']
        
        kf_sub = KFold(n_splits=4, shuffle=True, random_state=42)
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in kf_sub.split(X_sub):
            sstl_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=5e-4, weight_decay=1e-3))
            sstl_state = train_state.TrainState.create(apply_fn=model.apply, params=global_params_backup, tx=sstl_tx)
            
            sub_train_loader = DataLoader(TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), batch_size=16, shuffle=True, drop_last=True)
            sub_test_loader = DataLoader(TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), batch_size=16, shuffle=False)
            
            for _ in range(5): 
                for batch_L, batch_X, batch_y in sub_train_loader:
                    j_L = jnp.array(batch_L.numpy())
                    j_X = jnp.array(batch_X.numpy()) 
                    j_y = jnp.array(batch_y.numpy())
                    rng, dropout_key = jax.random.split(rng)
                    sstl_state, _, _ = sstl_train_step(sstl_state, j_L, j_X, j_y, dropout_key)
                    
            corr, tot = 0, 0
            for batch_L, batch_X, batch_y in sub_test_loader:
                j_L = jnp.array(batch_L.numpy())
                j_X = jnp.array(batch_X.numpy()) 
                j_y = jnp.array(batch_y.numpy())
                batch_acc = eval_step(sstl_state, j_L, j_X, j_y)
                corr += batch_acc.item() * len(j_y)
                tot += len(j_y)
            subj_fold_accs.append(100 * corr / tot)
            
        fold_sstl_accs.append(np.mean(subj_fold_accs))
        
    avg_sstl_fold = np.mean(fold_sstl_accs)
    sstl_acc_list.append(avg_sstl_fold)
    print(f"Fold {fold + 1} SSTL Mean Acc: {avg_sstl_fold:.2f}%")

print("\n" + "="*50)
print(f"5-Fold Final Global Test Acc: {np.mean(global_acc_list):.2f}%")
print(f"5-Fold Final SSTL Acc: {np.mean(sstl_acc_list):.2f}%")
print("="*50)