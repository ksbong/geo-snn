import os
import mne
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import sqrtm, logm, inv
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

# JAX & Flax 생태계 임포트
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import flax.traverse_util

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =====================================================================
# 1. 기하학적 특징 전처리 (NumPy & PyTorch 기반 유지, 가장 안정적임)
# =====================================================================
def project_to_hardy_space(data, sfreq, l_freq=8.0, h_freq=30.0):
    n_times = data.shape[-1]
    freqs = fftfreq(n_times, 1/sfreq)
    mask = np.zeros_like(freqs)
    mask[(freqs >= l_freq) & (freqs <= h_freq)] = 2.0  
    X = fft(data, axis=-1)
    return ifft(X * mask, axis=-1)

def compute_spd_cov(envelope):
    cov = np.cov(envelope)
    epsilon = 1e-4 * (np.trace(cov) / cov.shape[0])
    return cov + np.eye(cov.shape[0]) * epsilon

DATA_DIR = './raw_data/files/'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './raw_data/files'

SAVE_DIR = './processed_graph_tensors'
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
            ep = mne.Epochs(raw, evs, {'Rest': ev_dict['T0'], 'Left Hand': ev_dict['T1'], 'Right Hand': ev_dict['T2']}, 
                            tmin=1.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            epochs_list.append(ep)

        for run in runs_feet:
            path = os.path.join(DATA_DIR, subj, f'{subj}{run}.edf')
            if not os.path.exists(path): continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
            events_fixed = evs.copy()
            events_fixed[events_fixed[:, 2] == ev_dict['T2'], 2] = 4
            ep = mne.Epochs(raw, events_fixed, {'Rest': ev_dict['T0'], 'Both Feet': 4}, 
                            tmin=1.0, tmax=4.0, baseline=None, preload=True, verbose=False)
            epochs_list.append(ep)
            
        if not epochs_list: return None
        epochs_all = mne.concatenate_epochs(epochs_list, verbose=False)
        
        data = epochs_all.get_data() * 1e6  
        labels = np.array(epochs_all.events[:, 2]) - 1 
        sfreq = epochs_all.info['sfreq']

        hardy_signals = project_to_hardy_space(data, sfreq)
        envelopes = np.abs(hardy_signals) 
        envelopes = envelopes[:, :, :480] # 480스텝 컷
        
        rest_idx = (labels == 0)
        rest_envelopes = envelopes[rest_idx]
        rest_covs = np.array([compute_spd_cov(env) for env in rest_envelopes])
        p_rest = np.mean(rest_covs, axis=0) 
        p_rest_sqrt = sqrtm(p_rest).real
        p_rest_inv_sqrt = inv(p_rest_sqrt)

        L_norm_list = []
        X_feat_list = []
        
        for ep_idx in range(envelopes.shape[0]):
            env = envelopes[ep_idx] 
            
            cov_full = compute_spd_cov(env)
            inner = p_rest_inv_sqrt @ cov_full @ p_rest_inv_sqrt
            tangent = p_rest_sqrt @ logm(inner).real @ p_rest_sqrt
            
            A = np.abs(tangent)
            np.fill_diagonal(A, 0) 
            
            k = 8 
            A_sparse = np.zeros_like(A)
            for i in range(A.shape[0]):
                idx = np.argsort(A[i])[-k:]
                A_sparse[i, idx] = A[i, idx]
            
            A_sparse = np.maximum(A_sparse, A_sparse.T) 
            
            D = np.diag(np.sum(A_sparse, axis=1))
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
            L_norm = np.eye(A.shape[0]) - (D_inv_sqrt @ A_sparse @ D_inv_sqrt)
            
            X_seq = env.T.reshape(480, 64, 1) 
            L_norm_list.append(L_norm)
            X_feat_list.append(X_seq) 
        
        L_norm_list = np.array(L_norm_list, dtype=np.float32) 
        X_feat_list = np.array(X_feat_list, dtype=np.float32) 

        scale_X = np.max(np.abs(X_feat_list))
        if scale_X > 0: X_feat_list = X_feat_list / scale_X

        torch.save(torch.tensor(L_norm_list), save_path_L)
        torch.save(torch.tensor(X_feat_list), save_path_X)
        torch.save(torch.tensor(labels, dtype=torch.long), save_path_y)
        return None 
        
    except Exception as e:
        return f"🚨 {subj} 전처리 에러: {e}"

print("🚀 1단계: O(N^3) 최적화 전처리 확인")
for subj in tqdm(all_subjects, desc="Processing", leave=True):
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
# 2. JAX/Flax 용 커스텀 Surrogate Gradient & LIF 뉴런
# =====================================================================
@jax.custom_vjp
def fast_sigmoid(x):
    # Forward pass: 발화 임계점 넘으면 1, 아니면 0 (Heaviside step)
    return jnp.where(x > 0, 1.0, 0.0)

def fs_fwd(x):
    return fast_sigmoid(x), x

def fs_bwd(res, g):
    # Backward pass: snntorch.surrogate.fast_sigmoid 와 동일한 수식 적용
    x = res
    slope = 25.0
    grad = slope / (slope * jnp.abs(x) + 1.0)**2
    return (g * grad,)

fast_sigmoid.defvjp(fs_fwd, fs_bwd)

class LIF(nn.Module):
    beta: float = 0.9

    @nn.compact
    def __call__(self, mem, x):
        mem = mem * self.beta + x
        spk = fast_sigmoid(mem - 1.0) # threshold = 1.0
        mem = mem - spk # reset by subtraction
        return mem, spk

# =====================================================================
# 3. JAX 기반 SNN 모듈 및 메인 모델
# =====================================================================
class DynamicLaplacianConv(nn.Module):
    out_features: int

    @nn.compact
    def __call__(self, L_dynamic, X):
        # X: (B, N, F)
        support = nn.Dense(self.out_features, use_bias=True)(X)
        # JAX의 행렬곱: (B, N, N) @ (B, N, F) -> (B, N, F)
        return jnp.matmul(L_dynamic, support)

class CausalSpikingTCN(nn.Module):
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        # Causal padding 적용
        padding = self.kernel_size - 1
        x_pad = jnp.pad(x, ((0, 0), (padding, 0), (0, 0)))
        out = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), padding='VALID')(x_pad)
        # JAX에서 BatchNorm 상태 관리가 번거로우므로 LayerNorm으로 깔끔하게 대체
        out = nn.LayerNorm()(out) 
        return out

class TemporalAttentionDecoder(nn.Module):
    num_classes: int = 4

    @nn.compact
    def __call__(self, x_seq, deterministic: bool):
        B, T, Feat = x_seq.shape
        
        attn_scores = nn.Dense(32)(x_seq)
        attn_scores = nn.tanh(attn_scores)
        attn_scores = nn.Dense(1)(attn_scores) 
        
        attn_weights = jax.nn.softmax(attn_scores, axis=1)
        context_vector = jnp.sum(x_seq * attn_weights, axis=1) 
        
        x = nn.Dense(128)(context_vector)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.5, deterministic=deterministic)(x)
        x = nn.Dense(self.num_classes)(x)
        return x

class Ultimate_STCN_GraphSNN(nn.Module):
    num_steps: int = 160

    @nn.compact
    def __call__(self, L_norm, X_seq, deterministic: bool):
        B = X_seq.shape[0]
        
        # TCN 전방향 처리
        x_tcn_in = X_seq.squeeze(-1) 
        tcn_out = CausalSpikingTCN(features=64, kernel_size=3)(x_tcn_in) 
        
        # 동적 그래프 파라미터 1회 셋업
        mask1_param = self.param('mask1', jax.nn.initializers.ones, (64, 64))
        mask1 = jax.nn.sigmoid(mask1_param + mask1_param.T) / 2.0
        L_dyn1 = L_norm * jnp.expand_dims(mask1, 0)
        
        mask2_param = self.param('mask2', jax.nn.initializers.ones, (64, 64))
        mask2 = jax.nn.sigmoid(mask2_param + mask2_param.T) / 2.0
        L_dyn2 = L_norm * jnp.expand_dims(mask2, 0)
        
        # 막전위 초기화
        mem_tcn = jnp.zeros((B, 64))
        mem_g1 = jnp.zeros((B, 64, 16))
        mem_g2 = jnp.zeros((B, 64, 32))
        
        potentials_rec = []
        
        for step in range(self.num_steps):
            tcn_step = tcn_out[:, step, :] 
            mem_tcn, spk_tcn = LIF(beta=0.9)(mem_tcn, tcn_step)
            
            cur1 = DynamicLaplacianConv(out_features=16)(L_dyn1, jnp.expand_dims(spk_tcn, -1))
            mem_g1, spk1 = LIF(beta=0.85)(mem_g1, cur1)
            
            cur2 = DynamicLaplacianConv(out_features=32)(L_dyn2, spk1)
            mem_g2, spk2 = LIF(beta=0.9)(mem_g2, cur2)
            
            potentials_rec.append(mem_g2.reshape((B, -1)))
            
        all_time_potentials = jnp.stack(potentials_rec, axis=1) 
        
        out = TemporalAttentionDecoder()(all_time_potentials, deterministic=deterministic)
        return out

# =====================================================================
# 4. JAX XLA 학습 스텝 정의 (핵심 속도 펌핑 엔진)
# =====================================================================
@jax.jit
def train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, L_batch, X_batch, 
            deterministic=False, rngs={'dropout': dropout_key}
        )
        # JAX의 CrossEntropy는 One-hot 대신 직접 라벨을 받아서 계산 가능
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return state, loss, acc

@jax.jit
def eval_step(state, L_batch, X_batch, targets):
    logits = state.apply_fn(
        {'params': state.params}, L_batch, X_batch, 
        deterministic=True
    )
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return acc

@jax.jit
def sstl_train_step(state, L_batch, X_batch, targets, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, L_batch, X_batch, 
            deterministic=False, rngs={'dropout': dropout_key}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # 💡 디코더 외의 모든 그래디언트를 0으로 만들어버림 (SSTL 파인튜닝)
    flat_grads = flax.traverse_util.flatten_dict(grads)
    flat_grads = {k: (v if 'TemporalAttentionDecoder' in k[0] else jnp.zeros_like(v)) for k, v in flat_grads.items()}
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

# 시드 설정
rng = jax.random.PRNGKey(42)

print("\n" + "="*50)
print("⚡ JAX/Flax 가속: 동적 리만 + Spiking TCN 5-Fold 학습 시작")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(kf_global.split(all_subjects)):
    print(f"\n🚀 [Fold {fold + 1}/5] 시작")
    global_train_subjs = [all_subjects[i] for i in train_idx]
    global_test_subjs = [all_subjects[i] for i in test_idx]
    
    L_train, X_train, y_train = load_graph_data(global_train_subjs)
    if L_train is None: continue
    
    # PyTorch 텐서를 NumPy 배열로 변환해서 DataLoader에 태움
    train_loader = DataLoader(
        TensorDataset(L_train, X_train, y_train), 
        batch_size=32, shuffle=True, drop_last=True # JAX JIT은 고정 크기 배치를 좋아함
    )
    
    # 모델 초기화 (Dummy 데이터 필요)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_L = jnp.ones((1, 64, 64))
    dummy_X = jnp.ones((1, 160, 64, 1))
    model = Ultimate_STCN_GraphSNN(num_steps=160)
    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_L, dummy_X, deterministic=True)
    
    # TrainState (옵티마이저 묶기)
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # 글로벌 학습 루프
    for epoch in range(30): 
        for batch_L, batch_X, batch_y in train_loader:
            # JAX array로 변환
            j_L = jnp.array(batch_L.numpy()[:, ...])
            # 시간축 건너뛰기 160 압축을 로더 단에서 처리
            j_X = jnp.array(batch_X.numpy()[:, ::3, :, :]) 
            j_y = jnp.array(batch_y.numpy())
            
            rng, dropout_key = jax.random.split(rng)
            state, loss, acc = train_step(state, j_L, j_X, j_y, dropout_key)

    # 평가
    L_unseen, X_unseen, y_unseen = load_graph_data(global_test_subjs)
    unseen_loader = DataLoader(TensorDataset(L_unseen, X_unseen, y_unseen), batch_size=32, shuffle=False)
    
    unseen_corr, unseen_tot = 0, 0
    for batch_L, batch_X, batch_y in unseen_loader:
        j_L = jnp.array(batch_L.numpy())
        j_X = jnp.array(batch_X.numpy()[:, ::3, :, :])
        j_y = jnp.array(batch_y.numpy())
        
        batch_acc = eval_step(state, j_L, j_X, j_y)
        unseen_corr += batch_acc.item() * j_L.shape[0]
        unseen_tot += j_L.shape[0]
            
    true_global_acc = 100 * unseen_corr / unseen_tot
    global_acc_list.append(true_global_acc)
    print(f"🔥 Fold {fold + 1} 진짜 Global Test Acc (p0): {true_global_acc:.2f}%")
    
    global_params_backup = state.params
    fold_sstl_accs = []
    
    # SSTL (Subject Specific Transfer Learning)
    for subj in global_test_subjs:
        L_sub, X_sub, y_sub = load_graph_data([subj])
        if L_sub is None: continue
        
        kf_sub = KFold(n_splits=4, shuffle=True, random_state=42)
        subj_fold_accs = []
        
        for sub_train_idx, sub_test_idx in kf_sub.split(L_sub):
            # 상태 복원 및 옵티마이저 재설정 (SSTL용 lr=5e-4)
            sstl_tx = optax.adam(learning_rate=5e-4)
            sstl_state = train_state.TrainState.create(apply_fn=model.apply, params=global_params_backup, tx=sstl_tx)
            
            sub_train_loader = DataLoader(TensorDataset(L_sub[sub_train_idx], X_sub[sub_train_idx], y_sub[sub_train_idx]), batch_size=16, shuffle=True, drop_last=True)
            sub_test_loader = DataLoader(TensorDataset(L_sub[sub_test_idx], X_sub[sub_test_idx], y_sub[sub_test_idx]), batch_size=16, shuffle=False)
            
            for _ in range(15): 
                for batch_L, batch_X, batch_y in sub_train_loader:
                    j_L = jnp.array(batch_L.numpy())
                    j_X = jnp.array(batch_X.numpy()[:, ::3, :, :])
                    j_y = jnp.array(batch_y.numpy())
                    
                    rng, dropout_key = jax.random.split(rng)
                    sstl_state, loss, acc = sstl_train_step(sstl_state, j_L, j_X, j_y, dropout_key)
                    
            corr, tot = 0, 0
            for batch_L, batch_X, batch_y in sub_test_loader:
                j_L = jnp.array(batch_L.numpy())
                j_X = jnp.array(batch_X.numpy()[:, ::3, :, :])
                j_y = jnp.array(batch_y.numpy())
                
                batch_acc = eval_step(sstl_state, j_L, j_X, j_y)
                corr += batch_acc.item() * j_L.shape[0]
                tot += j_L.shape[0]
            subj_fold_accs.append(100 * corr / tot)
            
        fold_sstl_accs.append(np.mean(subj_fold_accs))
        
    avg_sstl_fold = np.mean(fold_sstl_accs)
    sstl_acc_list.append(avg_sstl_fold)
    print(f"🎯 Fold {fold + 1} SSTL 평균 정확도 (ps): {avg_sstl_fold:.2f}%")

print("\n" + "="*50)
print(f"🏆 5-Fold 최종 평균 Global Test Acc (p0): {np.mean(global_acc_list):.2f}%")
print(f"🏆 5-Fold 최종 평균 SSTL Acc (ps): {np.mean(sstl_acc_list):.2f}%")
print("="*50)