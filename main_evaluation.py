

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm
from sklearn.model_selection import KFold, train_test_split

import jax
import jax.numpy as jnp

try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    print("✅ TPU 환경 감지")
except Exception as e:
    print(f"✅ GPU/CPU 환경 가동 (디바이스 수: {jax.device_count()}) - RSNN 모드")

from jax import random as jrandom
import flax.linen as nn
from flax.training import train_state
import flax.traverse_util
import optax
from functools import partial
from typing import Any
from flax.jax_utils import replicate, unreplicate
from flax.core import unfreeze

mne.set_log_level('ERROR')
DATA_DIR = './07_Data'

def apply_ra(X_train, X_test=None):
    # 🔥 lwf 적용: 퓨샷(Few-shot) 데이터 행렬 붕괴 원천 차단
    covs = Covariances(estimator='lwf').fit_transform(X_train)
    C_ref = mean_covariance(covs, metric='riemann')
    whiten = invsqrtm(C_ref)
    
    X_train_ra = np.array([np.dot(whiten, trial) for trial in X_train])
    X_train_out = np.transpose(X_train_ra, (0, 2, 1))[:, :, :, np.newaxis]
    
    if X_test is not None and len(X_test) > 0:
        X_test_ra = np.array([np.dot(whiten, trial) for trial in X_test])
        X_test_out = np.transpose(X_test_ra, (0, 2, 1))[:, :, :, np.newaxis]
        return X_train_out, X_test_out
    return X_train_out

def load_balanced_data(subj_list, desc="Loading", verbose=False):
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"🚨 데이터 경로 오류!")
        
    X_out, Y_out = [], []

    for s in subj_list:
        subj_dir = os.path.join(DATA_DIR, s)
        subj_x, subj_y = [], []
        for run in ['R04','R08','R12','R06','R10','R14']:
            path = os.path.join(subj_dir, f'{s}{run}.edf')
            if not os.path.exists(path): continue
            
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.resample(160.0); raw.filter(8.0, 30.0, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            
            t0, t1, t2 = ed.get('T0', 1), ed.get('T1', 2), ed.get('T2', 3)
            mask = np.isin(evs[:, 2], [t0, t1, t2]) if run in ['R04','R08','R12'] else np.isin(evs[:, 2], [t0, t2])
            e = evs[mask].copy()
            
            if run in ['R04','R08','R12']:
                e[e[:, 2] == t0, 2] = 0; e[e[:, 2] == t1, 2] = 1; e[e[:, 2] == t2, 2] = 2
            else:
                e[e[:, 2] == t0, 2] = 0; e[e[:, 2] == t1, 2] = 3; e[e[:, 2] == t2, 2] = 3 
                
            ep = mne.Epochs(raw, e, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0:
                subj_x.append(ep.get_data(copy=True) * 1e6)
                subj_y.extend(ep.events[:, 2])
        
        if not subj_x: continue
        X_raw = np.concatenate(subj_x, axis=0)[:, :, :480]
        Y_raw = np.array(subj_y)
        
        idx_0, idx_1 = np.where(Y_raw == 0)[0], np.where(Y_raw == 1)[0]
        idx_2, idx_3 = np.where(Y_raw == 2)[0], np.where(Y_raw == 3)[0]
        min_cnt = min(len(idx_0), len(idx_1), len(idx_2), len(idx_3))
        if min_cnt == 0: continue 
        
        np.random.shuffle(idx_0); np.random.shuffle(idx_1)
        np.random.shuffle(idx_2); np.random.shuffle(idx_3)
        bal_idx = np.concatenate([idx_0[:min_cnt], idx_1[:min_cnt], idx_2[:min_cnt], idx_3[:min_cnt]])
        np.random.shuffle(bal_idx)
        
        X_out.append(X_raw[bal_idx])
        Y_out.extend(Y_raw[bal_idx])
        
    if len(X_out) == 0: return np.array([]), np.array([])
    return np.concatenate(X_out), np.array(Y_out)

num_devices = jax.device_count() 
batch_size = 32 
per_device = max(1, batch_size // num_devices)
batch_size = per_device * num_devices 

def prepare_gpu_data(X, Y, batch_size, shuffle=True):
    num_batches = int(np.ceil(len(X) / batch_size))
    pad_size = (num_batches * batch_size) - len(X)
    
    if pad_size > 0:
        X = np.concatenate([X, X[:pad_size]], axis=0)
        Y = np.concatenate([Y, Y[:pad_size]], axis=0)
        
    if shuffle:
        perm = np.random.permutation(len(X))
        X = X[perm]
        Y = Y[perm]
        
    X_b = X.astype(np.float32).reshape(num_batches, num_devices, per_device, *X.shape[1:])
    Y_b = Y.astype(np.int32).reshape(num_batches, num_devices, per_device)
    return jax.device_put(X_b), jax.device_put(Y_b), num_batches

@jax.custom_vjp
def spike_fn(x, alpha):
    return jnp.where(x > 0, 1.0, 0.0)

def spike_fn_fwd(x, alpha):
    return spike_fn(x, alpha), (x, alpha)

def spike_fn_bwd(res, g):
    x, alpha = res
    grad_x = g * (1.0 / (1.0 + jnp.square(alpha * x)))
    return (grad_x, None) 

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

class TrainState(train_state.TrainState):
    batch_stats: Any

class SEBlock1D(nn.Module):
    reduction: int = 16
    @nn.compact
    def __call__(self, x):
        b, t, s, c = x.shape
        y = jnp.mean(x, axis=(1, 2)) 
        y = nn.Dense(max(c // self.reduction, 4), use_bias=False)(y)
        y = jax.nn.relu(y)
        y = nn.Dense(c, use_bias=False)(y)
        y = jax.nn.sigmoid(y) 
        y = jnp.reshape(y, (b, 1, 1, c))
        return x * y

# 🔥 [기본 콩나물 뉴런] Layer 1 (공간 압축)에만 사용
class HR_HLIFLayer2D(nn.Module):
    @nn.compact
    def __call__(self, x, hp_v_m, hp_v_s, hp_d_m, hp_d_s, hp_alpha):
        x_seq = jnp.moveaxis(x, 1, 0)
        shape = x.shape[2:]
        raw_vth = self.param('vth_raw', nn.initializers.normal(stddev=1.0), shape)
        raw_decay = self.param('decay_raw', nn.initializers.normal(stddev=1.0), shape)
        vth = jax.nn.softplus(raw_vth * hp_v_s + hp_v_m) + 0.5
        decay = jnp.clip(jax.nn.sigmoid(raw_decay * hp_d_s + hp_d_m), 0.0, 0.99)
        def scan_fn(v, x_t):
            v = v * decay + x_t
            s = spike_fn(v - vth, hp_alpha)
            return v - jax.lax.stop_gradient(s) * vth, s
        _, spikes = jax.lax.scan(scan_fn, jnp.zeros_like(x_seq[0]), x_seq)
        return jnp.moveaxis(spikes, 0, 1)

# 🔥 [피드백 장착 RSNN 뉴런 1] Layer 2에 이식 (기억력 강화)
class Recurrent_HR_HLIFLayer2D(nn.Module):
    @nn.compact
    def __call__(self, x, hp_v_m, hp_v_s, hp_d_m, hp_d_s, hp_alpha):
        x_seq = jnp.moveaxis(x, 1, 0)
        shape = x.shape[2:]
        raw_vth = self.param('vth_raw', nn.initializers.normal(stddev=1.0), shape)
        raw_decay = self.param('decay_raw', nn.initializers.normal(stddev=1.0), shape)
        
        # 순환 가중치(Recurrent Weight) 추가: 자기들끼리 스파이크 통신
        w_rec = self.param('w_rec', nn.initializers.normal(stddev=0.05), (shape[-1], shape[-1]))
        
        vth = jax.nn.softplus(raw_vth * hp_v_s + hp_v_m) + 0.5
        decay = jnp.clip(jax.nn.sigmoid(raw_decay * hp_d_s + hp_d_m), 0.0, 0.99)
        
        def scan_fn(carry, x_t):
            v, prev_s = carry
            rec_input = jnp.dot(prev_s, w_rec) # 과거의 스파이크가 현재의 나를 자극
            v = v * decay + x_t + rec_input
            s = spike_fn(v - vth, hp_alpha)
            v_next = v - jax.lax.stop_gradient(s) * vth
            return (v_next, s), s
            
        init_carry = (jnp.zeros_like(x_seq[0]), jnp.zeros_like(x_seq[0]))
        _, spikes = jax.lax.scan(scan_fn, init_carry, x_seq)
        return jnp.moveaxis(spikes, 0, 1)

# 🔥 [피드백 장착 RSNN 뉴런 2] Layer 3에 이식 (장기 기억 및 적응형 임계값)
class Recurrent_ALIFLayer2D(nn.Module):
    @nn.compact
    def __call__(self, x, hp_alif_d, hp_alif_adp, hp_alif_beta, hp_alpha):
        x_seq = jnp.moveaxis(x, 1, 0)
        shape = x.shape[2:]
        
        # 순환 가중치(Recurrent Weight)
        w_rec = self.param('w_rec', nn.initializers.normal(stddev=0.05), (shape[-1], shape[-1]))
        
        def scan_fn(carry, x_t):
            v, theta, prev_s = carry
            theta_next = theta * hp_alif_adp + prev_s * hp_alif_beta
            rec_input = jnp.dot(prev_s, w_rec)
            v_next = v * hp_alif_d + x_t + rec_input
            vth_t = 0.5 + theta_next
            s = spike_fn(v_next - vth_t, hp_alpha)
            v_next = v_next - jax.lax.stop_gradient(s) * vth_t
            return (v_next, theta_next, s), s
            
        init_carry = (jnp.zeros_like(x_seq[0]), jnp.zeros_like(x_seq[0]), jnp.zeros_like(x_seq[0]))
        _, spikes = jax.lax.scan(scan_fn, init_carry, x_seq)
        return jnp.moveaxis(spikes, 0, 1)

class SE_RG_RSNN_Final(nn.Module):
    @nn.compact
    def __call__(self, x, hp, train_bn=True, train_drop=True):
        
        # Layer 1: 공간 압축 (피드백 없이 순방향)
        x1 = nn.Conv(128, kernel_size=(1, 64), padding='VALID', use_bias=False)(x)
        x1 = SEBlock1D(reduction=16)(x1) 
        x1 = nn.BatchNorm(use_running_average=not train_bn)(x1)
        if train_drop: 
            drop_mask1 = jrandom.bernoulli(self.make_rng('dropout'), p=1.0 - hp['drop1'], shape=(x1.shape[0], 1, 1, x1.shape[3]))
            x1 = x1 * drop_mask1 * (1.0 / (1.0 - hp['drop1'] + 1e-8))
        spk1 = HR_HLIFLayer2D()(x1, hp['v1_m'], hp['v1_s'], hp['d1_m'], hp['d1_s'], hp['a1'])

        # Layer 2: RSNN 피드백 적용
        x2 = nn.Conv(256, kernel_size=(32, 1), kernel_dilation=(4, 1), padding='SAME', use_bias=False)(spk1)
        x2 = SEBlock1D(reduction=16)(x2) 
        x2 = nn.BatchNorm(use_running_average=not train_bn)(x2)
        if train_drop: 
            drop_mask2 = jrandom.bernoulli(self.make_rng('dropout'), p=1.0 - hp['drop2'], shape=(x2.shape[0], 1, 1, x2.shape[3]))
            x2 = x2 * drop_mask2 * (1.0 / (1.0 - hp['drop2'] + 1e-8))
        spk2 = Recurrent_HR_HLIFLayer2D()(x2, hp['v2_m'], hp['v2_s'], hp['d2_m'], hp['d2_s'], hp['a2'])

        # Layer 3: RSNN + 적응형 임계값
        x3 = nn.Conv(256, kernel_size=(32, 1), kernel_dilation=(12, 1), padding='SAME', use_bias=False)(spk2)
        x3 = SEBlock1D(reduction=16)(x3) 
        x3 = nn.BatchNorm(use_running_average=not train_bn)(x3)
        if train_drop: 
            drop_mask3 = jrandom.bernoulli(self.make_rng('dropout'), p=1.0 - hp['drop3'], shape=(x3.shape[0], 1, 1, x3.shape[3]))
            x3 = x3 * drop_mask3 * (1.0 / (1.0 - hp['drop3'] + 1e-8))
        spk3 = Recurrent_ALIFLayer2D()(x3, hp['alif_d'], hp['alif_adp'], hp['alif_beta'], hp['a3'])
        
        x_pool = jnp.mean(spk3, axis=2) 
        dense_out = nn.Dense(4, name='class_dense')(x_pool) 
        
        def li_scan(v, x_t):
            decay = hp['decay_out']
            v = v * decay + x_t * (1.0 - decay)
            return v, v
            
        _, v_seq = jax.lax.scan(li_scan, jnp.zeros_like(dense_out[:, 0, :]), jnp.moveaxis(dense_out, 1, 0))
        v_seq = jnp.moveaxis(v_seq, 0, 1) 
        
        v_reshaped = jnp.reshape(v_seq, (v_seq.shape[0], 20, 24, 4))
        dp_features = jnp.mean(v_reshaped[:, :, -12:, :], axis=2) - jnp.mean(v_reshaped[:, :, :12, :], axis=2) 
        
        attn_w = nn.Dense(1, name='attn_dense2')(jax.nn.relu(nn.Dense(8, name='attn_dense1')(dp_features)))
        attn_w = jax.nn.softmax(attn_w, axis=1)
        out_logits = jnp.sum(dp_features * attn_w, axis=1)
        
        return out_logits, (spk1, spk2, spk3)

class PBTManager:
    def __init__(self, num_workers=8, alpha=1.0):
        self.num_workers = num_workers
        self.alpha = alpha
        self.worker_hps = []
        for _ in range(num_workers):
            self.worker_hps.append({
                'lr': 10**np.random.uniform(-4.0, -2.5),
                'drop1': np.random.uniform(0.2, 0.45),      
                'drop2': np.random.uniform(0.2, 0.45),
                'drop3': np.random.uniform(0.2, 0.45),
                'v1_m': np.random.uniform(0.5, 1.5), 'v1_s': np.random.uniform(0.1, 0.5),
                'd1_m': np.random.uniform(1.0, 3.0), 'd1_s': np.random.uniform(0.1, 0.5),
                'a1': np.random.uniform(2.0, 5.0),
                'v2_m': np.random.uniform(0.5, 1.5), 'v2_s': np.random.uniform(0.1, 0.5),
                'd2_m': np.random.uniform(1.0, 3.0), 'd2_s': np.random.uniform(0.1, 0.5),
                'a2': np.random.uniform(2.0, 5.0),
                'alif_d': np.random.uniform(0.7, 0.95),      
                'alif_adp': np.random.uniform(0.9, 0.99),   
                'alif_beta': np.random.uniform(0.01, 0.5),   
                'a3': np.random.uniform(2.0, 5.0),           
                'decay_out': np.random.uniform(0.5, 0.99),
                'reg_spike': 10**np.random.uniform(-5, -4) 
            })
            
    def exploit_and_explore(self, states, fitness_scores):
        sorted_idx = np.argsort(fitness_scores)
        bottom_idx = sorted_idx[:self.num_workers//4] 
        top_idx = sorted_idx[-self.num_workers//4:]   
        
        for b, t in zip(bottom_idx, top_idx):
            new_hp = self.worker_hps[t].copy()
            for k in new_hp:
                mutation_factor = np.random.uniform(0.8, 1.2)
                new_val = new_hp[k] * mutation_factor
                
                if 'drop' in k: new_hp[k] = jnp.clip(new_val, 0.1, 0.55) 
                elif k in ['decay_out', 'alif_d']: new_hp[k] = jnp.clip(new_val, 0.5, 0.999)
                elif k == 'alif_adp': new_hp[k] = jnp.clip(new_val, 0.85, 0.999) 
                elif k == 'alif_beta': new_hp[k] = jnp.clip(new_val, 0.01, 1.5) 
                elif 's' in k: new_hp[k] = jnp.clip(new_val, 0.01, 2.0) 
                elif 'a' in k: new_hp[k] = jnp.clip(new_val, 1.0, 20.0) 
                else: new_hp[k] = new_val
            self.worker_hps[b] = new_hp

            new_hyperparams = {'learning_rate': jnp.array(new_hp['lr'], dtype=jnp.float32)}
            new_opt_state = states[t].opt_state._replace(hyperparams=new_hyperparams)
            states[b] = states[t].replace(opt_state=new_opt_state)
            
        return states

def create_sstl_mask(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    mask = {k: 'class_dense' in k for k, v in flat_params.items()}
    return flax.traverse_util.unflatten_dict(mask)

all_subjs = [f'S{i:03d}' for i in range(1, 110)]
exclude = ['S088', 'S092', 'S100', 'S104'] 
subjects = sorted([s for s in all_subjs if s not in exclude])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(subjects))

for FOLD_IDX, (train_idx, test_idx) in enumerate(splits):
    if FOLD_IDX != 0: continue

    print(f"\n{'='*60}\n🚀 [PHASE 1] RSNN + PBT 글로벌 진화 훈련 (10 Generations)\n{'='*60}")
    
    train_subjs = [subjects[i] for i in train_idx]
    test_subjs = [subjects[i] for i in test_idx]
    train_subjs, val_subjs = train_test_split(train_subjs, test_size=0.25, random_state=42)
    
    X_tr_list, Y_tr_list = [], []
    for s in train_subjs:
        X, Y = load_balanced_data([s], verbose=False)
        if len(X) > 0:
            X_tr_list.append(apply_ra(X))
            Y_tr_list.extend(Y)
    X_tr = np.concatenate(X_tr_list) if X_tr_list else np.array([])
    Y_tr = np.array(Y_tr_list)

    val_subject_data = []
    for s in val_subjs:
        X, Y = load_balanced_data([s], verbose=False)
        if len(X) > 0:
            val_subject_data.append((apply_ra(X), Y))

    X_tr_gpu, Y_tr_gpu, num_train_batches = prepare_gpu_data(X_tr, Y_tr, batch_size, shuffle=True)

    pbt = PBTManager(num_workers=8, alpha=1.0)
    rng = jrandom.PRNGKey(42 + FOLD_IDX)
    global_model = SE_RG_RSNN_Final()
    global_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.001, weight_decay=0.001)

    population_states = []
    init_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in pbt.worker_hps[0].items()}
    var = global_model.init(rng, jnp.ones((1, 480, 64, 1)), init_hp, train_bn=False, train_drop=False)

    for i in range(8):
        hp = pbt.worker_hps[i]
        state = TrainState.create(
            apply_fn=global_model.apply, params=var['params'], tx=global_tx, batch_stats=var['batch_stats']
        )
        new_hyperparams = {'learning_rate': jnp.array(hp['lr'], dtype=jnp.float32)}
        state = state.replace(opt_state=state.opt_state._replace(hyperparams=new_hyperparams))
        population_states.append(state)

    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, 0, None))
    def train_step(state, x, y, drop_rng, hp):
        def loss_fn(params):
            (logits, (spk1, spk2, spk3)), new_vars = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats}, 
                x, hp, train_bn=True, train_drop=True, rngs={'dropout': drop_rng}, mutable=['batch_stats']
            )
            smooth_y = optax.smooth_labels(jax.nn.one_hot(y, 4), 0.1)
            ce_loss = jnp.mean(optax.softmax_cross_entropy(logits, smooth_y))
            spike_loss = hp['reg_spike'] * (jnp.mean(spk1) + jnp.mean(spk2) + jnp.mean(spk3))
            total_loss = ce_loss + spike_loss
            return total_loss, new_vars['batch_stats']
            
        grads, new_bs = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, 'batch'), batch_stats=jax.lax.pmean(new_bs, 'batch'))
        return state

    @jax.jit
    def eval_step_single(state, x, y, hp):
        logits, _ = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats}, 
            x, hp, train_bn=False, train_drop=False
        )
        return jnp.mean(jnp.argmax(logits, -1) == y)

    # 딱 10세대만 돌림
    for gen in range(1, 11): 
        print(f"Gen {gen:02d} | Fold {FOLD_IDX} Train 데이터 학습 중...")
        worker_fitnesses, worker_val_means = [], []
        
        for w_idx in range(8):
            hp = pbt.worker_hps[w_idx]
            p_state = replicate(population_states[w_idx]) 
            dynamic_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in hp.items()}
            
            for _ in range(5): 
                for b in range(num_train_batches):
                    rng, drop_rng = jrandom.split(rng)
                    drop_rngs = jrandom.split(drop_rng, num_devices)
                    p_state = train_step(p_state, X_tr_gpu[b], Y_tr_gpu[b], drop_rngs, dynamic_hp)
                    
            unrep_state = unreplicate(p_state)
            subj_accs = []
            
            for vx, vy in val_subject_data:
                chunk_accs = []
                for i in range(0, len(vx), 32):
                    bx = jnp.array(vx[i:i+32])
                    by = jnp.array(vy[i:i+32])
                    if len(bx) == 0: continue
                    acc = eval_step_single(unrep_state, bx, by, dynamic_hp)
                    chunk_accs.append(acc)
                if chunk_accs:
                    subj_accs.append(np.mean(chunk_accs))
                    
            val_acc_mean = np.mean(subj_accs)
            val_acc_std = np.std(subj_accs)
            fitness = val_acc_mean - (pbt.alpha * val_acc_std)
            
            population_states[w_idx] = unrep_state
            worker_fitnesses.append(fitness)
            worker_val_means.append(val_acc_mean)
            
        population_fitnesses = np.array(worker_fitnesses)
        best_w_idx_gen = int(np.argmax(population_fitnesses))
        print(f"Gen {gen:02d} 완료 | 1st worker 성능 (Val): [Mean: {worker_val_means[best_w_idx_gen]*100:.1f}%]")
        population_states = pbt.exploit_and_explore(population_states, population_fitnesses)

    top_k = 3 
    top_indices = np.argsort(population_fitnesses)[-top_k:]
    top_states = [population_states[idx] for idx in top_indices]
    top_hps = [pbt.worker_hps[idx] for idx in top_indices]

    print(f"\n{'='*60}\n🔥 [PHASE 2] 논문 룰 기반 4-FOLD SSTL 평가\n{'='*60}")
    
    mask_tree = create_sstl_mask(top_states[0].params)
    sstl_tx = optax.masked(optax.adamw(learning_rate=5e-4, weight_decay=0.001), mask_tree)
    
    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, 0, None))
    def sstl_train_step(state, x, y, drop_rng, hp):
        def loss_fn(params):
            (logits, _ ) = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats}, 
                x, hp, train_bn=False, train_drop=True, rngs={'dropout': drop_rng}
            )
            smooth_y = optax.smooth_labels(jax.nn.one_hot(y, 4), 0.1)
            ce_loss = jnp.mean(optax.softmax_cross_entropy(logits, smooth_y))
            return ce_loss
            
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, 'batch'))
        return state

    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, None))
    def get_logits_pmap(state, x, hp):
        logits, _ = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats}, 
            x, hp, train_bn=False, train_drop=False
        )
        return logits

    sstl_accuracies = []
    
    for subj in test_subjs:
        X_subj, Y_subj = load_balanced_data([subj], desc=f"SSTL Subj {subj}", verbose=False)
        if len(X_subj) < 20: continue
        
        kf_sstl = KFold(n_splits=4, shuffle=True, random_state=42)
        fold_accs = []
        
        for adapt_idx, eval_idx in kf_sstl.split(X_subj):
            X_adapt_raw, X_eval_raw = X_subj[adapt_idx], X_subj[eval_idx]
            Y_adapt, Y_eval = Y_subj[adapt_idx], Y_subj[eval_idx]
            
            X_adapt, X_eval = apply_ra(X_adapt_raw, X_eval_raw)
            
            X_adapt_gpu, Y_adapt_gpu, num_adapt_batches = prepare_gpu_data(X_adapt, Y_adapt, batch_size, shuffle=True)
            X_test_gpu, Y_test_gpu, num_test_batches = prepare_gpu_data(X_eval, Y_eval, batch_size, shuffle=False)
            
            adapted_states = []
            
            for w_state, w_hp in zip(top_states, top_hps):
                sstl_state = TrainState.create(
                    apply_fn=global_model.apply, params=w_state.params, tx=sstl_tx, batch_stats=w_state.batch_stats
                )
                p_sstl_state = replicate(sstl_state)
                dynamic_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in w_hp.items()}
                
                for _ in range(5):
                    for b in range(num_adapt_batches):
                        rng, drop_rng = jrandom.split(rng)
                        drop_rngs = jrandom.split(drop_rng, num_devices)
                        p_sstl_state = sstl_train_step(p_sstl_state, X_adapt_gpu[b], Y_adapt_gpu[b], drop_rngs, dynamic_hp)
                
                adapted_states.append(unreplicate(p_sstl_state))
                
            ensemble_correct = 0
            
            for b in range(num_test_batches):
                batch_x = X_test_gpu[b]
                batch_y = Y_test_gpu[b]
                
                summed_probs = 0
                for a_state, w_hp in zip(adapted_states, top_hps):
                    rep_a_state = replicate(a_state)
                    dynamic_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in w_hp.items()}
                    
                    logits = get_logits_pmap(rep_a_state, batch_x, dynamic_hp) 
                    probs = jax.nn.softmax(logits, axis=-1)
                    summed_probs += probs
                    
                avg_probs = summed_probs / top_k
                preds = jnp.argmax(avg_probs, axis=-1)
                
                actual_batch_size = len(Y_eval) - (b * batch_size) if b == num_test_batches - 1 else batch_size
                if actual_batch_size <= 0: break
                
                preds = preds.flatten()[:actual_batch_size]
                batch_y_flat = batch_y.flatten()[:actual_batch_size]
                
                ensemble_correct += jnp.sum(preds == batch_y_flat)
                
            fold_accs.append((ensemble_correct / len(Y_eval)) * 100)
            
        subj_mean_acc = np.mean(fold_accs)
        sstl_accuracies.append(subj_mean_acc)
        print(f"Subject {subj} | 4-Fold SSTL Test Acc: {subj_mean_acc:.2f}%")
        
    final_sstl_acc = np.mean(sstl_accuracies)
    print(f"\n🏆 [최종 결과] 완전체 RSNN (PBT 10 Gen + 4-Fold SSTL) 평균 정확도: {final_sstl_acc:.2f}% 🚀\n")