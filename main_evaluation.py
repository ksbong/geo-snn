
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
    print("TPU 준비 완료")
except Exception as e:
    print("TPU 환경 아님")

from jax import random as jrandom
import flax.linen as nn
from flax.training import train_state
import optax
from functools import partial
from typing import Any
from flax.jax_utils import replicate, unreplicate
from flax.core import unfreeze

mne.set_log_level('ERROR')

DATA_DIR = './07_Data'

def load_balanced_ra_2d(subj_list, desc="Loading", verbose=True):
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"데이터 경로 오류")
        
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
        X_bal, Y_bal = X_raw[bal_idx], Y_raw[bal_idx]
        
        covs = Covariances(estimator='scm').fit_transform(X_bal)
        C_ref = mean_covariance(covs, metric='riemann')
        whiten = invsqrtm(C_ref)
        
        X_ra = np.array([np.dot(whiten, trial) for trial in X_bal])
        X_ra_transposed = np.transpose(X_ra, (0, 2, 1))[:, :, :, np.newaxis]
        
        X_out.append(X_ra_transposed) 
        Y_out.extend(Y_bal)
        
    return np.concatenate(X_out) if len(X_out) > 0 else np.array([]), np.array(Y_out)

num_devices = jax.device_count() 
batch_size = 32 
per_device = batch_size // num_devices

def prepare_gpu_data(X, Y, num_batches, shuffle=True):
    if shuffle:
        perm = np.random.permutation(len(X))
        X = X[perm]
        Y = Y[perm]
    X_b = X[:num_batches * batch_size].astype(np.float32).reshape(num_batches, num_devices, per_device, *X.shape[1:])
    Y_b = Y[:num_batches * batch_size].astype(np.int32).reshape(num_batches, num_devices, per_device)
    return jax.device_put(X_b), jax.device_put(Y_b)

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

class ALIFLayer2D(nn.Module):
    @nn.compact
    def __call__(self, x, hp_alif_d, hp_alif_adp, hp_alif_beta, hp_alpha):
        x_seq = jnp.moveaxis(x, 1, 0)
        def scan_fn(carry, x_t):
            v, theta, prev_s = carry
            theta_next = theta * hp_alif_adp + prev_s * hp_alif_beta
            v_next = v * hp_alif_d + x_t
            vth_t = 0.5 + theta_next
            s = spike_fn(v_next - vth_t, hp_alpha)
            v_next = v_next - jax.lax.stop_gradient(s) * vth_t
            return (v_next, theta_next, jax.lax.stop_gradient(s)), s
        init_carry = (jnp.zeros_like(x_seq[0]), jnp.zeros_like(x_seq[0]), jnp.zeros_like(x_seq[0]))
        _, spikes = jax.lax.scan(scan_fn, init_carry, x_seq)
        return jnp.moveaxis(spikes, 0, 1)

class RG_SNN_PureOriginal(nn.Module):
    @nn.compact
    def __call__(self, x, hp, train=True):
        x1 = nn.Conv(64, kernel_size=(1, 64), padding='VALID', use_bias=False)(x)
        x1 = nn.BatchNorm(use_running_average=not train)(x1)
        if train:
            x1 = x1 * jrandom.bernoulli(self.make_rng('dropout'), p=1.0 - hp['drop1'], shape=x1.shape) * (1.0 / (1.0 - hp['drop1'] + 1e-8))
        spk1 = HR_HLIFLayer2D()(x1, hp['v1_m'], hp['v1_s'], hp['d1_m'], hp['d1_s'], hp['a1'])

        x2 = nn.Conv(128, kernel_size=(32, 1), kernel_dilation=(4, 1), padding='SAME', use_bias=False)(spk1)
        x2 = nn.BatchNorm(use_running_average=not train)(x2)
        if train:
            x2 = x2 * jrandom.bernoulli(self.make_rng('dropout'), p=1.0 - hp['drop2'], shape=x2.shape) * (1.0 / (1.0 - hp['drop2'] + 1e-8))
        spk2 = HR_HLIFLayer2D()(x2, hp['v2_m'], hp['v2_s'], hp['d2_m'], hp['d2_s'], hp['a2'])

        x3 = nn.Conv(128, kernel_size=(32, 1), kernel_dilation=(12, 1), padding='SAME', use_bias=False)(spk2)
        x3 = nn.BatchNorm(use_running_average=not train)(x3)
        if train:
            x3 = x3 * jrandom.bernoulli(self.make_rng('dropout'), p=1.0 - hp['drop3'], shape=x3.shape) * (1.0 / (1.0 - hp['drop3'] + 1e-8))
        spk3 = ALIFLayer2D()(x3, hp['alif_d'], hp['alif_adp'], hp['alif_beta'], hp['a3'])
        
        x_pool = jnp.mean(spk3, axis=2) 
        dense_out = nn.Dense(4)(x_pool) 
        
        def li_scan(v, x_t):
            decay = hp['decay_out']
            v = v * decay + x_t * (1.0 - decay)
            return v, v
            
        _, v_seq = jax.lax.scan(li_scan, jnp.zeros_like(dense_out[:, 0, :]), jnp.moveaxis(dense_out, 1, 0))
        v_seq = jnp.moveaxis(v_seq, 0, 1) 
        
        v_reshaped = jnp.reshape(v_seq, (v_seq.shape[0], 20, 24, 4))
        dp_features = jnp.mean(v_reshaped[:, :, -12:, :], axis=2) - jnp.mean(v_reshaped[:, :, :12, :], axis=2) 
        
        attn_w = nn.Dense(1)(jax.nn.relu(nn.Dense(8)(dp_features)))
        attn_w = jax.nn.softmax(attn_w, axis=1)
        out_logits = jnp.sum(dp_features * attn_w, axis=1)
        
        return out_logits, (spk1, spk2, spk3)

# 🔥 타겟팅된 PBT 최적 범위
class PBTManager:
    def __init__(self, num_workers=20, alpha=1.0):
        self.num_workers = num_workers
        self.alpha = alpha
        self.worker_hps = []
        for _ in range(num_workers):
            self.worker_hps.append({
                'lr': 10**np.random.uniform(-3.5, -2.5),
                'drop1': np.random.uniform(0.35, 0.55),      
                'drop2': np.random.uniform(0.35, 0.50),
                'drop3': np.random.uniform(0.20, 0.40),
                'v1_m': np.random.uniform(0.4, 0.7), 'v1_s': np.random.uniform(0.3, 0.5),
                'd1_m': np.random.uniform(0.9, 1.3), 'd1_s': np.random.uniform(0.2, 0.5),
                'a1': np.random.uniform(2.0, 3.5),
                'v2_m': np.random.uniform(0.3, 0.6), 'v2_s': np.random.uniform(0.3, 0.6),
                'd2_m': np.random.uniform(1.4, 1.9), 'd2_s': np.random.uniform(0.05, 0.25),
                'a2': np.random.uniform(2.0, 3.5),
                'alif_d': np.random.uniform(0.7, 0.85),      
                'alif_adp': np.random.uniform(0.75, 0.95),   
                'alif_beta': np.random.uniform(0.9, 1.3),   
                'a3': np.random.uniform(1.5, 2.5),           
                'decay_out': np.random.uniform(0.5, 0.75),
                'reg_spike': 10**np.random.uniform(-2.5, -1.5) 
            })
            
    def exploit_and_explore(self, states, fitness_scores):
        sorted_idx = np.argsort(fitness_scores)
        bottom_idx = sorted_idx[:self.num_workers//4] 
        top_idx = sorted_idx[-self.num_workers//4:]   
        
        for b, t in zip(bottom_idx, top_idx):
            new_hp = self.worker_hps[t].copy()
            for k in new_hp:
                mutation_factor = np.random.uniform(0.85, 1.15)
                new_val = new_hp[k] * mutation_factor
                
                if 'drop' in k: new_hp[k] = jnp.clip(new_val, 0.1, 0.6) 
                elif k in ['decay_out', 'alif_d']: new_hp[k] = jnp.clip(new_val, 0.5, 0.999)
                elif k == 'alif_adp': new_hp[k] = jnp.clip(new_val, 0.5, 0.999) 
                elif k == 'alif_beta': new_hp[k] = jnp.clip(new_val, 0.1, 2.0)
                elif 's' in k: new_hp[k] = jnp.clip(new_val, 0.01, 1.0) 
                elif 'a' in k: new_hp[k] = jnp.clip(new_val, 1.0, 10.0) 
                else: new_hp[k] = new_val
            self.worker_hps[b] = new_hp

            new_hyperparams = {'learning_rate': jnp.array(new_hp['lr'], dtype=jnp.float32)}
            new_opt_state = states[t].opt_state._replace(hyperparams=new_hyperparams)
            states[b] = states[t].replace(opt_state=new_opt_state)
            
        return states

all_subjs = [f'S{i:03d}' for i in range(1, 110)]
exclude = ['S088', 'S092', 'S100', 'S104'] 
subjects = sorted([s for s in all_subjs if s not in exclude])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(subjects))

for FOLD_IDX, (train_idx, test_idx) in enumerate(splits):
    # 요청대로 FOLD 0만 재실행
    if FOLD_IDX != 0: continue

    print(f"\n{'='*60}\n[FOLD {FOLD_IDX}/5] targeted PBT evo 시작 \n{'='*60}")
    
    CACHE_FILE = f'/workspace/geo-snn/raw_eeg_cache_fold{FOLD_IDX}_tpu.npz'
    save_path = f'/workspace/geo-snn/rg_snn_pure_fold{FOLD_IDX}_tpu_rerun.npz'
    
    val_subject_data = []

    if os.path.exists(CACHE_FILE):
        data = np.load(CACHE_FILE, allow_pickle=True)
        X_tr, Y_tr = data['X_tr'], data['Y_tr']
        X_test, Y_test = data['X_test'], data['Y_test']
        val_subject_data = data['val_subject_data'].tolist()
        print(f"-- 데이터 캐시 로딩 완료 (Val: {len(val_subject_data)}명)")
    else:
        print(f"-- 전체 데이터 로딩 및 전처리 시작...")
        train_subjs_full = [subjects[i] for i in train_idx]
        test_subjs = [subjects[i] for i in test_idx]
        train_subjs, val_subjs = train_test_split(train_subjs_full, test_size=0.25, random_state=42)
        
        X_tr, Y_tr = load_balanced_ra_2d(train_subjs, f"Train Data")
        X_test, Y_test = load_balanced_ra_2d(test_subjs, f"Test Data")
        
        print(f"-- Validation Data 로딩 시작...")
        for s in val_subjs:
            vx, vy = load_balanced_ra_2d([s], desc=f"Val Subj {s}", verbose=False)
            if len(vx) > 0:
                val_subject_data.append((vx, vy))
        
        np.savez_compressed(CACHE_FILE, X_tr=X_tr, Y_tr=Y_tr, X_test=X_test, Y_test=Y_test, val_subject_data=np.array(val_subject_data, dtype=object))

    num_train_batches = len(X_tr) // batch_size
    num_test_batches = len(X_test) // batch_size
    X_test_gpu, Y_test_gpu = prepare_gpu_data(X_test, Y_test, num_test_batches, shuffle=False)

    pbt = PBTManager(num_workers=20, alpha=1.0)
    rng = jrandom.PRNGKey(42 + FOLD_IDX)
    global_model = RG_SNN_PureOriginal()
    global_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.001, weight_decay=0.001)

    population_states = []
    init_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in pbt.worker_hps[0].items()}
    var = global_model.init(rng, jnp.ones((1, 480, 64, 1)), init_hp, train=False)

    for i in range(20):
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
                x, hp, train=True, rngs={'dropout': drop_rng}, mutable=['batch_stats']
            )
            smooth_y = optax.smooth_labels(jax.nn.one_hot(y, 4), 0.1)
            ce_loss = jnp.mean(optax.softmax_cross_entropy(logits, smooth_y))
            spike_loss = hp['reg_spike'] * (jnp.mean(spk1) + jnp.mean(spk2) + jnp.mean(spk3))
            total_loss = ce_loss + spike_loss
            acc = jnp.mean(jnp.argmax(logits, -1) == y)
            return total_loss, (logits, new_vars['batch_stats'], acc, total_loss)
            
        grads, (logits, new_bs, tr_acc, tr_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, 'batch'), batch_stats=jax.lax.pmean(new_bs, 'batch'))
        return state, jax.lax.pmean(tr_acc, 'batch'), jax.lax.pmean(tr_loss, 'batch')

    @jax.jit
    def eval_step_single(state, x, y, hp):
        logits, _ = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, hp, train=False)
        return jnp.mean(jnp.argmax(logits, -1) == y)

    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, None))
    def eval_step_pmap(state, x, y, hp):
        logits, _ = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, hp, train=False)
        return jnp.mean(jnp.argmax(logits, -1) == y)
    
    history_test_acc = []
    
    for gen in range(1, 31): 
        print(f"Gen {gen:02d} | Fold {FOLD_IDX} Train 데이터 shuffle & TPU uploading...")
        X_tr_gpu, Y_tr_gpu = prepare_gpu_data(X_tr, Y_tr, num_train_batches, shuffle=True)
        
        worker_fitnesses, worker_val_means, worker_val_stds = [], [], []
        
        for w_idx in range(20):
            hp = pbt.worker_hps[w_idx]
            p_state = replicate(population_states[w_idx]) 
            dynamic_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in hp.items()}
            
            for _ in range(5): 
                for b in range(num_train_batches):
                    rng, drop_rng = jrandom.split(rng)
                    drop_rngs = jrandom.split(drop_rng, num_devices)
                    p_state, _, _ = train_step(p_state, X_tr_gpu[b], Y_tr_gpu[b], drop_rngs, dynamic_hp)
                    
            unrep_state = unreplicate(p_state)
            subj_accs = []
            
            for vx, vy in val_subject_data:
                chunk_accs = []
                for i in range(0, len(vx), 32):
                    bx = jnp.array(vx[i:i+32])
                    by = jnp.array(vy[i:i+32])
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
            worker_val_stds.append(val_acc_std)
            
        population_fitnesses = np.array(worker_fitnesses)
        best_w_idx_gen = int(np.argmax(population_fitnesses))
        best_hp_temp = pbt.worker_hps[best_w_idx_gen]
        
        print(f"* Gen {gen:02d} 완료 | 1st worker performance: [Mean: {worker_val_means[best_w_idx_gen]*100:.1f}% | Std: {worker_val_stds[best_w_idx_gen]*100:.1f}% | Fitness: {population_fitnesses[best_w_idx_gen]*100:.1f}]")
        
        best_state_gen = replicate(population_states[best_w_idx_gen])
        best_hp_gen = {k: jnp.array(v, dtype=jnp.float32) for k, v in best_hp_temp.items()}
        
        test_accs_gen = []
        for b in range(num_test_batches):
            acc = eval_step_pmap(best_state_gen, X_test_gpu[b], Y_test_gpu[b], best_hp_gen)
            test_accs_gen.append(np.mean(acc))
            
        history_test_acc.append(np.mean(test_accs_gen))
        print(f"# [monitoring] 현재 1등 워커의 실시간 Test Acc: {np.mean(test_accs_gen)*100:.1f}%")

        population_states = pbt.exploit_and_explore(population_states, population_fitnesses)

    best_w_idx = int(np.argmax(population_fitnesses))
    best_final_state = replicate(population_states[best_w_idx])
    best_final_hp = pbt.worker_hps[best_w_idx]
    best_final_hp_jax = {k: jnp.array(v, dtype=jnp.float32) for k, v in best_final_hp.items()}

    final_test_accs = []
    for b in range(num_test_batches):
        acc = eval_step_pmap(best_final_state, X_test_gpu[b], Y_test_gpu[b], best_final_hp_jax)
        final_test_accs.append(np.mean(acc))
        
    final_test_score = np.mean(final_test_accs) * 100
    print(f"\n[FOLD {FOLD_IDX} 최종 결과] RG-SNN Test Accuracy: {final_test_score:.2f}% 🎯\n")

    final_params = unfreeze(best_final_state.params)

    np.savez(save_path, 
             final_test_score=final_test_score,
             best_final_hp=best_final_hp, 
             history_test_acc=history_test_acc,
             model_params=np.array(final_params, dtype=object))
             
    print(f"Fold {FOLD_IDX} 저장 완료: {save_path}")