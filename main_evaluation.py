import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import mne
import time
from sklearn.model_selection import KFold, train_test_split

import jax
import jax.numpy as jnp

try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    print("TPU initialized.")
except Exception as e:
    print(f"GPU/CPU initialized. Devices: {jax.device_count()}")

from jax import random as jrandom
import flax.linen as nn
from flax.training import train_state
import optax
from functools import partial
from typing import Any
from flax.jax_utils import replicate, unreplicate

mne.set_log_level('ERROR')
DATA_DIR = './07_Data'

all_subjs = [f'S{i:03d}' for i in range(1, 110)]
exclude = ['S088', 'S092', 'S100', 'S104'] 
subjects = sorted([s for s in all_subjs if s not in exclude])
subj2idx = {s: i for i, s in enumerate(subjects)}
NUM_SUBJECTS = len(subjects)

# ==========================================
# 1. Data Loading 
# ==========================================
def load_balanced_data(subj_list, verbose=False):
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError("Data path error.")
        
    X_out, Y_out, Y_subj_out = [], [], []

    for s in subj_list:
        subj_dir = os.path.join(DATA_DIR, s)
        subj_x, subj_y = [], []
        s_idx = subj2idx[s]
        
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
        Y_subj_raw = np.full(Y_raw.shape, s_idx)
        
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
        Y_subj_out.extend(Y_subj_raw[bal_idx])
        
    if len(X_out) == 0: return np.array([]), np.array([]), np.array([])
    return np.concatenate(X_out), np.array(Y_out), np.array(Y_subj_out)

num_devices = jax.device_count() 
batch_size = 64 
per_device = max(1, batch_size // num_devices)
batch_size = per_device * num_devices 

def prepare_gpu_data(X, Y, Y_subj, batch_size, shuffle=True):
    if len(X) == 0:
        return jax.device_put(np.array([])), jax.device_put(np.array([])), jax.device_put(np.array([])), 0, 0
        
    num_batches = int(np.ceil(len(X) / batch_size))
    target_size = num_batches * batch_size
    
    if target_size > len(X):
        indices = np.arange(target_size) % len(X)
        X, Y, Y_subj = X[indices], Y[indices], Y_subj[indices]
        
    if shuffle:
        perm = np.random.permutation(len(X))
        X, Y, Y_subj = X[perm], Y[perm], Y_subj[perm]
        
    X_b = X.astype(np.float32).reshape(num_batches, num_devices, per_device, *X.shape[1:])
    Y_b = Y.astype(np.int32).reshape(num_batches, num_devices, per_device)
    
    unique_subjs = np.unique(Y_subj)
    subj_map = {old: new for new, old in enumerate(unique_subjs)}
    Y_subj_mapped = np.array([subj_map[s] for s in Y_subj])
    Y_subj_b = Y_subj_mapped.astype(np.int32).reshape(num_batches, num_devices, per_device)
    
    return jax.device_put(X_b), jax.device_put(Y_b), jax.device_put(Y_subj_b), num_batches, len(unique_subjs)

# ==========================================
# 2. Fundamental Operations & Riemannian Math
# ==========================================
@jax.custom_vjp
def grl(x, alpha):
    return x

def grl_fwd(x, alpha):
    return x, alpha

def grl_bwd(alpha, g):
    return (-alpha * g, None)

grl.defvjp(grl_fwd, grl_bwd)

@jax.custom_vjp
def spike_fn(x, alpha=3.0):
    return jnp.where(x > 0, 1.0, 0.0)

def spike_fn_fwd(x, alpha):
    return spike_fn(x, alpha), (x, alpha)

def spike_fn_bwd(res, g):
    x, alpha = res
    grad_x = g * (1.0 / (1.0 + jnp.square(alpha * x)))
    return (grad_x, None) 

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

def get_synaptic_trace(spikes, decay=0.9):
    spikes_seq = jnp.moveaxis(spikes, 1, 0) 
    def scan_fn(tr, s_t):
        tr_next = tr * decay + s_t
        return tr_next, tr_next
    _, trace_seq = jax.lax.scan(jax.checkpoint(scan_fn), jnp.zeros_like(spikes_seq[0]), spikes_seq)
    return jnp.moveaxis(trace_seq, 0, 1) 

def log_euclidean_mapping(x_raw):
    x_val = jnp.squeeze(x_raw, -1)
    x_val = jnp.swapaxes(x_val, 1, 2)
    
    x_val = x_val - jnp.mean(x_val, axis=-1, keepdims=True)
    
    cov = jnp.matmul(x_val, jnp.swapaxes(x_val, 1, 2)) / (x_val.shape[-1] - 1.0)
    cov = cov + jnp.eye(x_val.shape[1]) * 1e-4 
    
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    log_eigvals = jnp.log(jnp.clip(eigvals, a_min=1e-6))
    
    log_cov = jnp.matmul(eigvecs * jnp.expand_dims(log_eigvals, 1), jnp.swapaxes(eigvecs, 1, 2))
    return log_cov 

class TrainState(train_state.TrainState):
    pass 

class CausalConv2D(nn.Module):
    features: int
    kernel_size: tuple
    dilation: tuple = (1, 1)

    @nn.compact
    def __call__(self, x):
        pad_t = (self.kernel_size[0] - 1) * self.dilation[0]
        x_pad = jnp.pad(x, ((0, 0), (pad_t, 0), (0, 0), (0, 0)))
        return nn.Conv(
            self.features, 
            kernel_size=self.kernel_size, 
            kernel_dilation=self.dilation, 
            padding='VALID', 
            use_bias=False
        )(x_pad)

class SpatialUnmixing(nn.Module):
    num_sources: int
    @nn.compact
    def __call__(self, x):
        V = x.shape[2]
        W = self.param('unmix_W', nn.initializers.normal(stddev=0.05), (V, self.num_sources))
        return jnp.einsum('btvf, vs -> btsf', x, W)

class Causal_ALIFLayer2D(nn.Module):
    @nn.compact
    def __call__(self, x, hp_alpha, hp_base_step, hp_base_decay):
        x_seq = jnp.moveaxis(x, 1, 0)
        shape = x.shape[2:]
        
        step_w = jax.nn.softplus(self.param('step_w', nn.initializers.ones, shape))
        decay_w = jax.nn.sigmoid(self.param('decay_w', nn.initializers.normal(stddev=0.1), shape))
        gamma = self.param('gamma', nn.initializers.ones, shape)
        beta = self.param('beta', nn.initializers.zeros, shape)
        
        decay_v = 0.8
        vth_base = 0.5 

        def scan_fn(state, x_t):
            v, vth_dyn = state
            x_t_scaled = x_t * gamma + beta
            v = v * decay_v + x_t_scaled
            
            step_eff = hp_base_step * step_w
            decay_eff = hp_base_decay + (1.0 - hp_base_decay) * decay_w 
            
            vth_eff = vth_base + vth_dyn
            s = spike_fn(v - vth_eff, hp_alpha)
            
            v_next = v - vth_eff * jax.lax.stop_gradient(s) 
            vth_dyn_next = vth_dyn * decay_eff + jax.lax.stop_gradient(s) * step_eff
            
            return (v_next, vth_dyn_next), (s, v)
            
        init_state = (jnp.zeros_like(x_seq[0]), jnp.zeros_like(x_seq[0]))
        _, (spikes, voltages) = jax.lax.scan(jax.checkpoint(scan_fn), init_state, x_seq)
        return jnp.moveaxis(spikes, 0, 1), jnp.moveaxis(voltages, 0, 1)

class TemporalAttentionReadout(nn.Module):
    @nn.compact
    def __call__(self, x_g):
        x_pool = jnp.mean(x_g, axis=3, keepdims=True) 
        
        kernel_size = 7
        pad_t = kernel_size - 1
        x_pool_pad = jnp.pad(x_pool, ((0, 0), (pad_t, 0), (0, 0), (0, 0)))
        
        score = nn.Conv(1, kernel_size=(kernel_size, 1), padding='VALID', name='temp_attn_conv')(x_pool_pad)
        
        attn_weights = jax.nn.softmax(score, axis=1) 
        readout = jnp.sum(x_g * attn_weights, axis=1) 
        return readout, attn_weights

class CausalMultiEdgeFunctionalGraph(nn.Module):
    out_features: int
    decay: float = 0.99 

    @nn.compact
    def __call__(self, x_sources):
        B, T, S, F = x_sources.shape
        x_seq = jnp.moveaxis(x_sources, 1, 0) 

        init_mu = jnp.zeros((B, S, F))
        init_cov = jnp.zeros((B, F, S, S)) 
        init_t_step = jnp.array(0.0)

        W_graph = self.param('graph_weight', nn.initializers.normal(stddev=0.05), (F, S, S))
        W_sym = 0.5 * (W_graph + jnp.transpose(W_graph, (0, 2, 1)))

        def scan_fn(state, x_t):
            t_step, mu, cov = state
            
            t_step = t_step + 1.0
            bias_correction = 1.0 - jnp.power(self.decay, t_step)
            
            mu_next = self.decay * mu + (1.0 - self.decay) * x_t
            mu_corrected = mu_next / bias_correction
            
            x_centered = x_t - mu_corrected
            x_c_f = jnp.transpose(x_centered, (0, 2, 1)) 
            
            outer_prod = jnp.matmul(jnp.expand_dims(x_c_f, -1), jnp.expand_dims(x_c_f, -2))
            
            cov_next = self.decay * cov + (1.0 - self.decay) * outer_prod
            cov_corrected = cov_next / bias_correction

            std = jnp.sqrt(jnp.diagonal(cov_corrected, axis1=2, axis2=3) + 1e-8) 
            std_matrix = jnp.expand_dims(std, 3) * jnp.expand_dims(std, 2) 
            
            corr = cov_corrected / (std_matrix + 1e-8)

            A_eff = jax.nn.relu(corr * jnp.expand_dims(W_sym, 0))
            D_hat = jnp.sum(A_eff, axis=3) + 1e-8
            norm_matrix = jnp.sqrt(jnp.expand_dims(D_hat, 2) * jnp.expand_dims(D_hat, 3))
            A_norm = A_eff / norm_matrix 

            x_g_t = jnp.einsum('bfvw, bfw -> bfv', A_norm, x_c_f)
            x_g_t = jnp.transpose(x_g_t, (0, 2, 1)) 

            return (t_step, mu_next, cov_next), x_g_t

        _, x_g_seq = jax.lax.scan(jax.checkpoint(scan_fn), (init_t_step, init_mu, init_cov), x_seq)
        x_g = jnp.moveaxis(x_g_seq, 0, 1) 

        x_out = nn.Dense(self.out_features, use_bias=False)(x_g)
        return x_out

# ==========================================
# 3. Model Architecture
# ==========================================
class TrialBuffer_Hybrid_SNN_Final(nn.Module):
    num_train_classes: int

    @nn.compact
    def __call__(self, x, hp, grl_alpha, train_bn=True, train_drop=True):
        
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        std = jnp.std(x, axis=(2, 3), keepdims=True) + 1e-8
        x_norm = (x - mean) / std
        
        log_cov = log_euclidean_mapping(x_norm) 
        log_cov = jax.lax.stop_gradient(log_cov)
        
        V = log_cov.shape[-1]
        scaling_matrix = jnp.ones((V, V)) + (jnp.sqrt(2.0) - 1.0) * (jnp.ones((V, V)) - jnp.eye(V))
        scaling_matrix_expanded = jnp.expand_dims(scaling_matrix, axis=0) 
        scaled_log_cov = log_cov * scaling_matrix_expanded
        
        idx_triu = jnp.triu_indices(V)
        tangent_feat = scaled_log_cov[:, idx_triu[0], idx_triu[1]] 

        if train_drop:
            sensor_mask = jrandom.bernoulli(self.make_rng('dropout_sensor'), p=1.0 - hp['drop_sensor'], shape=(x.shape[0], 1, x.shape[2], 1))
            x_snn_input = x_norm * sensor_mask * (1.0 / (1.0 - hp['drop_sensor'] + 1e-8))
        else:
            x_snn_input = x_norm

        x_mu_t = CausalConv2D(8, kernel_size=(32, 1))(x_snn_input) 
        x_mu_t = nn.LayerNorm(reduction_axes=(-2, -1))(x_mu_t)
        
        x_beta_t = CausalConv2D(8, kernel_size=(16, 1))(x_snn_input) 
        x_beta_t = nn.LayerNorm(reduction_axes=(-2, -1))(x_beta_t)
        
        num_sources = 24 
        x_mu_s = SpatialUnmixing(num_sources)(x_mu_t)
        x_mu_s = nn.LayerNorm(reduction_axes=(-2, -1))(x_mu_s)
        
        x_beta_s = SpatialUnmixing(num_sources)(x_beta_t)
        x_beta_s = nn.LayerNorm(reduction_axes=(-2, -1))(x_beta_s)

        x_sources = jnp.concatenate([x_mu_s, x_beta_s], axis=-1)

        x2_current = CausalConv2D(128, kernel_size=(8, 1))(x_sources)
        x2_current = nn.LayerNorm(reduction_axes=(-2, -1))(x2_current)
        spk2, _ = Causal_ALIFLayer2D()(x2_current, hp['a2'], hp['step_th'], hp['decay_th'])

        shared_class_mask = None

        spk2_trace = get_synaptic_trace(spk2, decay=0.9)
        x_g_spk2 = CausalMultiEdgeFunctionalGraph(128)(nn.Dense(16, use_bias=False)(spk2_trace))
        
        readout_spk2, attn_weights_spk2 = TemporalAttentionReadout()(x_g_spk2)
        
        if train_drop: 
            drop_rng = self.make_rng('dropout')
            shared_class_mask = jrandom.bernoulli(drop_rng, p=1.0 - hp['drop_class'], shape=(readout_spk2.shape[0], readout_spk2.shape[1], 1))
            readout_spk2 = readout_spk2 * shared_class_mask * (1.0 / (1.0 - hp['drop_class'] + 1e-8))
            
        x_features_spk2_flat = readout_spk2.reshape((readout_spk2.shape[0], -1))
        x_features_spk2_proj = nn.LayerNorm(name='spk2_norm')(nn.Dense(128)(x_features_spk2_flat))
        aux_logits = nn.Dense(4, name='aux_class_dense')(x_features_spk2_proj)

        x3_main_current = CausalConv2D(128, kernel_size=(4, 1), dilation=(2, 1))(spk2)
        x3_current = nn.LayerNorm(reduction_axes=(-2, -1))(x3_main_current)
        spk3, _ = Causal_ALIFLayer2D()(x3_current, hp['a2'], hp['step_th'], hp['decay_th'])

        spk3_trace = get_synaptic_trace(spk3, decay=0.9)
        spk3_reduced = nn.Dense(16, use_bias=False)(spk3_trace)

        x_g_spk = CausalMultiEdgeFunctionalGraph(128)(spk3_reduced)

        readout_spk, attn_weights_spk3 = TemporalAttentionReadout()(x_g_spk) 
        
        if train_drop and shared_class_mask is not None: 
            readout_spk = readout_spk * shared_class_mask * (1.0 / (1.0 - hp['drop_class'] + 1e-8))

        x_features_spk_flat = readout_spk.reshape((readout_spk.shape[0], -1)) 
        x_features_spk_proj = nn.Dense(128, name='spk_proj')(x_features_spk_flat)
        x_features_spk_proj = nn.LayerNorm(name='spk_norm')(x_features_spk_proj) 
        
        tangent_bias = nn.Dense(128, name='tangent_bias')(tangent_feat)
        tangent_bias = nn.relu(tangent_bias)
        tangent_bias = nn.LayerNorm(name='tangent_norm')(tangent_bias) 
        
        combined_feat = jnp.concatenate([x_features_spk_proj, tangent_bias], axis=-1) 
        
        out_logits = nn.Dense(4, name='class_dense')(combined_feat)
        
        domain_feats = grl(combined_feat, grl_alpha)
        domain_hidden = nn.Dense(128, name='domain_hidden')(domain_feats)
        domain_hidden = nn.relu(domain_hidden)
        domain_logits = nn.Dense(self.num_train_classes, name='domain_out')(domain_hidden)
        
        xai_dict = {
            'attn_weights_aux': attn_weights_spk2,
            'attn_weights_main': attn_weights_spk3
        }
        
        return out_logits, domain_logits, (spk2, spk3), aux_logits, xai_dict

# ==========================================
# 4. PBT Manager (수정 완료)
# ==========================================
class PBTManager:
    def __init__(self, num_workers=20, alpha=1.0):
        self.num_workers = num_workers
        self.alpha = alpha
        self.worker_hps = []
        
        golden_hp = {
            'lr': 1.0e-3,
            'drop_sensor': 0.1,   
            'drop_class': 0.2, 
            'noise_scale': 0.05,  
            'a1': 3.0, 'a2': 3.0,
            'step_th': 0.1, 'decay_th': 0.95 
        } 
        
        for i in range(num_workers):
            if i == 0:
                self.worker_hps.append(golden_hp)
            else:
                self.worker_hps.append({
                    'lr': 10**np.random.uniform(-4.0, -2.5),
                    'drop_sensor': np.random.uniform(0.05, 0.15),
                    'drop_class': np.random.uniform(0.1, 0.3), 
                    'noise_scale': np.random.uniform(0.01, 0.08), 
                    'a1': np.random.uniform(2.0, 5.0), 'a2': np.random.uniform(2.0, 5.0),
                    'step_th': np.random.uniform(0.01, 0.3), 'decay_th': np.random.uniform(0.90, 0.99)
                })
            
    def exploit_and_explore(self, states, fitness_scores):
        sorted_idx = np.argsort(fitness_scores)
        
        # 🔥 [수정] 25% 학살 -> 하위 10%만 교체 (팔다리 보존)
        num_to_replace = max(1, self.num_workers // 10)
        bottom_idx = sorted_idx[:num_to_replace] 
        top_idx = sorted_idx[-num_to_replace:]   
        
        for b, t in zip(bottom_idx, top_idx):
            new_hp = self.worker_hps[t].copy()
            for k in new_hp:
                # 🔥 [수정] 돌연변이 변동폭 대폭 축소 (0.95 ~ 1.05)
                mutation_factor = np.random.uniform(0.95, 1.05)
                
                if k == 'lr':
                    new_hp[k] = jnp.clip(new_hp[k] * mutation_factor, 5e-5, 5e-3)
                elif 'decay' in k or 'drop' in k:
                    new_hp[k] = jnp.clip(new_hp[k] * mutation_factor, 0.05, 0.99)
                elif 'a' in k:
                    new_hp[k] = jnp.clip(new_hp[k] * mutation_factor, 1.0, 20.0)
                elif 'step_th' in k:
                    new_hp[k] = jnp.clip(new_hp[k] * mutation_factor, 0.01, 0.5)
                elif k == 'noise_scale':
                    new_hp[k] = jnp.clip(new_hp[k] * mutation_factor, 0.0, 0.1)
                else:
                    new_hp[k] = new_hp[k] * mutation_factor
                    
            self.worker_hps[b] = new_hp

            new_lr = jnp.array(new_hp['lr'], dtype=jnp.float32)
            t_opt_state = states[t].opt_state
            
            new_hyperparams = {hk: (new_lr if hk == 'learning_rate' else hv) 
                               for hk, hv in t_opt_state.hyperparams.items()}
            
            new_opt_state = t_opt_state._replace(hyperparams=new_hyperparams)
            states[b] = states[t].replace(opt_state=new_opt_state)
            
        return states

# ==========================================
# 5. Main Training Loop
# ==========================================
import time

kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(subjects))

for FOLD_IDX, (train_idx, test_idx) in enumerate(splits):
    if FOLD_IDX != 0: continue

    print(f"\n{'='*50}\n[PHASE 1] TrialBuffer Hybrid SNN (OOM Fixed, Final Check)\n{'='*50}")
    
    train_subjs = [subjects[i] for i in train_idx]
    test_subjs = [subjects[i] for i in test_idx]
    
    X_all, Y_all, Y_subj_all = load_balanced_data(train_subjs, verbose=False)
    if len(X_all) > 0: X_all = np.transpose(X_all, (0, 2, 1))[:, :, :, np.newaxis]

    unique_train_subjs = np.unique(Y_subj_all)
    subj_map = {old: new for new, old in enumerate(unique_train_subjs)}
    Y_subj_mapped = np.array([subj_map[s] for s in Y_subj_all])
    num_train_classes = len(unique_train_subjs)

    seed = int(time.time()) % 10000
    X_tr, X_val, Y_tr, Y_val, Y_subj_tr, Y_subj_val = train_test_split(
        X_all, Y_all, Y_subj_mapped, test_size=0.2, random_state=seed
    )

    num_train_batches = int(np.ceil(len(X_tr) / batch_size))
    target_size = num_train_batches * batch_size
    if target_size > len(X_tr):
        indices = np.arange(target_size) % len(X_tr)
        X_tr, Y_tr, Y_subj_tr = X_tr[indices], Y_tr[indices], Y_subj_tr[indices]

    perm = np.random.permutation(len(X_tr))
    X_tr, Y_tr, Y_subj_tr = X_tr[perm], Y_tr[perm], Y_subj_tr[perm]

    X_tr_gpu = jax.device_put(X_tr.astype(np.float32).reshape(num_train_batches, num_devices, per_device, *X_tr.shape[1:]))
    Y_tr_gpu = jax.device_put(Y_tr.astype(np.int32).reshape(num_train_batches, num_devices, per_device))
    Y_subj_tr_gpu = jax.device_put(Y_subj_tr.astype(np.int32).reshape(num_train_batches, num_devices, per_device))

    X_val_gpu, Y_val_gpu, _, num_val_batches, _ = prepare_gpu_data(X_val, Y_val, np.zeros_like(Y_val), batch_size, shuffle=False)

    pbt = PBTManager(num_workers=20, alpha=1.0)
    rng = jrandom.PRNGKey(seed + FOLD_IDX)
    
    global_model = TrialBuffer_Hybrid_SNN_Final(num_train_classes=num_train_classes)
    global_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.001, weight_decay=0.001) 

    population_states = []
    var = global_model.init(rng, jnp.ones((1, 480, 64, 1)), hp={k: jnp.array(v, dtype=jnp.float32) for k, v in pbt.worker_hps[0].items()}, grl_alpha=0.0, train_bn=False, train_drop=False)
    params = var['params']

    for i in range(20):
        hp = pbt.worker_hps[i]
        state = TrainState.create(
            apply_fn=global_model.apply, params=params, tx=global_tx
        )
        new_hyperparams = {'learning_rate': jnp.array(hp['lr'], dtype=jnp.float32)}
        state = state.replace(opt_state=state.opt_state._replace(hyperparams=new_hyperparams))
        population_states.append(state)

    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))
    def train_step(state, x, y, y_subj, drop_sensor_rng, drop_rng, noise_rng, hp, grl_alpha):
        noise = jrandom.normal(noise_rng, x.shape) * hp['noise_scale']
        x_aug = x + noise
        
        def loss_fn(params):
            (logits, domain_logits, (spk2, spk3), aux_logits, _) = state.apply_fn(
                {'params': params}, 
                x_aug, hp, grl_alpha, train_bn=True, train_drop=True, 
                rngs={'dropout_sensor': drop_sensor_rng, 'dropout': drop_rng}
            )
            smooth_y = optax.smooth_labels(jax.nn.one_hot(y, 4), 0.1)
            mi_loss = jnp.mean(optax.softmax_cross_entropy(logits, smooth_y))
            
            aux_loss = jnp.mean(optax.softmax_cross_entropy(aux_logits, smooth_y))
            
            domain_loss = jnp.mean(optax.softmax_cross_entropy(domain_logits, jax.nn.one_hot(y_subj, num_train_classes)))
            
            neuron_rates2_batch = jnp.mean(spk2, axis=(0, 1))
            neuron_rates3_batch = jnp.mean(spk3, axis=(0, 1))
            target_rate = 0.05
            
            rate_loss2 = jnp.mean(jnp.abs(neuron_rates2_batch - target_rate))
            rate_loss3 = jnp.mean(jnp.abs(neuron_rates3_batch - target_rate))
            
            total_loss = mi_loss + (0.3 * aux_loss) + domain_loss + (10.0 * rate_loss2) + (20.0 * rate_loss3)
            return total_loss, (total_loss, mi_loss, domain_loss, jnp.mean(spk2), jnp.mean(spk3))
            
        grads, (tr_loss, m_loss, d_loss, r2, r3) = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, 'batch'))
        return state, jax.lax.pmean(tr_loss, 'batch'), jax.lax.pmean(d_loss, 'batch'), jax.lax.pmean(r2, 'batch'), jax.lax.pmean(r3, 'batch')

    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, None, 0, 0))
    def get_eval_data_pmap(state, x, hp, drop_sensor_rng, drop_rng):
        logits, _, _, _, xai_dict = state.apply_fn(
            {'params': state.params}, 
            x, hp, 0.0, train_bn=False, train_drop=True, 
            rngs={'dropout_sensor': drop_sensor_rng, 'dropout': drop_rng}
        )
        return logits, xai_dict

    for gen in range(1, 31): 
        # 🔥 [수정] GRL Alpha 속도 대폭 하향. 2차 함수로 웜업 & 최고점 0.4 제한
        p_gen = gen / 30.0
        current_alpha = jnp.array(0.4 * (p_gen ** 2), dtype=jnp.float32) 
        
        print(f"Gen {gen:02d} | Fold {FOLD_IDX} Training... (GRL Alpha: {current_alpha:.2f})")
        worker_fitnesses, worker_val_means = [], []
        
        for w_idx in range(20):
            hp = pbt.worker_hps[w_idx]
            p_state = replicate(population_states[w_idx]) 
            dynamic_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in hp.items()}
            
            gen_losses, gen_d_losses, gen_r2, gen_r3 = [], [], [], []
            for _ in range(5): 
                for b in range(num_train_batches):
                    rng, drop_sensor_rng, drop_rng, noise_rng = jrandom.split(rng, 4)
                    drop_sensor_rngs = jrandom.split(drop_sensor_rng, num_devices)
                    drop_rngs = jrandom.split(drop_rng, num_devices)
                    noise_rngs = jrandom.split(noise_rng, num_devices)
                    
                    p_state, p_loss, d_loss, p_r2, p_r3 = train_step(
                        p_state, X_tr_gpu[b], Y_tr_gpu[b], Y_subj_tr_gpu[b], 
                        drop_sensor_rngs, drop_rngs, noise_rngs, dynamic_hp, current_alpha
                    )
                    gen_losses.append(jnp.mean(p_loss))
                    gen_d_losses.append(jnp.mean(d_loss))
                    gen_r2.append(jnp.mean(p_r2))
                    gen_r3.append(jnp.mean(p_r3))
                    
            unrep_state = unreplicate(p_state)
            
            ensemble_correct = 0
            total_val_samples = len(Y_val)
            batch_accs = []
            
            MC_PASSES_VAL = 3 
            
            for b in range(num_val_batches):
                mc_val_probs = 0
                for _ in range(MC_PASSES_VAL):
                    rng, val_drop_sensor_rng, val_drop_rng = jrandom.split(rng, 3)
                    val_drop_sensor_rngs = jrandom.split(val_drop_sensor_rng, num_devices)
                    val_drop_rngs = jrandom.split(val_drop_rng, num_devices)
                    
                    logits, _ = get_eval_data_pmap(p_state, X_val_gpu[b], dynamic_hp, val_drop_sensor_rngs, val_drop_rngs)
                    mc_val_probs += jax.nn.softmax(logits, axis=-1)
                
                probs = mc_val_probs / MC_PASSES_VAL
                preds = jnp.argmax(probs, axis=-1)
                
                actual_batch_size = total_val_samples - (b * batch_size) if b == num_val_batches - 1 else batch_size
                if actual_batch_size <= 0: break
                
                preds = preds.flatten()[:actual_batch_size]
                batch_y_flat = Y_val_gpu[b].flatten()[:actual_batch_size]
                
                correct_cnt = jnp.sum(preds == batch_y_flat)
                ensemble_correct += correct_cnt
                batch_accs.append(correct_cnt / actual_batch_size)
                
            val_acc_mean = ensemble_correct / total_val_samples
            val_acc_std = np.std(batch_accs) if len(batch_accs) > 0 else 0.0
            fitness = val_acc_mean - (pbt.alpha * val_acc_std)
            
            population_states[w_idx] = unrep_state 
            worker_fitnesses.append(fitness)
            worker_val_means.append(val_acc_mean)

            print(f"  [Worker {w_idx:02d}] -> Loss: {float(np.mean(gen_losses)):.4f} | D_Loss: {float(np.mean(gen_d_losses)):.4f} | Val Acc: {val_acc_mean*100:.1f}% | FR2: {float(np.mean(gen_r2)):.3f} | FR3: {float(np.mean(gen_r3)):.3f}")
    
            
        population_fitnesses = np.array(worker_fitnesses)
        best_w_idx_gen = int(np.argmax(population_fitnesses))
        print(f"Gen {gen:02d} Done | Best Val Acc: {worker_val_means[best_w_idx_gen]*100:.1f}%")
        population_states = pbt.exploit_and_explore(population_states, population_fitnesses)

    top_k = 5 
    top_indices = np.argsort(population_fitnesses)[-top_k:]
    top_states = [population_states[idx] for idx in top_indices]
    top_hps = [pbt.worker_hps[idx] for idx in top_indices]

    print(f"\n{'='*50}\n[PHASE 2] Zero-Shot Evaluation (MC Dropout Ensemble & XAI Extract)\n{'='*50}")
    
    zeroshot_accuracies = []
    MC_PASSES = 10 
    
    subject_xai_main = []
    subject_xai_aux = []
    
    for subj in test_subjs:
        X_subj, Y_subj, _ = load_balanced_data([subj], verbose=False)
        if len(X_subj) == 0: continue
        
        X_subj = np.transpose(X_subj, (0, 2, 1))[:, :, :, np.newaxis] 
        
        num_test_batches = int(np.ceil(len(X_subj) / batch_size))
        target_size = num_test_batches * batch_size
        if target_size > len(X_subj):
            indices = np.arange(target_size) % len(X_subj)
            X_subj, Y_subj = X_subj[indices], Y_subj[indices]

        X_test_gpu = jax.device_put(X_subj.astype(np.float32).reshape(num_test_batches, num_devices, per_device, *X_subj.shape[1:]))
        Y_test_gpu = jax.device_put(Y_subj.astype(np.int32).reshape(num_test_batches, num_devices, per_device))
        
        ensemble_correct = 0
        replicated_top_states = [replicate(w) for w in top_states]
        
        for b in range(num_test_batches):
            batch_x = X_test_gpu[b]
            batch_y = Y_test_gpu[b]
            
            summed_probs = 0
            summed_xai_main = 0
            summed_xai_aux = 0
            
            for rep_a_state, w_hp in zip(replicated_top_states, top_hps):
                dynamic_hp = {k: jnp.array(v, dtype=jnp.float32) for k, v in w_hp.items()}
                
                mc_probs = 0
                mc_xai_main = 0
                mc_xai_aux = 0
                
                for mc in range(MC_PASSES):
                    rng, drop_sensor_rng, drop_rng = jrandom.split(rng, 3)
                    drop_sensor_rngs = jrandom.split(drop_sensor_rng, num_devices)
                    drop_rngs = jrandom.split(drop_rng, num_devices)
                    
                    logits, xai_dict = get_eval_data_pmap(rep_a_state, batch_x, dynamic_hp, drop_sensor_rngs, drop_rngs) 
                    mc_probs += jax.nn.softmax(logits, axis=-1)
                    
                    mc_xai_main += np.array(xai_dict['attn_weights_main'])
                    mc_xai_aux += np.array(xai_dict['attn_weights_aux'])
                
                probs = mc_probs / MC_PASSES
                summed_probs += probs
                
                summed_xai_main += (mc_xai_main / MC_PASSES)
                summed_xai_aux += (mc_xai_aux / MC_PASSES)
                
            avg_probs = summed_probs / top_k
            preds = jnp.argmax(avg_probs, axis=-1)
            
            actual_batch_size = len(Y_subj) - (b * batch_size) if b == num_test_batches - 1 else batch_size
            if actual_batch_size <= 0: break
            
            preds = preds.flatten()[:actual_batch_size]
            batch_y_flat = batch_y.flatten()[:actual_batch_size]
            
            ensemble_correct += jnp.sum(preds == batch_y_flat)
            
            subject_xai_main.append(summed_xai_main / top_k)
            subject_xai_aux.append(summed_xai_aux / top_k)
            
        subj_acc = (ensemble_correct / len(Y_subj)) * 100
        zeroshot_accuracies.append(subj_acc)
        print(f"Subject {subj} | Zero-shot Test Acc: {subj_acc:.2f}%")
        
    final_zeroshot_acc = np.mean(zeroshot_accuracies)
    print(f"\n[FINAL RESULT] Mean Zero-shot Accuracy: {final_zeroshot_acc:.2f}%\n")