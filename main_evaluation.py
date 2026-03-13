import os
import time
import numpy as np
import mne
from scipy.signal import hilbert, savgol_filter
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from sklearn.model_selection import train_test_split, KFold

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

DATA_DIR_PHYSIONET = './raw_data/files/'
if not os.path.exists(DATA_DIR_PHYSIONET):
    DATA_DIR_PHYSIONET = './raw_data/files'

# =========================================================
# [1] SNN Surrogate Spike
# =========================================================
@jax.custom_vjp
def spike_fn(x):
    return jnp.where(x >= 0, 1.0, 0.0)

def spike_fn_fwd(x):
    return spike_fn(x), x

def spike_fn_bwd(res, g):
    x = res
    grad_x = g / jnp.square(1.0 + 5.0 * jnp.abs(x))
    return (grad_x,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

# =========================================================
# [2] Novel Architecture: Geo-HR-SNN (XLA Optimized)
# =========================================================
class Geo_HR_SNN(nn.Module):
    eeg_ch: int = 64
    geo_ch: int = 18
    F1: int = 8
    hid_ch: int = 32
    out_ch: int = 4

    def setup(self):
        # 1. EEG Backbone: Temporal Depthwise Conv -> Spatial Dense Mixing
        self.eeg_temp_conv = nn.Conv(features=self.eeg_ch * self.F1, kernel_size=(32,), feature_group_count=self.eeg_ch, padding='SAME')
        self.eeg_bn1 = nn.BatchNorm()
        self.eeg_spat_dense = nn.Dense(features=self.hid_ch)
        self.eeg_bn2 = nn.BatchNorm()
        self.eeg_drop = nn.Dropout(rate=0.4)
        
        # 2. Geo Backbone: Temporal Depthwise Conv -> Spatial Dense Mixing
        self.geo_temp_conv = nn.Conv(features=self.geo_ch * self.F1, kernel_size=(32,), feature_group_count=self.geo_ch, padding='SAME')
        self.geo_bn1 = nn.BatchNorm()
        self.geo_spat_dense = nn.Dense(features=self.hid_ch)
        self.geo_bn2 = nn.BatchNorm()
        
        # 3. Geo-ALIF Modulator Params
        self.theta_0 = 0.5
        self.beta = 1.8
        self.tau_a = 0.36
        self.gamma = self.param('gamma', nn.initializers.constant(0.5), (self.hid_ch,))
        
        # 4. Classifier
        self.classifier_drop = nn.Dropout(rate=0.5)
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=self.out_ch)

    @nn.compact
    def __call__(self, x_eeg, x_geo, train: bool = True):
        B, T, _ = x_eeg.shape
        
        # EEG Feature Extraction
        xe = self.eeg_temp_conv(x_eeg)
        xe = self.eeg_bn1(xe, use_running_average=not train)
        xe = self.eeg_spat_dense(xe)
        xe = self.eeg_bn2(xe, use_running_average=not train)
        eeg_encoded = self.eeg_drop(nn.relu(xe), deterministic=not train)
        
        # Geo Feature Extraction
        xg = self.geo_temp_conv(x_geo)
        xg = self.geo_bn1(xg, use_running_average=not train)
        xg = self.geo_spat_dense(xg)
        xg = self.geo_bn2(xg, use_running_average=not train)
        geo_mod = nn.sigmoid(xg) 
        
        def scan_fn(carry, inputs):
            mem_lif1, mem_lif2, mem_lif3, mem_alif, eta, mem_li, prev_spk = carry
            cur_eeg, cur_geo_mod = inputs
            
            # HR-Module (3 Parallel LIFs)
            mem_lif1 = 0.9 * mem_lif1 + cur_eeg
            spk1 = spike_fn(mem_lif1 - 0.7)
            mem_lif1 = mem_lif1 * (1.0 - spk1)
            
            mem_lif2 = 0.8 * mem_lif2 + cur_eeg
            spk2 = spike_fn(mem_lif2 - 0.5)
            mem_lif2 = mem_lif2 * (1.0 - spk2)
            
            mem_lif3 = 0.6 * mem_lif3 + cur_eeg
            spk3 = spike_fn(mem_lif3 - 0.3)
            mem_lif3 = mem_lif3 * (1.0 - spk3)
            
            hr_out = (spk1 + spk2 + spk3) / 3.0
            
            # Geo-ALIF Mechanism
            eta = self.tau_a * eta + (1 - self.tau_a) * prev_spk
            theta_t = self.theta_0 + self.beta * eta - (self.gamma * cur_geo_mod)
            
            mem_alif = 0.8 * mem_alif + hr_out
            spk_alif = spike_fn(mem_alif - theta_t)
            mem_alif = mem_alif * (1.0 - spk_alif)
            
            # Leaky Integrator for Pooling
            mem_li = 0.9 * mem_li + spk_alif
            
            new_carry = (mem_lif1, mem_lif2, mem_lif3, mem_alif, eta, mem_li, spk_alif)
            return new_carry, mem_li

        init_carry = (
            jnp.zeros((B, self.hid_ch)), jnp.zeros((B, self.hid_ch)), jnp.zeros((B, self.hid_ch)),
            jnp.zeros((B, self.hid_ch)), jnp.zeros((B, self.hid_ch)), jnp.zeros((B, self.hid_ch)), jnp.zeros((B, self.hid_ch))
        )
        
        inputs = (jnp.swapaxes(eeg_encoded, 0, 1), jnp.swapaxes(geo_mod, 0, 1))
        _, mem_li_seq = jax.lax.scan(scan_fn, init_carry, inputs)
        
        mem_li_seq = jnp.swapaxes(mem_li_seq, 0, 1) 
        mem_li_seq = jnp.swapaxes(mem_li_seq, 1, 2) 
        
        # Temporal DP-Pooling
        windows = mem_li_seq.reshape((B, self.hid_ch, 10, 32))
        start_mean = jnp.mean(windows[:, :, :, :16], axis=-1) 
        end_mean = jnp.mean(windows[:, :, :, 16:], axis=-1)   
        
        dp_feat = (end_mean - start_mean).reshape((B, -1)) 
        
        out = self.classifier_drop(dp_feat, deterministic=not train)
        out = nn.elu(self.fc1(out))
        out = self.fc2(out)
        
        return out

# =========================================================
# [3] Optax Training State (Restored CE + Label Smoothing)
# =========================================================
class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng, model, eeg_shape, geo_shape, max_epochs, steps_per_epoch):
    eeg_dummy = jnp.ones(eeg_shape)
    geo_dummy = jnp.ones(geo_shape)
    
    variables = model.init({'params': rng, 'dropout': rng}, eeg_dummy, geo_dummy, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    scheduler = optax.cosine_decay_schedule(init_value=2e-3, decay_steps=max_epochs * steps_per_epoch, alpha=0.01)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.add_decayed_weights(1e-4),
        optax.adamw(learning_rate=scheduler)
    )
    
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

@jax.jit
def train_step(state, x_eeg, x_geo, y, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x_eeg, x_geo, train=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )
        one_hot_y = jax.nn.one_hot(y, 4)
        smooth_labels = optax.smooth_labels(one_hot_y, 0.1)
        # Restore standard Cross Entropy for healthy gradient flow
        loss = optax.softmax_cross_entropy(logits=logits, labels=smooth_labels).mean()
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    
    correct_count = jnp.sum(jnp.argmax(logits, -1) == y)
    return state, loss, correct_count, new_dropout_rng

@jax.jit
def eval_step(state, x_eeg, x_geo, y):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x_eeg, x_geo, train=False
    )
    correct_count = jnp.sum(jnp.argmax(logits, -1) == y)
    return correct_count

def get_batches(x_eeg, x_geo, y, batch_size, shuffle=True, drop_last=False):
    num_samples = len(y)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        if drop_last and end_idx > num_samples:
            break
        batch_idx = indices[start_idx:end_idx]
        yield x_eeg[batch_idx], x_geo[batch_idx], y[batch_idx]

# =========================================================
# [4] Data Pipeline
# =========================================================
EXCLUDE_SUBS = [88, 92, 100, 104]
VALID_SUBS = [s for s in range(1, 110) if s not in EXCLUDE_SUBS]
MOTOR_CHANNELS = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

def process_single_subject_dual(sub):
    try:
        def load_runs(runs):
            raws = [mne.io.read_raw_edf(os.path.join(DATA_DIR_PHYSIONET, f'S{sub:03d}', f'S{sub:03d}R{r:02d}.edf'), preload=True, verbose=False) for r in runs]
            if not raws: return None
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage('standard_1005', on_missing='ignore')
            return raw

        raw_lr = load_runs([4, 8, 12]) 
        raw_f = load_runs([6, 10, 14])  
        if raw_lr is None or raw_f is None: return None
        
        ch_names = raw_lr.ch_names
        motor_idx = [ch_names.index(ch) for ch in MOTOR_CHANNELS if ch in ch_names]
        if not motor_idx: 
            motor_idx = list(range(len(ch_names)))

        def get_epochs(raw):
            evs, ev_id = mne.events_from_annotations(raw, verbose=False)
            t0 = next((v for k, v in ev_id.items() if 'T0' in k), None)
            t1 = next((v for k, v in ev_id.items() if 'T1' in k), None)
            t2 = next((v for k, v in ev_id.items() if 'T2' in k), None)
            ep = mne.Epochs(raw, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            csd = mne.preprocessing.compute_current_source_density(raw.copy(), sphere=(0,0,0,0.095))
            ep_c = mne.Epochs(csd, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            return ep.get_data(), ep_c.get_data(), ep.events[:, 2]

        d_lr, c_lr, y_lr = get_epochs(raw_lr)
        d_f, c_f, y_f = get_epochs(raw_f)
        
        idx_lh, idx_rh, idx_ft = np.where(y_lr == 2)[0], np.where(y_lr == 3)[0], np.where(y_f == 3)[0] 
        idx_rest_lr, idx_rest_f = np.where(y_lr == 1)[0], np.where(y_f == 1)[0]
        
        min_trials = min(len(idx_lh), len(idx_rh), len(idx_ft), len(idx_rest_lr) + len(idx_rest_f))
        if min_trials == 0: return None
        
        d_lh, c_lh = d_lr[idx_lh[:min_trials]], c_lr[idx_lh[:min_trials]]
        d_rh, c_rh = d_lr[idx_rh[:min_trials]], c_lr[idx_rh[:min_trials]]
        d_ft, c_ft = d_f[idx_ft[:min_trials]], c_f[idx_ft[:min_trials]]
        d_rest = np.concatenate([d_lr[idx_rest_lr], d_f[idx_rest_f]])[:min_trials]
        c_rest = np.concatenate([c_lr[idx_rest_lr], c_f[idx_rest_f]])[:min_trials]
        
        raw_eeg = np.concatenate([d_lh, d_rh, d_ft, d_rest])[:, :, 80:400]
        raw_eeg = raw_eeg.transpose(0, 2, 1).astype(np.float32)
        
        xb = np.concatenate([d_lh, d_rh, d_ft, d_rest])[:, motor_idx, :]
        xc = np.concatenate([c_lh, c_rh, c_ft, c_rest])[:, motor_idx, :]
        y = np.concatenate([np.zeros(min_trials), np.ones(min_trials), np.full(min_trials, 2), np.full(min_trials, 3)])
        
        win = 15
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), win, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), win, 3, axis=-1)
        du = savgol_filter(u, win, 3, deriv=1, axis=-1)
        dv = savgol_filter(v, win, 3, deriv=1, axis=-1)
        ddu = savgol_filter(u, win, 3, deriv=2, axis=-1)
        ddv = savgol_filter(v, win, 3, deriv=2, axis=-1)
        
        areal_vel = (0.5 * np.abs(u * dv - v * du))[:, :, 80:400]
        denom = np.clip((du**2 + dv**2)**(1.5), 1e-6, None)
        curvature = (np.clip(np.abs(du * ddv - dv * ddu) / denom, 0, 10))[:, :, 80:400]
        
        areal_vel = np.nan_to_num(areal_vel, nan=0.0, posinf=0.0, neginf=0.0)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        
        vel_mean, vel_std = np.mean(areal_vel, axis=(1, 2), keepdims=True), np.std(areal_vel, axis=(1, 2), keepdims=True) + 1e-8
        curv_mean, curv_std = np.mean(curvature, axis=(1, 2), keepdims=True), np.std(curvature, axis=(1, 2), keepdims=True) + 1e-8
        
        vel_norm = (areal_vel - vel_mean) / vel_std 
        curv_norm = (curvature - curv_mean) / curv_std 
        
        geo_features = np.concatenate([vel_norm, curv_norm], axis=1).astype(np.float32)
        geo_features = geo_features.transpose(0, 2, 1) 
        
        return raw_eeg, geo_features, y
    except Exception as e: 
        return None

if __name__ == "__main__":
    print("INIT: Geo-HR-SNN (1D Depthwise + Dense + HR-Module)")
    
    subject_data = {}
    valid_loaded_subs = []
    
    for sub in VALID_SUBS:
        print(f"LOAD: S{sub:03d}/109...", flush=True)
        data = process_single_subject_dual(sub)
        if data is not None:
            subject_data[sub] = data
            valid_loaded_subs.append(sub)
            
    valid_subs_arr = np.array(valid_loaded_subs)
    print(f"LOAD: Complete. Total subjects: {len(valid_subs_arr)}")
    
    sample_geo_ch = subject_data[valid_subs_arr[0]][1].shape[-1]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results_acc = {}
    rng = jax.random.PRNGKey(42)
    
    fold = 1
    for train_idx, test_idx in kf.split(valid_subs_arr):
        train_subs = valid_subs_arr[train_idx]
        test_subs = valid_subs_arr[test_idx]
        
        print(f"\n============================================================")
        print(f"FOLD {fold}/5 | GLOBAL_TRAIN: {len(train_subs)} | SSTL_TEST: {len(test_subs)}")
        print(f"============================================================")
        
        X_eeg_g = np.concatenate([subject_data[s][0] for s in train_subs])
        X_geo_g = np.concatenate([subject_data[s][1] for s in train_subs])
        y_g = np.concatenate([subject_data[s][2] for s in train_subs])
        
        X_eeg_val = np.concatenate([subject_data[s][0] for s in test_subs])
        X_geo_val = np.concatenate([subject_data[s][1] for s in test_subs])
        y_val = np.concatenate([subject_data[s][2] for s in test_subs])
        
        global_model = Geo_HR_SNN(geo_ch=sample_geo_ch)
        
        batch_size = 256
        steps_per_epoch = len(y_g) // batch_size
        max_epochs = 200
        
        rng, init_rng = jax.random.split(rng)
        state = create_train_state(
            init_rng, global_model, 
            eeg_shape=(1, 320, 64), geo_shape=(1, 320, sample_geo_ch),
            max_epochs=max_epochs, steps_per_epoch=steps_per_epoch
        )
        
        print(f"STAGE 1: GLOBAL PRE-TRAINING (Train: {len(y_g)}, Val: {len(y_val)})")
        rng, dropout_rng = jax.random.split(rng)
        
        best_val_acc = 0.0
        best_global_state = state
        
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            tr_loss, tr_c, tr_t = 0.0, 0, 0
            
            for xb_eeg, xb_geo, yb in get_batches(X_eeg_g, X_geo_g, y_g, batch_size=batch_size, shuffle=True, drop_last=True):
                xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                state, loss, correct_count, dropout_rng = train_step(state, xb_eeg, xb_geo, yb, dropout_rng)
                
                bs = len(yb)
                tr_loss += loss.item() * bs
                tr_c += correct_count.item() 
                tr_t += bs
                
            val_c, val_t = 0, 0
            for xb_eeg, xb_geo, yb in get_batches(X_eeg_val, X_geo_val, y_val, batch_size=256, shuffle=False):
                xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                correct_count = eval_step(state, xb_eeg, xb_geo, yb)
                val_c += correct_count.item()
                val_t += len(yb)
                
            val_acc = 100 * val_c / val_t
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_global_state = state
                mark = "*"
            else:
                mark = ""
            
            epoch_time = time.time() - epoch_start
            print(f"[G-EP {epoch:03d}/{max_epochs}] TLoss: {tr_loss/tr_t:.4f} | TAcc: {100*tr_c/tr_t:>5.1f}% | VAcc: {val_acc:>5.1f}% {mark} | T: {epoch_time:.2f}s")
                
        print(f"\nSTAGE 2: SSTL (Loading Best Global Model: {best_val_acc:.1f}%)")
        
        for sub in test_subs:
            X_eeg_sub, X_geo_sub, y_sub = subject_data[sub]
            tr_idx, ts_idx = train_test_split(np.arange(len(y_sub)), test_size=0.2, stratify=y_sub, random_state=42)
            
            sstl_tx = optax.chain(optax.add_decayed_weights(1e-4), optax.adamw(learning_rate=0.0005))
            sstl_opt_state = sstl_tx.init(best_global_state.params) 
            sstl_state = best_global_state.replace(tx=sstl_tx, opt_state=sstl_opt_state)
            
            sub_best_test_acc = 0.0
            for epoch in range(1, 11): 
                for xb_eeg, xb_geo, yb in get_batches(X_eeg_sub[tr_idx], X_geo_sub[tr_idx], y_sub[tr_idx], batch_size=8, shuffle=True):
                    xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                    sstl_state, _, _, dropout_rng = train_step(sstl_state, xb_eeg, xb_geo, yb, dropout_rng)
                
                ts_c, ts_t = 0, 0
                for xb_eeg, xb_geo, yb in get_batches(X_eeg_sub[ts_idx], X_geo_sub[ts_idx], y_sub[ts_idx], batch_size=32, shuffle=False):
                    xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                    correct_count = eval_step(sstl_state, xb_eeg, xb_geo, yb)
                    ts_c += correct_count.item()
                    ts_t += len(yb)
                
                test_acc = 100 * ts_c / ts_t
                if test_acc > sub_best_test_acc:
                    sub_best_test_acc = test_acc
                    
            results_acc[sub] = sub_best_test_acc
            print(f"  [SSTL] S{sub:03d} -> Best Test Acc: {sub_best_test_acc:.1f}%")
            
        fold += 1

    print(f"\n{'='*50}")
    print(f"FINAL EVAL: Geo-HR-SNN (1D Depthwise + Spatial Dense)")
    print(f"{'='*50}")
    valid_accs = list(results_acc.values())
    print(f"Mean Acc: {np.mean(valid_accs):.2f}%")
    print(f"Max Acc:  {np.max(valid_accs):.2f}%")
    print(f"Min Acc:  {np.min(valid_accs):.2f}%")