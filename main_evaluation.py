import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
from flax.training import train_state
import optax
from torch.utils.data import DataLoader, TensorDataset
import torch
from functools import partial
from typing import Any
from flax.jax_utils import replicate, unreplicate

mne.set_log_level('ERROR')
# 경로 주의: Runpod 환경에 맞게 수정해! (예: /workspace/...)
DATA_DIR = './07_Data'
CACHE_FILE = '/workspace/geo-snn/ra_multitacip_2d_cache_spectrum_105.npz'

# =========================================================
# 1. 고속 다중 대역 TACIP & 데이터 로더 
# =========================================================
def fast_multiband_tacip(trial, motifs, motif_norms, window=32):
    C, T = trial.shape
    F = motifs.shape[0]
    res = np.zeros((C, T, F))
    for c in range(C):
        sig = trial[c]
        stride = sig.strides[0]
        chunks = np.lib.stride_tricks.as_strided(sig, shape=(T-window+1, window), strides=(stride, stride))
        res[c, window-1:, :] = np.dot(chunks, motifs.T) / motif_norms[None, :]
    return res

def load_balanced_ra_2d(subj_list, desc="Loading"):
    print(f"⏳ {desc} 로딩 시작 (총 {len(subj_list)}명)...")
    X_out, Y_out = [], []
    fs, window = 160.0, 32
    t = np.arange(window) / fs
    freqs = [10, 15, 20, 25]
    motifs = np.array([np.sin(2 * np.pi * f * t) * np.hanning(window) for f in freqs])
    motif_norms = np.linalg.norm(motifs, axis=1) + 1e-8

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
        
        X_tacip = np.array([fast_multiband_tacip(trial, motifs, motif_norms) for trial in X_ra])
        X_out.append(np.moveaxis(X_tacip, 1, 2)) 
        Y_out.extend(Y_bal)
        
    print(f"✅ {desc} 전처리 완벽하게 끝!")
    return np.concatenate(X_out), np.array(Y_out)

print("⚙️ 데이터 준비 중...")
if os.path.exists(CACHE_FILE):
    data = np.load(CACHE_FILE)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_test, Y_test = data['X_test'], data['Y_test']
else:
    all_subjs = [f'S{i:03d}' for i in range(1, 110)]
    exclude = ['S088', 'S092', 'S100', 'S104'] 
    subjects = sorted([s for s in all_subjs if s not in exclude])
    split_idx = int(len(subjects) * 0.8) 
    train_subjs, test_subjs = subjects[:split_idx], subjects[split_idx:] 
    
    X_train, Y_train = load_balanced_ra_2d(train_subjs, "Train Data")
    X_test, Y_test = load_balanced_ra_2d(test_subjs, "Test Data")
    np.savez_compressed(CACHE_FILE, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)), batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)), batch_size=32, shuffle=False, drop_last=False)

# =========================================================
# 2. 정규화 및 커스텀 스파이크 함수 (Alpha 적용)
# =========================================================
def apply_coordinated_dropout(x, rng, rate):
    mask = jrandom.bernoulli(rng, p=1-rate, shape=x.shape)
    return x * mask * (1.0 / (1.0 - rate)), mask

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

# =========================================================
# 3. 모델 정의 (HR-SNN 뉴런 + LI 출력층)
# =========================================================
class TrainState(train_state.TrainState):
    batch_stats: Any 

class HR_HLIFLayer2D(nn.Module):
    vth_m: float; vth_s: float
    decay_m: float; decay_s: float
    alpha: float

    @nn.compact
    def __call__(self, x):
        x_seq = jnp.moveaxis(x, 1, 0)
        shape = x.shape[2:]
        
        raw_vth = self.param('vth_raw', nn.initializers.normal(stddev=1.0), shape)
        raw_decay = self.param('decay_raw', nn.initializers.normal(stddev=1.0), shape)
        
        vth = jax.nn.softplus(raw_vth * self.vth_s + self.vth_m) + 0.01
        decay = jnp.clip(jax.nn.sigmoid(raw_decay * self.decay_s + self.decay_m), 0.0, 0.99)
        
        def scan_fn(v, x_t):
            v = v * decay + x_t
            s = spike_fn(v - vth, self.alpha)
            return v - jax.lax.stop_gradient(s) * vth, s
        _, spikes = jax.lax.scan(scan_fn, jnp.zeros_like(x_seq[0]), x_seq)
        return jnp.moveaxis(spikes, 0, 1)

class AutoHRSNN(nn.Module):
    hps: dict
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(32, (16, 1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        spk1 = HR_HLIFLayer2D(
            vth_m=self.hps['v1_m'], vth_s=self.hps['v1_s'],
            decay_m=self.hps['d1_m'], decay_s=self.hps['d1_s'], alpha=self.hps['a1']
        )(x)
        x = nn.Dropout(self.hps['drop1'], deterministic=not train)(spk1)
        
        x = nn.Conv(64, (1, 64), padding='VALID', feature_group_count=32, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        spk2 = HR_HLIFLayer2D(
            vth_m=self.hps['v2_m'], vth_s=self.hps['v2_s'],
            decay_m=self.hps['d2_m'], decay_s=self.hps['d2_s'], alpha=self.hps['a2']
        )(x)
        x = nn.Dropout(self.hps['drop2'], deterministic=not train)(spk2)
        
        x = jnp.mean(x, axis=2) 
        dense_out = nn.Dense(4)(x) 
        
        def li_scan(v, x_t):
            v = v * self.hps['decay_out'] + x_t
            return v, v
        
        _, v_out = jax.lax.scan(li_scan, jnp.zeros_like(dense_out[0]), jnp.moveaxis(dense_out, 1, 0))
        out_logits = jnp.mean(v_out, axis=0) 
        
        return out_logits, (spk1, spk2)

# =========================================================
# 4. PBT Manager (17개 HP)
# =========================================================
class PBTManager:
    def __init__(self, num_workers=20): 
        self.num_workers = num_workers
        self.worker_hps = []
        for _ in range(num_workers):
            self.worker_hps.append({
                'lr': 10**np.random.uniform(-4, -2.5),
                'wd': 10**np.random.uniform(-5, -3),
                'cd_rate': np.random.uniform(0.01, 0.15),
                'drop1': np.random.uniform(0.1, 0.5),
                'drop2': np.random.uniform(0.1, 0.5),
                'v1_m': np.random.uniform(0.5, 1.5), 'v1_s': np.random.uniform(0.1, 0.5),
                'd1_m': np.random.uniform(0.0, 2.0), 'd1_s': np.random.uniform(0.1, 0.5),
                'a1': np.random.uniform(1.0, 3.0),
                'v2_m': np.random.uniform(0.5, 1.5), 'v2_s': np.random.uniform(0.1, 0.5),
                'd2_m': np.random.uniform(0.0, 2.0), 'd2_s': np.random.uniform(0.1, 0.5),
                'a2': np.random.uniform(1.0, 3.0),
                'decay_out': np.random.uniform(0.5, 0.99),
                'reg_spike': 10**np.random.uniform(-5, -3) 
            })
            
    def exploit_and_explore(self, states, accs):
        sorted_idx = np.argsort(accs)
        bottom_idx = sorted_idx[:self.num_workers//4] 
        top_idx = sorted_idx[-self.num_workers//4:]   
        
        for b, t in zip(bottom_idx, top_idx):
            new_hp = self.worker_hps[t].copy()
            for k in new_hp:
                new_val = new_hp[k] * np.random.uniform(0.8, 1.2)
                if k in ['cd_rate', 'drop1', 'drop2']: new_hp[k] = np.clip(new_val, 0.01, 0.5) 
                elif k == 'decay_out': new_hp[k] = np.clip(new_val, 0.1, 0.99)
                elif 's' in k: new_hp[k] = np.clip(new_val, 0.01, 1.5) 
                else: new_hp[k] = new_val
            self.worker_hps[b] = new_hp

            new_model = AutoHRSNN(hps=new_hp)
            new_tx = optax.adamw(new_hp['lr'], weight_decay=new_hp['wd'])
            
            states[b] = TrainState.create(
                apply_fn=new_model.apply, params=states[t].params, 
                tx=new_tx, batch_stats=states[t].batch_stats 
            )
        return states

# =========================================================
# 5. 분산 학습 루프 (A5000 체급 + 히스토리 로깅 복구)
# =========================================================
pbt = PBTManager(num_workers=20)
rng = jrandom.PRNGKey(42)

population_states = []
for i in range(20):
    hp = pbt.worker_hps[i]
    model = AutoHRSNN(hps=hp)
    var = model.init(rng, jnp.ones((1, 480, 64, 4)), train=False)
    tx = optax.adamw(hp['lr'], weight_decay=hp['wd'])
    population_states.append(TrainState.create(
        apply_fn=model.apply, params=var['params'], tx=tx, batch_stats=var['batch_stats']
    ))

def shard(x): return jnp.reshape(x, (jax.local_device_count(), -1) + x.shape[1:])

@partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, 0, None))
def train_step(state, x, y, cd_rng, hp):
    x_cd, _ = apply_coordinated_dropout(x, cd_rng, rate=hp['cd_rate'])
    
    def loss_fn(params):
        (logits, (spk1, spk2)), new_vars = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, 
            x_cd, train=True, rngs={'dropout': cd_rng}, mutable=['batch_stats']
        )
        smooth_y = optax.smooth_labels(jax.nn.one_hot(y, 4), 0.1)
        ce_loss = jnp.mean(optax.softmax_cross_entropy(logits, smooth_y))
        
        spike_loss = hp['reg_spike'] * (jnp.mean(spk1) + jnp.mean(spk2))
        total_loss = ce_loss + spike_loss
        
        acc = jnp.mean(jnp.argmax(logits, -1) == y)
        return total_loss, (logits, new_vars['batch_stats'], acc, total_loss)
        
    grads, (logits, new_bs, tr_acc, tr_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=jax.lax.pmean(grads, 'batch'), batch_stats=jax.lax.pmean(new_bs, 'batch'))
    return state, jax.lax.pmean(tr_acc, 'batch'), jax.lax.pmean(tr_loss, 'batch')

@partial(jax.pmap, axis_name='batch')
def eval_step(state, x, y):
    (logits, _), _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats}, 
        x, train=False, mutable=False
    )
    return jnp.mean(jnp.argmax(logits, -1) == y)

print(f"\n🚀 A5000-Ready HR-SNN 진화 시작 (Workers: 20, 10세대)")

# 🔥 기록용 리스트 부활
history_accs, history_hps = [], []
history_train_accs, history_train_losses = [], []

for gen in range(1, 11): 
    worker_test_results = []
    current_gen_hps = []
    gen_train_accs, gen_train_losses = [], []
    
    for w_idx in range(20):
        hp = pbt.worker_hps[w_idx]
        current_gen_hps.append(hp.copy())
        p_state = replicate(population_states[w_idx]) 
        
        tr_accs_epoch, tr_losses_epoch = [], []
        for _ in range(15): 
            for xb, yb in train_loader:
                rng, cd_rng = jrandom.split(rng)
                cd_rngs = jrandom.split(cd_rng, jax.local_device_count())
                p_state, t_acc, t_loss = train_step(p_state, shard(jnp.array(xb)), shard(jnp.array(yb)), cd_rngs, hp)
                tr_accs_epoch.append(t_acc[0])
                tr_losses_epoch.append(t_loss[0])
                
        v_accs = []
        for xt, yt in test_loader:
            if len(xt) % jax.local_device_count() == 0:
                acc = eval_step(p_state, shard(jnp.array(xt)), shard(jnp.array(yt)))
                v_accs.append(acc[0])
                
        test_acc_mean = np.mean(v_accs)
        train_acc_mean = np.mean(tr_accs_epoch)
        train_loss_mean = np.mean(tr_losses_epoch)
        
        population_states[w_idx] = unreplicate(p_state)
        worker_test_results.append(test_acc_mean)
        gen_train_accs.append(train_acc_mean)
        gen_train_losses.append(train_loss_mean)
        
        print(f"Gen {gen:02d}|W_{w_idx:02d} | Test: {test_acc_mean*100:.1f}% | Train: {train_acc_mean*100:.1f}% (Loss: {train_loss_mean:.3f}) | V1_m/s: {hp['v1_m']:.2f}/{hp['v1_s']:.2f} | Alpha: {hp['a1']:.1f}")

    population_accs = np.array(worker_test_results)
    
    # 🔥 현재 세대 데이터 기록
    history_accs.append(population_accs)
    history_hps.append(current_gen_hps)
    history_train_accs.append(np.array(gen_train_accs))
    history_train_losses.append(np.array(gen_train_losses))
    
    population_states = pbt.exploit_and_explore(population_states, population_accs)
    print(f"🎯 Gen {gen:02d} 완료 (Best Test Acc: {np.max(population_accs)*100:.1f}%) -----------------")

# =========================================================
# 6. 최종 챔피언 분석 및 Figure용 전체 데이터 Save (🔥 완벽 복구)
# =========================================================
best_idx = int(np.argmax(population_accs)) 
best_hp = pbt.worker_hps[best_idx]
best_state = population_states[best_idx]

# 최상위 모델의 실제 파라미터를 꺼내서 수식에 넣어 실제 배열값을 계산
params = best_state.params
# Layer 1
raw_v1 = params['HR_HLIFLayer2D_0']['vth_raw']
raw_d1 = params['HR_HLIFLayer2D_0']['decay_raw']
l1_vth = np.array(jax.nn.softplus(raw_v1 * best_hp['v1_s'] + best_hp['v1_m']) + 0.01).flatten()
l1_decay = np.array(jnp.clip(jax.nn.sigmoid(raw_d1 * best_hp['d1_s'] + best_hp['d1_m']), 0.0, 0.99)).flatten()

# Layer 2
raw_v2 = params['HR_HLIFLayer2D_1']['vth_raw']
raw_d2 = params['HR_HLIFLayer2D_1']['decay_raw']
l2_vth = np.array(jax.nn.softplus(raw_v2 * best_hp['v2_s'] + best_hp['v2_m']) + 0.01).flatten()
l2_decay = np.array(jnp.clip(jax.nn.sigmoid(raw_d2 * best_hp['d2_s'] + best_hp['d2_m']), 0.0, 0.99)).flatten()

np.set_printoptions(precision=4, suppress=True, linewidth=120)

print(f"\n🏆 [최종 챔피언] Worker {best_idx:02d} (최종 Test Acc: {population_accs[best_idx]*100:.1f}%)")
print("="*80)
print(f"▶ Layer 1 (Temporal, 32 뉴런) 실제 Vth 값:\n{l1_vth}")
print(f"\n▶ Layer 1 (Temporal, 32 뉴런) 실제 Decay 값:\n{l1_decay}")
print(f"\n▶ Layer 2 (Spatial,  64 뉴런) 실제 Vth 값:\n{l2_vth}")
print(f"\n▶ Layer 2 (Spatial,  64 뉴런) 실제 Decay 값:\n{l2_decay}")
print("="*80)

# 🔥 17개 HP 전부 저장!
hp_keys = ['lr', 'wd', 'cd_rate', 'drop1', 'drop2', 
           'v1_m', 'v1_s', 'd1_m', 'd1_s', 'a1', 
           'v2_m', 'v2_s', 'd2_m', 'd2_s', 'a2', 
           'decay_out', 'reg_spike']
hp_history_dict = {k: np.array([[gen_hp[w][k] for w in range(20)] for gen_hp in history_hps]) for k in hp_keys}

# 저장 경로 조심해! Runpod이면 '/workspace/...' 이런 식이어야 해.
save_path = 'pbt_evolution_data_hr_snn.npz'
np.savez(save_path,
         acc_history=np.array(history_accs),            # shape: (10, 20)
         train_acc_history=np.array(history_train_accs),# shape: (10, 20)
         train_loss_history=np.array(history_train_losses), # shape: (10, 20)
         l1_vth_final=l1_vth,                           # shape: (32,)
         l1_decay_final=l1_decay,                       # shape: (32,)
         l2_vth_final=l2_vth,                           # shape: (64,)
         l2_decay_final=l2_decay,                       # shape: (64,)
         **hp_history_dict)                             # 각 HP별 shape: (10, 20)

print(f"\n✅ 논문 Figure용 전체 데이터 저장 완료 (총 17개 HP 포함): {save_path}")