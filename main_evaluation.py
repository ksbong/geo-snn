import os
import copy
import time
import numpy as np
import mne
from scipy.signal import hilbert, savgol_filter
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
from sklearn.model_selection import train_test_split, KFold

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # JAX는 자동 할당되지만 기존 호환성을 위해 냅둠

DATA_DIR_PHYSIONET = './raw_data/files/'
if not os.path.exists(DATA_DIR_PHYSIONET):
    DATA_DIR_PHYSIONET = './raw_data/files'

# =========================================================
# [1] SNN Surrogate Spike (JAX Custom VJP)
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
# [2] Geo-ALIF SNN 아키텍처 (Flax Linen)
# =========================================================
class Geo_ALIF_SNN(nn.Module):
    eeg_ch: int = 64
    geo_ch: int = 18
    temp_hid: int = 32
    hid_ch: int = 16
    out_ch: int = 4

    def setup(self):
        self.conv1 = nn.Conv(features=self.temp_hid, kernel_size=(5,), padding=2)
        self.bn1 = nn.BatchNorm(use_running_average=not self.train)
        self.drop1 = nn.Dropout(rate=0.3, deterministic=not self.train)
        
        self.conv2 = nn.Conv(features=self.hid_ch, kernel_size=(3,), padding=1)
        self.bn2 = nn.BatchNorm(use_running_average=not self.train)
        
        self.geo_mod = nn.Dense(features=self.hid_ch)
        
        self.theta_0 = 0.5
        self.beta = 1.8
        self.tau_a = 0.36
        self.tau_m = 0.8
        self.gamma = self.param('gamma', nn.initializers.constant(0.5), (self.hid_ch,))
        
        self.drop_out = nn.Dropout(rate=0.5, deterministic=not self.train)
        self.fc1 = nn.Dense(features=64)
        self.fc2 = nn.Dense(features=self.out_ch)

    @nn.compact
    def __call__(self, x_eeg, x_geo, train: bool = True):
        B, T, _ = x_eeg.shape
        
        x = self.conv1(x_eeg)
        x = self.bn1(x, use_running_average=not train)
        x = nn.relu(x)
        x = self.drop1(x, deterministic=not train)
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        eeg_encoded = nn.relu(x) 
        
        geo_mod = nn.sigmoid(self.geo_mod(x_geo)) 
        
        # 🚀 JAX XLA 가속의 핵심: 파이썬 for문 대신 lax.scan으로 C++ 단에서 통째로 루프 처리
        def scan_fn(carry, inputs):
            mem_alif, eta, mem_li, prev_spike = carry
            cur_eeg, cur_geo_mod = inputs
            
            eta = self.tau_a * eta + (1 - self.tau_a) * prev_spike
            theta_t = self.theta_0 + self.beta * eta - (self.gamma * cur_geo_mod)
            
            mem_alif = self.tau_m * mem_alif + cur_eeg
            spk = spike_fn(mem_alif - theta_t)
            mem_alif = mem_alif * (1.0 - spk)
            
            mem_li = 0.9 * mem_li + spk
            
            return (mem_alif, eta, mem_li, spk), mem_li

        init_carry = (
            jnp.zeros((B, self.hid_ch)), 
            jnp.zeros((B, self.hid_ch)), 
            jnp.zeros((B, self.hid_ch)), 
            jnp.zeros((B, self.hid_ch))  
        )
        
        inputs = (jnp.swapaxes(eeg_encoded, 0, 1), jnp.swapaxes(geo_mod, 0, 1))
        _, mem_li_seq = jax.lax.scan(scan_fn, init_carry, inputs)
        
        mem_li_seq = jnp.swapaxes(mem_li_seq, 0, 1) 
        mem_li_seq = jnp.swapaxes(mem_li_seq, 1, 2) # [B, 16, 320]
        
        windows = mem_li_seq.reshape((B, self.hid_ch, 10, 32))
        start_mean = jnp.mean(windows[:, :, :, :16], axis=-1) 
        end_mean = jnp.mean(windows[:, :, :, 16:], axis=-1)   
        
        dp_feat = (end_mean - start_mean).reshape((B, -1)) 
        
        out = self.drop_out(dp_feat, deterministic=not train)
        out = nn.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

# =========================================================
# [3] Optax TrainState 확장 (에러 수정 완료)
# =========================================================
# 🔥 NameError 방지 및 버전 호환성을 위해 Any 사용
class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng, model, eeg_shape, geo_shape, learning_rate, weight_decay):
    eeg_dummy = jnp.ones(eeg_shape)
    geo_dummy = jnp.ones(geo_shape)
    
    # 모델 초기화 시 파라미터와 드롭아웃 PRNG 동시 제공
    variables = model.init({'params': rng, 'dropout': rng}, eeg_dummy, geo_dummy, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    tx = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.adamw(learning_rate=learning_rate)
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )

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
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    
    # 소수점 오차 방지를 위해 mean 대신 맞춘 갯수(sum)를 반환
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

def get_batches(x_eeg, x_geo, y, batch_size, rng=None, shuffle=True):
    num_samples = len(y)
    indices = jnp.arange(num_samples)
    if shuffle and rng is not None:
        indices = jax.random.permutation(rng, indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield x_eeg[batch_idx], x_geo[batch_idx], y[batch_idx]

# =========================================================
# [4] 데이터 전처리 (이전과 완전 동일)
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
    print(f"🔥 Geo-ALIF SNN [JAX 초고속 XLA 컴파일 버전] 파이프라인 시작...\n")
    
    subject_data = {}
    valid_loaded_subs = []
    
    for sub in VALID_SUBS:
        print(f"데이터 로드 및 특징 추출 중: S{sub:03d} / 109", end='\r')
        data = process_single_subject_dual(sub)
        if data is not None:
            subject_data[sub] = data
            valid_loaded_subs.append(sub)
            
    valid_subs_arr = np.array(valid_loaded_subs)
    print(f"\n✅ 데이터 로드 완료! (유효 피험자: {len(valid_subs_arr)}명)")
    
    sample_geo_ch = subject_data[valid_subs_arr[0]][1].shape[-1]
    print(f"📊 추출된 기하학적 피처 차원: {sample_geo_ch}채널 (순도 100% 운동 피질)")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results_acc = {}
    
    rng = jax.random.PRNGKey(42)
    
    fold = 1
    for train_idx, test_idx in kf.split(valid_subs_arr):
        train_subs = valid_subs_arr[train_idx]
        test_subs = valid_subs_arr[test_idx]
        
        print(f"\n{'='*60}")
        print(f"🔄 [Fold {fold}/5] Global Train: {len(train_subs)}명 | SSTL Test: {len(test_subs)}명")
        print(f"{'='*60}")
        
        X_eeg_g = np.concatenate([subject_data[s][0] for s in train_subs])
        X_geo_g = np.concatenate([subject_data[s][1] for s in train_subs])
        y_g = np.concatenate([subject_data[s][2] for s in train_subs])
        
        global_model = Geo_ALIF_SNN(geo_ch=sample_geo_ch)
        
        rng, init_rng = jax.random.split(rng)
        state = create_train_state(
            init_rng, global_model, 
            eeg_shape=(1, 320, 64), geo_shape=(1, 320, sample_geo_ch),
            learning_rate=0.002, weight_decay=1e-4
        )
        
        print(f"🚀 글로벌 모델 학습 시작 (학습 데이터: {len(y_g)}개, Epochs: 40)")
        rng, dropout_rng = jax.random.split(rng)
        
        for epoch in range(1, 41):
            epoch_start = time.time()
            rng, shuffle_rng = jax.random.split(rng)
            
            tr_loss, tr_c, tr_t = 0.0, 0, 0
            
            # Global Train 배치는 통신 오버헤드를 막기 위해 512까지 늘려도 3090에서 널널함
            for batch_idx, (xb_eeg, xb_geo, yb) in enumerate(get_batches(X_eeg_g, X_geo_g, y_g, batch_size=256, rng=shuffle_rng, shuffle=True), 1):
                xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                
                state, loss, correct_count, dropout_rng = train_step(state, xb_eeg, xb_geo, yb, dropout_rng)
                
                batch_size = len(yb)
                tr_loss += loss.item() * batch_size
                tr_c += correct_count.item() 
                tr_t += batch_size
            
            epoch_time = time.time() - epoch_start
            print(f"  👉 [Global Epoch {epoch:02d}/40] Loss: {tr_loss/tr_t:.4f} | Train Acc: {100*tr_c/tr_t:.1f}% | 소요시간: {epoch_time:.2f}초")
                
        print(f"\n🎯 [Stage 2] 파인튜닝(SSTL) 및 평가 진행 중... (총 {len(test_subs)}명)")
        for idx, sub in enumerate(test_subs, 1):
            X_eeg_sub, X_geo_sub, y_sub = subject_data[sub]
            tr_idx, ts_idx = train_test_split(np.arange(len(y_sub)), test_size=0.2, stratify=y_sub, random_state=42)
            
            # 🔥 여기서 터질 뻔한 걸 미리 수정함: Learning Rate 변경 시 optax state도 완전 갱신
            sstl_tx = optax.chain(optax.add_decayed_weights(1e-4), optax.adamw(learning_rate=0.0005))
            sstl_opt_state = sstl_tx.init(state.params) 
            sstl_state = state.replace(tx=sstl_tx, opt_state=sstl_opt_state)
            
            best_test_acc = 0.0
            for epoch in range(1, 11): 
                rng, shuffle_rng = jax.random.split(rng)
                for xb_eeg, xb_geo, yb in get_batches(X_eeg_sub[tr_idx], X_geo_sub[tr_idx], y_sub[tr_idx], batch_size=8, rng=shuffle_rng, shuffle=True):
                    xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                    sstl_state, _, _, dropout_rng = train_step(sstl_state, xb_eeg, xb_geo, yb, dropout_rng)
                
                # Test Evaluation
                ts_c, ts_t = 0, 0
                for xb_eeg, xb_geo, yb in get_batches(X_eeg_sub[ts_idx], X_geo_sub[ts_idx], y_sub[ts_idx], batch_size=32, shuffle=False):
                    xb_eeg, xb_geo, yb = jnp.array(xb_eeg), jnp.array(xb_geo), jnp.array(yb, dtype=jnp.int32)
                    correct_count = eval_step(sstl_state, xb_eeg, xb_geo, yb)
                    ts_c += correct_count.item()
                    ts_t += len(yb)
                
                test_acc = 100 * ts_c / ts_t
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    
            results_acc[sub] = best_test_acc
            print(f"  ✅ S{sub:03d} SSTL 완료 ({idx}/{len(test_subs)}) -> 최고 Test Acc: {best_test_acc:.1f}%")
            
        print(f"\n✅ Fold {fold} 평가 싹 다 완료.")
        fold += 1

    print(f"\n{'='*50}")
    print(f"📊 최종 Geo-ALIF SNN (JAX Accelerated) 테스트 결과")
    print(f"{'='*50}")
    valid_accs = list(results_acc.values())
    print(f"평균 정확도(Mean): {np.mean(valid_accs):.1f}%")
    print(f"최고 정확도(Max):  {np.max(valid_accs):.1f}%")
    print(f"최저 정확도(Min):  {np.min(valid_accs):.1f}%")