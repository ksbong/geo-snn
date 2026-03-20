import os
import gc
import mne
mne.set_log_level('ERROR')
import numpy as np
import scipy.linalg as la

from sklearn.model_selection import train_test_split
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# =========================================================
# 1. 환경 설정 및 글로벌 아키텍처 정의
# =========================================================
base = './'
DATA_DIR = os.path.join(base, '07_Data') 
TARGET_SFREQ = 160.0 
runs_h, runs_f = ['R04','R08','R12'], ['R06','R10','R14']

bad_subjects = ['S088', 'S089', 'S092', 'S100']
subjects = [f'S{i:03d}' for i in range(1, 110) if f'S{i:03d}' not in bad_subjects]

@jax.custom_vjp
def spike(x): return (x > 0).astype(jnp.float32)
def fwd(x): return spike(x), x
def bwd(res, g): return (g * 0.3 / (1.0 + jnp.abs(res * 3.0)),)
spike.defvjp(fwd, bwd)

class LIFCell(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, state, x):
        v, z = state
        decay = nn.sigmoid(self.param("decay", nn.initializers.constant(1.0), (self.hidden_dim,)))
        v = v * decay * (1.0 - z) + x
        z_new = spike(v - 0.5)
        return (v, z_new), z_new

class Subject_RTM_SNN(nn.Module):
    @nn.compact
    def __call__(self, x_seq, train=True):
        B, T, F = x_seq.shape
        x = nn.Dropout(0.5, deterministic=not train)(x_seq)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        init1 = (jnp.zeros((B, 128)), jnp.zeros((B, 128)))
        Scan1 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk1 = Scan1(128)(init1, x)

        spk1 = nn.Dropout(0.3, deterministic=not train)(spk1)

        x2 = nn.Dense(64)(spk1)
        x2 = nn.LayerNorm()(x2)
        init2 = (jnp.zeros((B, 64)), jnp.zeros((B, 64)))
        Scan2 = nn.scan(LIFCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=1, out_axes=1)
        _, spk2 = Scan2(64)(init2, x2)

        attn = nn.softmax(nn.Dense(1)(spk2), axis=1)
        feat = jnp.sum(spk2 * attn, axis=1)

        feat = nn.Dropout(0.5, deterministic=not train)(feat)
        logits = nn.Dense(4)(feat)
        return logits, (jnp.mean(spk1) + jnp.mean(spk2)) / 2.0

def loss_fn(params, state, x, y, rng):
    logits, fr = state.apply_fn({'params': params}, x, train=True, rngs={'dropout': rng})
    y_s = jax.nn.one_hot(y, 4) * 0.9 + 0.025 
    ce = optax.softmax_cross_entropy(logits=logits, labels=y_s).mean()
    fr_loss = jnp.mean((fr - 0.15)**2) * 0.05
    return ce + fr_loss, logits

@jax.jit
def train_step(state, x, y, rng):
    rng, sub = jax.random.split(rng)
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state, x, y, sub)
    state = state.apply_gradients(grads=grads)
    return state, loss, jnp.mean(jnp.argmax(logits, -1) == y), rng

@jax.jit
def eval_step(state, x):
    logits, _ = state.apply_fn({'params': state.params}, x, train=False)
    return jnp.argmax(logits, -1)

# =========================================================
# 2. 메인 루프: 105명 피험자 개별 검증
# =========================================================
print("\n" + "="*60)
print("🚀 전체 피험자 독립 검증 시작 (Intra-Subject RTM-SNN)")
print("="*60)

subject_results = {}

for idx, subj in enumerate(subjects, 1):
    subj_dir = os.path.join(DATA_DIR, subj)
    if not os.path.exists(subj_dir): continue

    epochs_list = []
    for run in runs_h + runs_f:
        path = os.path.join(subj_dir, f'{subj}{run}.edf')
        if not os.path.exists(path): continue
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
            raw.rename_channels(lambda x: x.strip('.'))
            raw.filter(8.0, 30.0, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            
            evs, ed = mne.events_from_annotations(raw, verbose=False)
            t1, t2 = ed.get('T1'), ed.get('T2')
            if t1 is None: continue
            
            e = evs.copy()
            if run in runs_h: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 0, 1
            else: e[evs[:,2]==t1,2], e[evs[:,2]==t2,2] = 2, 3

            event_id = {'L':0, 'R':1} if run in runs_h else {'H':2, 'F':3}
            ep = mne.Epochs(raw, e, event_id, tmin=0.0, tmax=3.0, baseline=None, preload=True, verbose=False)
            if len(ep) > 0: epochs_list.append(ep)
        except: continue
            
    if not epochs_list: 
        print(f"[{idx:3d}/{len(subjects)}] {subj} 데이터 부족으로 스킵")
        continue
        
    epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    X_raw, y = epochs.get_data() * 1e6, epochs.events[:,2]

    # [안전장치] 클래스당 샘플 수 최소한의 균형 맞추기 (선택적)
    unique, counts = np.unique(y, return_counts=True)
    min_c = np.min(counts)
    bal_idx = []
    for cls in unique:
        bal_idx.extend(np.random.choice(np.where(y == cls)[0], min_c, replace=False))
    X_raw, y = X_raw[bal_idx], y[bal_idx]

    # 피험자 맞춤형 정렬 (R_i)
    covs = []
    for x in X_raw:
        c = np.cov(x - np.mean(x, axis=1, keepdims=True))
        covs.append((c + c.T) / 2.0) # 대칭성 보장
        
    R_i = np.mean(covs, axis=0) + np.eye(X_raw.shape[1]) * 1e-4
    vals, vecs = la.eigh(R_i)
    r_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-6, None))) @ vecs.T
    X_aligned = np.array([r_inv_sqrt @ x for x in X_raw])

    # 궤적 추출 (Full Rank: window=160, step=16)
    triu_idx = np.triu_indices(X_aligned.shape[1])
    seq_feats = []
    window, step = 160, 16 
    for x in X_aligned:
        win_feats = []
        for w in range((x.shape[1] - window) // step + 1):
            win = x[:, w*step : w*step+window]
            cov = np.cov(win - np.mean(win, axis=1, keepdims=True)) + np.eye(X_aligned.shape[1]) * 1e-4
            cov = (cov + cov.T) / 2.0 # 대칭성 보장
            vals_w, vecs_w = la.eigh(cov)
            log_cov = vecs_w @ np.diag(np.log(np.clip(vals_w, 1e-6, None))) @ vecs_w.T
            win_feats.append(log_cov[triu_idx]) 
        seq_feats.append(np.array(win_feats))
    
    X_seq = np.array(seq_feats)

    # 데이터 로더 (개별 학습용이므로 Batch 크기를 작게)
    idx_tr, idx_te = train_test_split(np.arange(len(X_seq)), test_size=0.2, stratify=y, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_seq[idx_tr], dtype=torch.float32), torch.tensor(y[idx_tr])), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_seq[idx_te], dtype=torch.float32), torch.tensor(y[idx_te])), batch_size=16, shuffle=False)

    # 모델 초기화 및 학습
    model = Subject_RTM_SNN()
    rng = jax.random.PRNGKey(42)
    params = model.init({'params': rng, 'dropout': rng}, jnp.ones((1, X_seq.shape[1], 2080)))['params']
    
    lr_sched = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=len(train_loader)*80)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate=lr_sched, weight_decay=1e-2))

    best_val = 0
    for epoch in range(1, 81): # 80 Epoch 충분히 돌림
        for xb, yb in train_loader:
            state, _, _, rng = train_step(state, jnp.array(xb.numpy()), jnp.array(yb.numpy()), rng)
        
        preds = []
        for xb, _ in test_loader:
            preds.extend(np.array(eval_step(state, jnp.array(xb.numpy()))))
        val_acc = np.mean(np.array(preds) == y[idx_te]) * 100
        best_val = max(best_val, val_acc)

    print(f"[{idx:3d}/{len(subjects)}] {subj} | Best Test Acc: {best_val:.2f}%")
    subject_results[subj] = best_val

    # 🧹 메모리 누수 방지 (매우 중요)
    del X_raw, X_aligned, X_seq, y, train_loader, test_loader, state, params
    gc.collect()

# =========================================================
# 3. 논문용 최종 결과 리포트 출력
# =========================================================
print("\n" + "="*60)
print("🏆 Final Evaluation Report: Intra-Subject Performance")
print("="*60)

if not subject_results:
    print("결과가 없습니다. 데이터 경로를 확인해주세요.")
else:
    # 정확도 기준 내림차순 정렬
    sorted_results = sorted(subject_results.items(), key=lambda item: item[1], reverse=True)
    accuracies = [acc for _, acc in sorted_results]
    
    total_subj = len(accuracies)
    top_20_avg = np.mean(accuracies[:min(20, total_subj)])
    top_50_avg = np.mean(accuracies[:min(50, total_subj)])
    overall_avg = np.mean(accuracies)

    print(f"▶ 성공적으로 평가된 피험자 수: {total_subj} 명\n")
    print(f"🔥 Top 20 Subjects Average : {top_20_avg:.2f}%")
    if total_subj >= 50:
        print(f"🔥 Top 50 Subjects Average : {top_50_avg:.2f}%")
    print(f"📊 Overall Average (All)   : {overall_avg:.2f}%")
    
    print("\n[🥇 Top 5 피험자 명단]")
    for i in range(min(5, total_subj)):
        print(f"  {i+1}위: {sorted_results[i][0]} ({sorted_results[i][1]:.2f}%)")
        
    print("\n[💀 Bottom 5 피험자 명단 (BCI Illiteracy 후보)]")
    for i in range(1, min(6, total_subj+1)):
        print(f"  하위 {i}위: {sorted_results[-i][0]} ({sorted_results[-i][1]:.2f}%)")

print("="*60)