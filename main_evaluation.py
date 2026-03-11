import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import mne
from scipy.signal import hilbert, savgol_filter
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR_PHYSIONET = './raw_data/files'

# [1] SNN 모델 (억제망 초기값 완화)
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1.0 + 10.0 * torch.abs(input)) ** 2

spike_fn = SurrogateSpike.apply

class PhysioNetGeoLIF_4Class(nn.Module):
    def __init__(self, num_eeg_channels=64, num_classes=4, leak=0.9, base_thresh=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.leak = leak
        self.base_thresh = base_thresh
        self.spatial_weights = nn.Linear(num_eeg_channels, num_classes, bias=False)
        # 초기 억제를 -0.1로 완화해서 학습 초기 스파이크 흐름 확보
        init_inhibition = torch.tensor([
            [ 1.0, -0.1, -0.1, -0.1], [ -0.1, 1.0, -0.1, -0.1], 
            [ -0.1, -0.1, 1.0, -0.1], [ -0.1, -0.1, -0.1, 1.0]
        ])
        self.lateral_weights = nn.Parameter(init_inhibition, requires_grad=True)
        self.tda_to_thresh = nn.Linear(50, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        batch_size, time_steps, _ = kin_spikes_seq.shape
        mem = torch.zeros(batch_size, self.num_classes, device=kin_spikes_seq.device)
        tda_mod = torch.sigmoid(self.tda_to_thresh(tda_features)) * 0.3 # 영향력 살짝 조절
        dynamic_thresh = self.base_thresh + tda_mod 
        spike_records = []
        for t in range(time_steps):
            cur_input = torch.matmul(self.spatial_weights(kin_spikes_seq[:, t, :]), self.lateral_weights)
            mem = self.leak * mem + cur_input
            spikes_out = spike_fn(mem - dynamic_thresh)
            mem = mem * (1.0 - spikes_out)
            spike_records.append(spikes_out)
        return torch.stack(spike_records, dim=1)

# [2] 병렬 데이터 엔진 (4-Class 라벨링 수정)
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    paths = [os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf"),
             os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")]
    for p in paths:
        if os.path.exists(p): return p
    return None

def process_single_subject(sub):
    try:
        def load_group(runs, expected_events):
            raws = []
            for r in runs:
                path = get_local_file_path(sub, r)
                if path: raws.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
            if not raws: return None, None, None, None
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            mapping = {ch: ch.replace('.', '').upper().replace('Z', 'z') for ch in raw.ch_names}
            raw.rename_channels(mapping); raw.set_montage('standard_1005', on_missing='ignore')
            try: csd = mne.preprocessing.compute_current_source_density(raw.copy())
            except: csd = raw.copy()
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            actual_event_map = {k: v for k, v in event_id.items() if any(target in k for target in expected_events)}
            if not actual_event_map: return None, None, None, None
            ep_b = mne.Epochs(raw, events, event_id=actual_event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c = mne.Epochs(csd, events, event_id=actual_event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            return ep_b.get_data(), ep_c.get_data(), ep_b.events[:, 2], ep_b.ch_names, event_id

        # 4-Class 구성: LH(0), RH(1), Feet(2), Rest(3) 
        res_lr = load_group([4, 8, 12], ['T0', 'T1', 'T2'])
        res_f = load_group([6, 10, 14], ['T0', 'T2'])
        if res_lr is None or res_f is None: return None
        
        # 라벨 정규화 로직 (T0=Rest, T1=LH, T2=RH/Feet)
        def map_labels(y, ev_id, group='lr'):
            y_new = np.zeros_like(y)
            t0_val = next(v for k, v in ev_id.items() if 'T0' in k)
            t1_val = next(v for k, v in ev_id.items() if 'T1' in k)
            t2_val = next(v for k, v in ev_id.items() if 'T2' in k)
            if group == 'lr':
                y_new[y == t1_val] = 0 # LH
                y_new[y == t2_val] = 1 # RH
                y_new[y == t0_val] = 3 # Rest
            else:
                y_new[y == t2_val] = 2 # Feet
                y_new[y == t0_val] = 3 # Rest
            return y_new

        y_lr = map_labels(res_lr[2], res_lr[4], 'lr')
        y_f = map_labels(res_f[2], res_f[4], 'f')

        xb = np.concatenate([res_lr[0], res_f[0]])
        xc = np.concatenate([res_lr[1], res_f[1]])
        y = np.concatenate([y_lr, y_f])

        # 기하학 피처 (Percentile 80%로 완화)
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), 51, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), 51, 3, axis=-1)
        du, dv = np.gradient(u, axis=-1), np.gradient(v, axis=-1)
        geo = (0.5 * np.abs(u * dv - v * du) * np.clip(np.abs(u*dv - v*du)/(np.clip((du**2 + dv**2)**1.5, 1e-6, None)), 0, 10))[:, :, 80:400]
        kin_spikes = (geo > np.percentile(geo, 80)).astype(np.float32).transpose(0, 2, 1) # 더 많은 정보 유입 [cite: 409]
        
        ch_upper = [c.upper() for c in res_lr[3]]
        idx = [ch_upper.index(n) for n in ['C3', 'CZ', 'C4'] if n in ch_upper]
        VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
        PL = PersistenceLandscape(n_bins=50)
        tda = PL.fit_transform(VR.fit_transform(np.stack([u[:,idx[0],:], v[:,idx[0],:], u[:,idx[1],:], v[:,idx[1],:], u[:,idx[2],:], v[:,idx[2],:]], axis=-1))).reshape(len(y), -1).astype(np.float32)

        return kin_spikes, tda, y
    except: return None

def get_full_data_parallel(subjects):
    print(f"🔥 {len(subjects)}명 데이터 병렬 전처리 시작...")
    with Pool(cpu_count()) as p:
        results = [r for r in list(tqdm(p.imap(process_single_subject, subjects), total=len(subjects), desc="전처리")) if r is not None]
    return torch.tensor(np.concatenate([r[0] for r in results])), \
           torch.tensor(np.concatenate([r[1] for r in results])), \
           torch.tensor(np.concatenate([r[2] for r in results]), dtype=torch.long)

if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    Xk, Xt, y = get_full_data_parallel(VALID_SUBS)
    Xk_tr, Xk_ts, Xt_tr, Xt_ts, y_tr, y_ts = train_test_split(Xk, Xt, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(Xk_tr, Xt_tr, y_tr), batch_size=64, shuffle=True) # 배치 사이즈 소폭 축소
    test_loader = DataLoader(TensorDataset(Xk_ts, Xt_ts, y_ts), batch_size=64, shuffle=False)

    model = PhysioNetGeoLIF_4Class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4) # 안정적인 학습률 [cite: 340]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 학습 재시작! 샘플 수: {len(y)}")
    pbar = tqdm(range(100), desc="에포크")
    for epoch in pbar:
        model.train()
        tr_c, tr_t = 0, 0
        for kb, tb, yb in train_loader:
            kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(kb, tb).sum(dim=1)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            tr_c += out.max(1)[1].eq(yb).sum().item(); tr_t += yb.size(0)
        
        scheduler.step()
        model.eval()
        ts_c, ts_t = 0, 0
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                ts_c += model(kb, tb).sum(dim=1).max(1)[1].eq(yb).sum().item(); ts_t += yb.size(0)
        
        pbar.set_postfix(Train=f"{100*tr_c/tr_t:.1f}%", Test=f"{100*ts_c/ts_t:.1f}%")