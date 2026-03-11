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

# =========================================================
# [1] Ultimate Geo-SNN 모델 (발화 임계값 대폭 완화)
# =========================================================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1.0 + 5.0 * torch.abs(input)) ** 2

spike_fn = SurrogateSpike.apply

class PhysioNetGeoLIF_4Class(nn.Module):
    # [수정 포인트 1] base_thresh를 1.0에서 0.2로 확 낮춰서 뉴런이 쉽게 반응하게 함
    def __init__(self, num_eeg_channels=64, hidden_dim=64, num_classes=4, leak=0.9, base_thresh=0.2):
        super().__init__()
        self.leak = leak
        self.base_thresh = base_thresh
        
        self.spatial_conv1x1 = nn.Linear(num_eeg_channels, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        
        self.class_conv1x1 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.lateral_weights = nn.Parameter(torch.eye(4) * 0.8 - 0.1, requires_grad=True)
        self.tda_to_thresh = nn.Linear(50, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        B, T, _ = kin_spikes_seq.shape
        
        x_proj = self.spatial_conv1x1(kin_spikes_seq) 
        x_proj = self.bn1(x_proj.transpose(1, 2)).transpose(1, 2) 
        
        mem1 = torch.zeros(B, 64, device=kin_spikes_seq.device)
        mem2 = torch.zeros(B, 4, device=kin_spikes_seq.device)
        spike_sum = torch.zeros(B, 4, device=kin_spikes_seq.device)
        
        tda_mod = torch.sigmoid(self.tda_to_thresh(tda_features)) * 0.5 
        dynamic_thresh = self.base_thresh + tda_mod 
        
        for t in range(T):
            mem1 = self.leak * mem1 + x_proj[:, t, :]
            spk1 = spike_fn(mem1 - self.base_thresh)
            mem1 = mem1 * (1.0 - spk1) 
            
            cur2 = torch.matmul(self.class_conv1x1(spk1), self.lateral_weights)
            mem2 = self.leak * mem2 + cur2
            spk2 = spike_fn(mem2 - dynamic_thresh)
            mem2 = mem2 * (1.0 - spk2)
            
            spike_sum += spk2 
            
        return spike_sum

# =========================================================
# [2] 병렬 데이터 엔진 (스파이크 유입량 증가)
# =========================================================
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    paths = [os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf"),
             os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")]
    for p in paths:
        if os.path.exists(p): return p
    return None

def process_single_subject(sub):
    try:
        def load_runs(runs):
            raws = [mne.io.read_raw_edf(get_local_file_path(sub, r), preload=True, verbose=False) for r in runs if get_local_file_path(sub, r)]
            if not raws: return None
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            raw.rename_channels(lambda x: x.strip('.').upper().replace('Z', 'z'))
            raw.set_montage('standard_1005', on_missing='ignore')
            return raw

        raw_lr = load_runs([4, 8, 12]) 
        raw_f = load_runs([6, 10, 14])  
        if raw_lr is None or raw_f is None: return None

        def get_epochs(raw):
            evs, ev_id = mne.events_from_annotations(raw, verbose=False)
            t0 = next((v for k, v in ev_id.items() if 'T0' in k), None)
            t1 = next((v for k, v in ev_id.items() if 'T1' in k), None)
            t2 = next((v for k, v in ev_id.items() if 'T2' in k), None)
            
            ep = mne.Epochs(raw, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            try: csd = mne.preprocessing.compute_current_source_density(raw.copy(), sphere=(0,0,0,0.095))
            except: csd = raw.copy()
            ep_c = mne.Epochs(csd, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            return ep.get_data(), ep_c.get_data(), ep.events[:, 2], t0, t1, t2, ep.ch_names

        d_lr, c_lr, y_lr, lr0, lr1, lr2, chs = get_epochs(raw_lr)
        d_f, c_f, y_f, f0, f1, f2, _ = get_epochs(raw_f)
        
        idx_lh = np.where(y_lr == lr1)[0] 
        idx_rh = np.where(y_lr == lr2)[0] 
        idx_ft = np.where(y_f == f2)[0]   
        idx_rest_lr = np.where(y_lr == lr0)[0]
        idx_rest_f = np.where(y_f == f0)[0]
        
        min_trials = min(len(idx_lh), len(idx_rh), len(idx_ft), len(idx_rest_lr) + len(idx_rest_f))
        if min_trials == 0: return None
        
        idx_lh, idx_rh, idx_ft = idx_lh[:min_trials], idx_rh[:min_trials], idx_ft[:min_trials]
        idx_rest = np.concatenate([idx_rest_lr, idx_rest_f + len(y_lr)])[:min_trials]
        
        d_all = np.concatenate([d_lr, d_f])
        c_all = np.concatenate([c_lr, c_f])
        
        xb = np.concatenate([d_all[idx_lh], d_all[idx_rh], d_all[idx_ft], d_all[idx_rest]])
        xc = np.concatenate([c_all[idx_lh], c_all[idx_rh], c_all[idx_ft], c_all[idx_rest]])
        
        y = np.concatenate([np.zeros(min_trials), np.ones(min_trials), np.full(min_trials, 2), np.full(min_trials, 3)])

        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), 51, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), 51, 3, axis=-1)
        du, dv = np.gradient(u, axis=-1), np.gradient(v, axis=-1)
        
        geo = (0.5 * np.abs(u * dv - v * du) * np.abs(u*dv - v*du)/(np.clip((du**2 + dv**2)**1.5, 1e-6, None)))[:, :, 80:400]
        geo_norm = (geo - geo.mean(axis=(1,2), keepdims=True)) / (geo.std(axis=(1,2), keepdims=True) + 1e-6)
        
        # [수정 포인트 2] 스파이크 유입량을 확 늘림 (Z > 1.0 -> Z > 0.5)
        kin_spikes = (geo_norm > 0.5).astype(np.float32).transpose(0, 2, 1)
        
        ch_upper = [c.upper() for c in chs]
        idx_c = [ch_upper.index(n) for n in ['C3', 'CZ', 'C4'] if n in ch_upper]
        if len(idx_c) < 3: return None
        
        VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
        PL = PersistenceLandscape(n_bins=50)
        tda_in = np.stack([u[:,idx_c[0],:], v[:,idx_c[0],:], u[:,idx_c[1],:], v[:,idx_c[1],:], u[:,idx_c[2],:], v[:,idx_c[2],:]], axis=-1)
        tda = PL.fit_transform(VR.fit_transform(tda_in)).reshape(len(y), -1).astype(np.float32)

        return kin_spikes, tda, y
    except Exception as e: 
        return None

def get_full_data_parallel(subjects):
    num_workers = min(12, cpu_count()) 
    print(f"🔥 CPU 코어 {num_workers}개 사용. 데이터 전처리 및 밸런싱 시작...")
    
    with Pool(num_workers) as p:
        results = [r for r in list(tqdm(p.imap(process_single_subject, subjects), total=len(subjects), desc="전처리", mininterval=5)) if r is not None]
    
    if not results: raise ValueError("⚠️ 데이터 추출 실패.")
    
    Xk = np.concatenate([r[0] for r in results])
    Xt = np.concatenate([r[1] for r in results])
    y = np.concatenate([r[2] for r in results])
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 [완벽한 클래스 분포] LH(0): {counts[0]}, RH(1): {counts[1]}, Feet(2): {counts[2]}, Rest(3): {counts[3]}")
    return torch.tensor(Xk), torch.tensor(Xt), torch.tensor(y, dtype=torch.long)

# =========================================================
# [3] 메인 학습 루프
# =========================================================
if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    Xk, Xt, y = get_full_data_parallel(VALID_SUBS)
    
    Xk_tr, Xk_ts, Xt_tr, Xt_ts, y_tr, y_ts = train_test_split(Xk, Xt, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(Xk_tr, Xt_tr, y_tr), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xk_ts, Xt_ts, y_ts), batch_size=128, shuffle=False)

    model = PhysioNetGeoLIF_4Class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 진짜 보물 사냥 시작! 샘플 수: {len(y)}")
    pbar = tqdm(range(100), desc="에포크", mininterval=5)
    best_acc = 0.0
    
    for epoch in pbar:
        model.train()
        tr_c, tr_t = 0, 0
        for kb, tb, yb in train_loader:
            kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(kb, tb)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            tr_c += out.max(1)[1].eq(yb).sum().item(); tr_t += yb.size(0)
        
        scheduler.step()
        
        model.eval()
        ts_c, ts_t = 0, 0
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                ts_c += model(kb, tb).max(1)[1].eq(yb).sum().item(); ts_t += yb.size(0)
        
        cur_acc = 100 * ts_c / ts_t
        if cur_acc > best_acc: best_acc = cur_acc
        
        pbar.set_postfix(Train=f"{100*tr_c/tr_t:.1f}%", Test=f"{cur_acc:.1f}%", Best=f"{best_acc:.1f}%")