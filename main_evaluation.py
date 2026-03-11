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

# [중요] 네 로컬 데이터 경로 확인
DATA_DIR_PHYSIONET = './raw_data/files'

# =========================================================
# [1] SNN 모델 (동일)
# =========================================================
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
    def __init__(self, num_eeg_channels=64, num_classes=4, leak=0.8, base_thresh=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.leak = leak
        self.base_thresh = base_thresh
        self.spatial_weights = nn.Linear(num_eeg_channels, num_classes, bias=False)
        init_inhibition = torch.tensor([
            [ 1.0, -0.4, -0.4, -0.4], [ -0.4, 1.0, -0.3, -0.3], 
            [ -0.4, -0.3, 1.0, -0.3], [ -0.4, -0.3, -0.3, 1.0]
        ])
        self.lateral_weights = nn.Parameter(init_inhibition, requires_grad=True)
        self.tda_to_thresh = nn.Linear(50, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        batch_size, time_steps, _ = kin_spikes_seq.shape
        mem = torch.zeros(batch_size, self.num_classes, device=kin_spikes_seq.device)
        tda_mod = torch.sigmoid(self.tda_to_thresh(tda_features)) * 0.5
        dynamic_thresh = self.base_thresh + tda_mod 
        spike_records = []
        for t in range(time_steps):
            cur_input = torch.matmul(self.spatial_weights(kin_spikes_seq[:, t, :]), self.lateral_weights)
            mem = self.leak * mem + cur_input
            spikes_out = spike_fn(mem - dynamic_thresh)
            mem = mem * (1.0 - spikes_out)
            spike_records.append(spikes_out)
        return torch.stack(spike_records, dim=1)

# =========================================================
# [2] 디버깅 기능이 강화된 병렬 데이터 엔진
# =========================================================
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    # 대소문자 및 다양한 경로 패턴 대응
    paths = [
        os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf"),
        os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf"),
        os.path.join(DATA_DIR_PHYSIONET, f"S{subject:03d}", f"S{subject:03d}R{run:02d}.edf")
    ]
    for p in paths:
        if os.path.exists(p): return p
    return None

def process_single_subject(sub):
    try:
        def load_group(runs, expected_events):
            raws = []
            for r in runs:
                path = get_local_file_path(sub, r)
                if path:
                    raws.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
            
            if not raws: return None, None, None, None
            
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            
            # [좌표 에러 해결] 몽타주를 입히기 전 채널 이름을 표준에 맞춰 정규화
            # PhysioNet 전극 이름은 보통 'Fc1.' 이런 식이라 점(.)을 빼야 함
            mapping = {ch: ch.replace('.', '').upper().replace('Z', 'z') for ch in raw.ch_names}
            raw.rename_channels(mapping)
            raw.set_montage('standard_1005', on_missing='ignore')
            
            # CSD 연산 (안전하게 구체 파라미터 제외하거나 기본값 사용)
            try:
                csd = mne.preprocessing.compute_current_source_density(raw.copy())
            except:
                csd = raw.copy() # CSD 실패 시 일반 RAW로 우회 (안전장치)
            
            # 이벤트 추출 (Annotation 기반)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            
            # PhysioNet은 보통 'T0', 'T1', 'T2' 이름을 가짐. 
            # 하지만 가끔 숫자로 들어오는 경우가 있어 이를 유연하게 처리
            actual_event_map = {}
            for target in expected_events:
                # T1, T2 등을 event_id에서 찾음
                found_key = [k for k in event_id.keys() if target in k]
                if found_key:
                    actual_event_map[found_key[0]] = event_id[found_key[0]]
            
            if not actual_event_map: return None, None, None, None
            
            ep_b = mne.Epochs(raw, events, event_id=actual_event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c = mne.Epochs(csd, events, event_id=actual_event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            
            return ep_b.get_data(), ep_c.get_data(), ep_b.events[:, 2], ep_b.ch_names

        # 1. LH, RH 로드 (T1, T2)
        xb_lr, xc_lr, y_lr, chs = load_group([4, 8, 12], ['T1', 'T2'])
        if xb_lr is None: return None
        
        # 2. Feet 로드 (T2)
        xb_f, xc_f, y_f, _ = load_group([6, 10, 14], ['T2'])
        if xb_f is None: return None
        
        # 라벨 정규화 (LH=0, RH=1, Feet=2, Rest=3 구성을 위해 임시 매핑)
        # PhysioNet 런마다 이벤트 값이 다르므로 수동 정렬
        y_lr_norm = np.where(y_lr == np.min(y_lr), 0, 1) # LH=0, RH=1
        y_f_norm = np.full(len(y_f), 2) # Feet=2
        
        xb = np.concatenate([xb_lr, xb_f])
        xc = np.concatenate([xc_lr, xc_f])
        y = np.concatenate([y_lr_norm, y_f_norm])

        # 기하학 피처 계산
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), 51, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), 51, 3, axis=-1)
        du, dv = np.gradient(u, axis=-1), np.gradient(v, axis=-1)
        ddu, ddv = np.gradient(du, axis=-1), np.gradient(dv, axis=-1)
        
        areal_vel = 0.5 * np.abs(u * dv - v * du)[:, :, 80:400]
        denom = np.clip((du**2 + dv**2)**1.5, 1e-6, None)
        curv = np.clip(np.abs(du * ddv - dv * ddu) / denom, 0, 10)[:, :, 80:400]
        
        geo_energy = areal_vel * curv
        kin_spikes = (geo_energy > np.percentile(geo_energy, 90)).astype(np.float32).transpose(0, 2, 1)
        
        # TDA (C3, Cz, C4 위치 찾기)
        ch_upper = [c.upper() for c in chs]
        idx = [ch_upper.index(n) for n in ['C3', 'CZ', 'C4'] if n in ch_upper]
        if len(idx) < 3: return None
        
        VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
        PL = PersistenceLandscape(n_bins=50)
        tda_in = np.stack([u[:,idx[0],:], v[:,idx[0],:], u[:,idx[1],:], v[:,idx[1],:], u[:,idx[2],:], v[:,idx[2],:]], axis=-1)
        tda = PL.fit_transform(VR.fit_transform(tda_in)).reshape(len(y), -1).astype(np.float32)

        return kin_spikes, tda, y
    except Exception as e:
        # 처음 몇 명만 에러 이유를 출력하게 함 (디버깅용)
        if sub < 5: print(f"❌ Sub {sub} Debug Error: {str(e)}")
        return None

def get_full_data_parallel(subjects):
    print(f"🔥 {len(subjects)}명 데이터 병렬 전처리 시작...")
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_single_subject, subjects), total=len(subjects), desc="전처리"))
    
    clean_results = [r for r in results if r is not None]
    if not clean_results:
        raise ValueError("⚠️ 모든 피험자 처리 실패! 상단에 찍힌 Debug Error 메시지를 확인하세요.")
    
    return torch.tensor(np.concatenate([r[0] for r in clean_results])), \
           torch.tensor(np.concatenate([r[1] for r in clean_results])), \
           torch.tensor(np.concatenate([r[2] for r in clean_results]), dtype=torch.long)

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
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 학습 시작! 총 샘플: {len(y)}")
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