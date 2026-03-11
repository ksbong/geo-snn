import os
import gc
import time
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
# [1] SNN 모델 (핵심 로직은 유지하되 연산 효율 극대화)
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
        
        # 억제망 초기값 세팅
        init_inhibition = torch.tensor([
            [ 1.0, -0.4, -0.4, -0.4], [ -0.4, 1.0, -0.3, -0.3], 
            [ -0.4, -0.3, 1.0, -0.3], [ -0.4, -0.3, -0.3, 1.0]
        ])
        self.lateral_weights = nn.Parameter(init_inhibition, requires_grad=True)
        self.tda_to_thresh = nn.Linear(50, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        batch_size, time_steps, _ = kin_spikes_seq.shape
        mem = torch.zeros(batch_size, self.num_classes, device=kin_spikes_seq.device)
        
        # TDA 기반 동적 임계값 (Batch별 고정)
        tda_mod = torch.sigmoid(self.tda_to_thresh(tda_features)) * 0.5
        dynamic_thresh = self.base_thresh + tda_mod 
        
        spike_records = []
        for t in range(time_steps):
            # 64채널 투영 -> 억제망 통과 -> LIF 누적
            cur_input = torch.matmul(self.spatial_weights(kin_spikes_seq[:, t, :]), self.lateral_weights)
            mem = self.leak * mem + cur_input
            spikes_out = spike_fn(mem - dynamic_thresh)
            mem = mem * (1.0 - spikes_out) # Reset
            spike_records.append(spikes_out)
            
        return torch.stack(spike_records, dim=1)

# =========================================================
# [2] 병렬 처리 엔진 (Ryzen 7950X 32쓰레드 풀가동)
# =========================================================
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
    return path if os.path.exists(path) else os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")
def process_single_subject(sub):
    """ 에러 원인을 출력하도록 수정된 워커 함수 """
    try:
        def load_group(runs, event_map):
            raws = []
            for r in runs:
                path = get_local_file_path(sub, r)
                if os.path.exists(path):
                    raws.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
            
            if not raws: 
                return None, None, None
            
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            mne.datasets.eegbci.standardize(raw)
            csd = mne.preprocessing.compute_current_source_density(raw.copy(), sphere=(0, 0, 0, 0.095))
            
            # 여기서 Annotation이 없으면 터짐
            ev, ev_id = mne.events_from_annotations(raw, verbose=False)
            
            # 만약 T0, T1 같은 이름이 없으면 런타임 에러 발생
            ep_b = mne.Epochs(raw, ev, event_id=event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c = mne.Epochs(csd, ev, event_id=event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            return ep_b.get_data(), ep_c.get_data(), ep_b.events[:, 2]

        # 데이터 로드
        res_lr = load_group([4, 8, 12], {'T0':1, 'T1':2, 'T2':3})
        res_f = load_group([6, 10, 14], {'T0':1, 'T2':3})

        if res_lr[0] is None or res_f[0] is None:
            print(f"⚠️ Sub {sub}: 파일을 찾을 수 없거나 이벤트가 비어있음.")
            return None

        xb, xc, y = np.concatenate([res_lr[0], res_f[0]]), np.concatenate([res_lr[1], res_f[1]]), np.concatenate([res_lr[2] - 1, np.where(res_f[2] == 1, 0, 3)])

        # 기하학 에너지 스파이크
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), 51, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), 51, 3, axis=-1)
        du, dv = np.gradient(u, axis=-1), np.gradient(v, axis=-1)
        ddu, ddv = np.gradient(du, axis=-1), np.gradient(dv, axis=-1)
        
        areal_vel = 0.5 * np.abs(u * dv - v * du)[:, :, 80:400]
        curv = np.clip(np.abs(du * ddv - dv * ddu) / np.clip((du**2 + dv**2)**1.5, 1e-6, None), 0, 10)[:, :, 80:400]
        
        geo_energy = areal_vel * curv
        kin_spikes = (geo_energy > np.percentile(geo_energy, 90)).astype(np.float32).transpose(0, 2, 1)
        
        # TDA (C3, Cz, C4 - 채널 이름으로 동적 인덱싱)
        ch_names = [ch.upper() for ch in raws[0].ch_names] if 'raws' in locals() else []
        # 만약 인덱스가 안 맞으면 여기서 에러 날 수 있으니 체크 필요
        point_clouds = np.stack([u[:,12,:], v[:,12,:], u[:,14,:], v[:,14,:], u[:,16,:], v[:,16,:]], axis=-1)

        VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
        PL = PersistenceLandscape(n_bins=50)
        tda = PL.fit_transform(VR.fit_transform(point_clouds)).reshape(len(y), -1).astype(np.float32)

        return kin_spikes, tda, y
    except Exception as e:
        print(f"❌ Sub {sub} Error: {str(e)}") # 어떤 에러인지 출력
        return None
    
def get_full_data_parallel(subjects):
    print(f"🔥 CPU 코어 {cpu_count()}개를 사용하여 {len(subjects)}명 데이터 병렬 전처리 시작...")
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_single_subject, subjects), total=len(subjects), desc="전처리 진행상황"))
    
    results = [r for r in results if r is not None]
    return torch.tensor(np.concatenate([r[0] for r in results])), \
           torch.tensor(np.concatenate([r[1] for r in results])), \
           torch.tensor(np.concatenate([r[2] for r in results]), dtype=torch.long)

# =========================================================
# [3] 메인 트레이닝 루프
# =========================================================
if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    
    # 1. 병렬 데이터 로드
    Xk, Xt, y = get_full_data_parallel(VALID_SUBS)
    
    # 2. 84명(Train) / 21명(Test) 분리 (논문 세팅)
    Xk_train, Xk_test, Xt_train, Xt_test, y_train, y_test = train_test_split(Xk, Xt, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(Xk_train, Xt_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xk_test, Xt_test, y_test), batch_size=128, shuffle=False)

    model = PhysioNetGeoLIF_4Class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 SNN 하이브리드 학습 개시 (총 샘플: {len(y)})")
    pbar = tqdm(range(100), desc="에포크")
    for epoch in pbar:
        model.train()
        tr_correct, tr_total = 0, 0
        for kb, tb, yb in train_loader:
            kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            out_sum = model(kb, tb).sum(dim=1)
            loss = criterion(out_sum, yb)
            loss.backward(); optimizer.step()
            tr_correct += out_sum.max(1)[1].eq(yb).sum().item(); tr_total += yb.size(0)
        
        scheduler.step()
        
        # Zero-shot Test
        model.eval()
        ts_correct, ts_total = 0, 0
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                ts_correct += model(kb, tb).sum(dim=1).max(1)[1].eq(yb).sum().item(); ts_total += yb.size(0)
        
        pbar.set_postfix(Train=f"{100*tr_correct/tr_total:.1f}%", Test=f"{100*ts_correct/ts_total:.1f}%")

    print("\n✅ 모든 과정 종료. 최종 정확도를 확인해!")