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
# [1] SNN 모델 (핵심 로직 유지)
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
# [2] 병렬 처리 엔진 (철벽 방어 및 동적 인덱싱 적용)
# =========================================================
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
    return path if os.path.exists(path) else os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")

def process_single_subject(sub):
    try:
        def load_group(runs, event_map):
            raws = []
            for r in runs:
                path = get_local_file_path(sub, r)
                if os.path.exists(path):
                    raws.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
            if not raws: return None, None, None, None
            
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            
            # [에러 해결 포인트 1] 전극 좌표 강제 재설정
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
            # PhysioNet 전극 이름 정규화 (예: 'Fc1.' -> 'FC1')
            mapping = {ch: ch.replace('.', '').upper().replace('Z', 'z') for ch in raw.ch_names}
            raw.rename_channels(mapping)
            
            # 표준 몽타주 설정 (Zero position 방지)
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
            
            # CSD 연산 (이제 전극 위치 에러 안 남)
            csd = mne.preprocessing.compute_current_source_density(raw.copy(), sphere=(0, 0, 0, 0.095))
            
            ev, ev_id = mne.events_from_annotations(raw, verbose=False)
            ep_b = mne.Epochs(raw, ev, event_id=event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c = mne.Epochs(csd, ev, event_id=event_map, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            
            return ep_b.get_data(), ep_c.get_data(), ep_b.events[:, 2], ep_b.ch_names

        # 데이터 로드 및 라벨 통합
        xb_lr, xc_lr, y_lr, chs = load_group([4, 8, 12], {'T1':2, 'T2':3}) # Rest 제외하고 운동만
        if xb_lr is None: return None
        y_lr -= 2 # Left: 0, Right: 1
        
        xb_f, xc_f, y_f, _ = load_group([6, 10, 14], {'T1':2, 'T2':3}) # Both Hands vs Feet
        if xb_f is None: return None
        # 논문 기준 4-Class 매핑: Left(0), Right(1), Feet(2), Rest(3)로 재구성하거나
        # 현재 코드 흐름대로 유지: Left(0), Right(1), BothFists(2), Feet(3)
        y_f = np.where(y_f == 2, 2, 3) 

        xb, xc, y = np.concatenate([xb_lr, xb_f]), np.concatenate([xc_lr, xc_f]), np.concatenate([y_lr, y_f])

        # [에러 해결 포인트 2] 동적 채널 인덱싱 (C3, Cz, C4)
        ch_names = [c.upper() for c in chs]
        idx_c3 = ch_names.index('C3')
        idx_cz = ch_names.index('CZ')
        idx_c4 = ch_names.index('C4')

        # 기하학 에너지 스파이크 인코딩
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), 51, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), 51, 3, axis=-1)
        du, dv = np.gradient(u, axis=-1), np.gradient(v, axis=-1)
        ddu, ddv = np.gradient(du, axis=-1), np.gradient(dv, axis=-1)
        
        areal_vel = 0.5 * np.abs(u * dv - v * du)[:, :, 80:400]
        curv = np.clip(np.abs(du * ddv - dv * ddu) / np.clip((du**2 + dv**2)**1.5, 1e-6, None), 0, 10)[:, :, 80:400]
        
        geo_energy = areal_vel * curv
        kin_spikes = (geo_energy > np.percentile(geo_energy, 90)).astype(np.float32).transpose(0, 2, 1)
        
        # TDA
        VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
        PL = PersistenceLandscape(n_bins=50)
        tda_input = np.stack([u[:,idx_c3,:], v[:,idx_c3,:], u[:,idx_cz,:], v[:,idx_cz,:], u[:,idx_c4,:], v[:,idx_c4,:]], axis=-1)
        tda = PL.fit_transform(VR.fit_transform(tda_input)).reshape(len(y), -1).astype(np.float32)

        return kin_spikes, tda, y
    except Exception as e:
        # print(f"❌ Sub {sub} Skipped due to: {str(e)}")
        return None

def get_full_data_parallel(subjects):
    print(f"🔥 CPU 코어 {cpu_count()}개 풀가동. 105명 데이터 병렬 전처리 개시...")
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_single_subject, subjects), total=len(subjects), desc="전처리 진행상황"))
    
    results = [r for r in results if r is not None]
    if not results: raise ValueError("⚠️ 모든 피험자 처리 실패! 데이터 경로와 포맷을 다시 확인하세요.")
    
    return torch.tensor(np.concatenate([r[0] for r in results])), \
           torch.tensor(np.concatenate([r[1] for r in results])), \
           torch.tensor(np.concatenate([r[2] for r in results]), dtype=torch.long)

# =========================================================
# [3] 메인 트레이닝 루프 (실행 환경 최적화)
# =========================================================
if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    
    Xk, Xt, y = get_full_data_parallel(VALID_SUBS)
    
    # Subject-wise Split이 아닌 Trial-wise 80:20 (간단 검증용)
    Xk_train, Xk_test, Xt_train, Xt_test, y_train, y_test = train_test_split(Xk, Xt, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(Xk_train, Xt_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xk_test, Xt_test, y_test), batch_size=128, shuffle=False)

    model = PhysioNetGeoLIF_4Class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Ultimate Geo-SNN 학습 시작 (샘플 수: {len(y)})")
    pbar = tqdm(range(100), desc="학습 에포크")
    for epoch in pbar:
        model.train()
        tr_corr, tr_total = 0, 0
        for kb, tb, yb in train_loader:
            kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(kb, tb).sum(dim=1)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            tr_corr += out.max(1)[1].eq(yb).sum().item(); tr_total += yb.size(0)
        
        scheduler.step()
        model.eval()
        ts_corr, ts_total = 0, 0
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                ts_corr += model(kb, tb).sum(dim=1).max(1)[1].eq(yb).sum().item(); ts_total += yb.size(0)
        
        pbar.set_postfix(Train=f"{100*tr_corr/tr_total:.1f}%", Test=f"{100*ts_corr/ts_total:.1f}%")

    print("\n✅ 모든 검증 완료. 최종 결과가 네 노력을 증명하길 빈다!")