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
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR_PHYSIONET = './raw_data/files'

# =========================================================
# [1] SNN 모델 (Readout 계층 및 내부 스파이크 반환 추가)
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
    def __init__(self, num_eeg_channels=64, hidden_dim=64, num_classes=4):
        super().__init__()
        
        # 1. 공간 차원 융합
        self.spatial_fc = nn.Linear(num_eeg_channels, hidden_dim, bias=False)
        
        # 2. TDA를 임계값 조작이 아닌 '어텐션 모듈'로 안전하게 활용
        self.tda_attention = nn.Linear(150, hidden_dim)
        
        # 3. Readout (출력층)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        B, T, _ = kin_spikes_seq.shape
        
        # 음수 가중치로 인해 전압이 깎이는 걸 방지하기 위해 ReLU 적용 (오직 양의 전류만 주입)
        x_proj = torch.relu(self.spatial_fc(kin_spikes_seq)) 
        
        mem = torch.zeros(B, 64, device=kin_spikes_seq.device)
        spike_sum = torch.zeros(B, 64, device=kin_spikes_seq.device)
        
        # [핵심] Leak 삭제 (순수 Integrate-and-Fire). 네 희소한 스파이크를 무조건 누적시킴
        for t in range(T):
            mem = mem + x_proj[:, t, :] # leak 곱하는 부분 완전히 제거
            spk = spike_fn(mem - 1.0)   # 고정 임계값 1.0 (안전한 스파이킹)
            mem = mem * (1.0 - spk)     # 발화 시 리셋
            spike_sum += spk
            
        # [핵심] 위상수학(TDA) 특징을 64개 은닉 뉴런의 활성도를 조절하는 가중치(Attention)로 사용
        tda_weight = torch.sigmoid(self.tda_attention(tda_features)) # [B, 64]
        modulated_spikes = spike_sum * tda_weight # TDA 공간 정보를 스파이크 빈도에 결합
        
        out = self.readout(modulated_spikes)
        return out, spike_sum
    
# =========================================================
# [2] 병렬 데이터 엔진 (TDA 차원 수정 및 네 원본 기하학 수식 유지)
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
        
        idx_lh, idx_rh, idx_ft = np.where(y_lr == lr1)[0], np.where(y_lr == lr2)[0], np.where(y_f == f2)[0]
        idx_rest_lr, idx_rest_f = np.where(y_lr == lr0)[0], np.where(y_f == f0)[0]
        
        min_trials = min(len(idx_lh), len(idx_rh), len(idx_ft), len(idx_rest_lr) + len(idx_rest_f))
        if min_trials == 0: return None
        
        idx_lh, idx_rh, idx_ft = idx_lh[:min_trials], idx_rh[:min_trials], idx_ft[:min_trials]
        idx_rest = np.concatenate([idx_rest_lr, idx_rest_f + len(y_lr)])[:min_trials]
        
        xb = np.concatenate([np.concatenate([d_lr, d_f])[idx_lh], np.concatenate([d_lr, d_f])[idx_rh], np.concatenate([d_lr, d_f])[idx_ft], np.concatenate([d_lr, d_f])[idx_rest]])
        xc = np.concatenate([np.concatenate([c_lr, c_f])[idx_lh], np.concatenate([c_lr, c_f])[idx_rh], np.concatenate([c_lr, c_f])[idx_ft], np.concatenate([c_lr, c_f])[idx_rest]])
        y = np.concatenate([np.zeros(min_trials), np.ones(min_trials), np.full(min_trials, 2), np.full(min_trials, 3)])

        win = 51
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), win, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), win, 3, axis=-1)
        du = savgol_filter(u, win, 3, deriv=1, axis=-1)
        dv = savgol_filter(v, win, 3, deriv=1, axis=-1)
        ddu = savgol_filter(u, win, 3, deriv=2, axis=-1)
        ddv = savgol_filter(v, win, 3, deriv=2, axis=-1)
        
        areal_vel = 0.5 * np.abs(u * dv - v * du)
        denom = np.clip((du**2 + dv**2)**(1.5), 1e-6, None)
        curvature = np.clip(np.abs(du * ddv - dv * ddu) / denom, 0, 10)
        
        geo_energy = (areal_vel * curvature)[:, :, 80:400] 
        th_geo = np.percentile(geo_energy, 85)
        kin_spikes = (geo_energy > th_geo).astype(np.float32).transpose(0, 2, 1) 
        
        ch_upper = [c.upper() for c in chs]
        idx_c = [ch_upper.index(n) for n in ['C3', 'CZ', 'C4'] if n in ch_upper]
        if len(idx_c) < 3: return None
        
        VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
        PL = PersistenceLandscape(n_bins=50)
        
        tda_landscapes = []
        for ch_idx in idx_c:
            pt_cloud = np.stack([u[:, ch_idx, 80:400], v[:, ch_idx, 80:400]], axis=-1) 
            diagrams = VR.fit_transform(pt_cloud)
            landscapes = PL.fit_transform(diagrams).reshape(len(y), -1) 
            tda_landscapes.append(landscapes)
            
        tda_final = np.concatenate(tda_landscapes, axis=1).astype(np.float32) 

        return kin_spikes, tda_final, y
    except Exception as e: 
        return None

def get_full_data_parallel(subjects):
    num_workers = min(12, cpu_count()) 
    print(f"🔥 오리지널 수식 + Readout 모델. {num_workers}개 코어로 전처리 시작...")
    
    # 전처리 과정은 진행 상황 파악을 위해 간단히 출력 (tqdm 제거)
    results = []
    with Pool(num_workers) as p:
        for i, r in enumerate(p.imap(process_single_subject, subjects)):
            if r is not None: results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  -> 전처리 진행중: {i+1}/{len(subjects)} 명 완료...")
                
    if not results: raise ValueError("⚠️ 데이터 추출 실패.")
    Xk = np.concatenate([r[0] for r in results])
    Xt = np.concatenate([r[1] for r in results])
    y = np.concatenate([r[2] for r in results])
    print(f"📊 최종 클래스 분포: {dict(zip(*np.unique(y, return_counts=True)))}")
    return torch.tensor(Xk), torch.tensor(Xt), torch.tensor(y, dtype=torch.long)

# =========================================================
# [3] 메인 학습 루프 (Line-by-Line 로깅 구현)
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

    print(f"\n🚀 본격적인 디버깅 로그 모드로 학습 개시 (샘플 수: {len(y)})")
    best_acc = 0.0
    
    for epoch in range(1, 101):
        model.train()
        tr_loss, tr_c, tr_t = 0.0, 0, 0
        total_spikes = 0.0 # 스파이크 발화량 추적
        all_preds = []     # 예측 쏠림 현상 추적
        
        for kb, tb, yb in train_loader:
            kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            out, spike_sum = model(kb, tb)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            
            preds = out.max(1)[1]
            tr_loss += loss.item() * yb.size(0)
            tr_c += preds.eq(yb).sum().item()
            tr_t += yb.size(0)
            
            total_spikes += spike_sum.sum().item()
            all_preds.extend(preds.cpu().numpy())
            
        scheduler.step()
        
        # 이번 에포크에서 뉴런들이 클래스를 어떻게 찍었는지 분포 계산
        pred_unique, pred_counts = np.unique(all_preds, return_counts=True)
        pred_dist = {int(k): int(v) for k, v in zip(pred_unique, pred_counts)}
        avg_spikes = total_spikes / (tr_t * 64) # 샘플당, 은닉 뉴런당 평균 스파이크 횟수
        
        model.eval()
        ts_c, ts_t = 0, 0
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                out, _ = model(kb, tb)
                ts_c += out.max(1)[1].eq(yb).sum().item()
                ts_t += yb.size(0)
        
        epoch_loss = tr_loss / tr_t
        tr_acc = 100 * tr_c / tr_t
        ts_acc = 100 * ts_c / ts_t
        if ts_acc > best_acc: best_acc = ts_acc
        
        # \r 덮어쓰기 없이 라인 바이 라인으로 투명하게 출력
        print(f"Epoch [{epoch:03d}/100] "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {tr_acc:.1f}% | "
              f"Test Acc: {ts_acc:.1f}% | "
              f"Avg Spikes: {avg_spikes:.2f} | "
              f"Pred Dist: {pred_dist}")