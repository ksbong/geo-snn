import os
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

# =========================================================
# [1] 모델 구조 (입력 채널 64 -> 128로 확장)
# =========================================================
class GeoHRSNN(nn.Module):
    def __init__(self, in_ch=128, hid_ch=64, out_ch=4):
        super().__init__()
        
        # 공간 피처 매핑
        self.spatial_fc = nn.Linear(in_ch, hid_ch)
        nn.init.normal_(self.spatial_fc.weight, mean=0.05, std=0.1)
        
        # LI (Leaky Integrator) Layer
        self.li_fc = nn.Linear(hid_ch * 2, hid_ch)
        
        # TDA 네트워크
        self.tda_net = nn.Sequential(
            nn.Linear(150, 64), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU()
        )
        
        # 최종 분류기 (SNN 거시적 특징 64 + TDA 32 = 96)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hid_ch + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_ch)
        )

    def forward(self, x, tda):
        B, T, C = x.shape
        x = nn.functional.dropout(x, p=0.1, training=self.training)
        
        # [핵심 솔루션: 시간 뭉개기 (Temporal Binning)]
        # 차원 변경: [B, 320, 128] -> [B, 128, 320]
        x = x.transpose(1, 2)
        
        # 320스텝을 16단위로 묶어 평균 밀도를 구함 -> 20스텝으로 압축
        # 뾰족한 스파이크가 부드럽고 거시적인 연속 전류로 변환됨 (RF의 강력함을 차용)
        x = torch.nn.functional.avg_pool1d(x, kernel_size=16, stride=16) 
        
        # SNN 입력을 위해 복구: [B, 128, 20] -> [B, 20, 128]
        x = x.transpose(1, 2)
        
        new_T = x.shape[1] # 20 스텝
        
        mem_fast = torch.zeros(B, 64, device=x.device)
        mem_slow = torch.zeros(B, 64, device=x.device)
        mem_li = torch.zeros(B, 64, device=x.device)
        
        li_sum = torch.zeros(B, 64, device=x.device)
        
        # 단 20번만 루프를 돌기 때문에 속도는 16배 빨라지고 타이밍 오차는 완벽히 방어됨
        for t in range(new_T):
            cur = self.spatial_fc(x[:, t, :])
            
            mem_fast = 0.5 * mem_fast + cur
            spk_fast = spike_fn(mem_fast - 0.5)
            mem_fast = mem_fast * (1.0 - spk_fast)
            
            mem_slow = 0.9 * mem_slow + cur
            spk_slow = spike_fn(mem_slow - 1.0)
            mem_slow = mem_slow * (1.0 - spk_slow)
            
            spk_cat = torch.cat([spk_fast, spk_slow], dim=-1)
            
            cur_li = self.li_fc(spk_cat)
            mem_li = 0.9 * mem_li + cur_li
            li_sum += mem_li
            
        # 20스텝 동안 누적된 거시적 에너지의 평균을 추출
        snn_feat = li_sum / new_T
        
        tda_feat = self.tda_net(tda) 
        out = self.classifier(torch.cat([snn_feat, tda_feat], dim=-1))
        
        return out
    
# =========================================================
# [2] 병렬 데이터 엔진 (네 아이디어: 피처 분리 탑재)
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
            mne.datasets.eegbci.standardize(raw)
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
            csd = mne.preprocessing.compute_current_source_density(raw.copy(), sphere=(0,0,0,0.095))
            ep_c = mne.Epochs(csd, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            return ep.get_data(), ep_c.get_data(), ep.events[:, 2], t0, t1, t2, ep.ch_names

        d_lr, c_lr, y_lr, lr0, lr1, lr2, chs = get_epochs(raw_lr)
        d_f, c_f, y_f, f0, f1, f2, _ = get_epochs(raw_f)
        
        idx_lh, idx_rh, idx_ft = np.where(y_lr == lr1)[0], np.where(y_lr == lr2)[0], np.where(y_f == f2)[0]
        idx_rest_lr, idx_rest_f = np.where(y_lr == lr0)[0], np.where(y_f == f0)[0]
        min_trials = min(len(idx_lh), len(idx_rh), len(idx_ft), len(idx_rest_lr) + len(idx_rest_f))
        if min_trials == 0: return None
        
        idx_lh, idx_rh, idx_ft = idx_lh[:min_trials], idx_rh[:min_trials], idx_ft[:min_trials]
        d_lh, c_lh = d_lr[idx_lh], c_lr[idx_lh]
        d_rh, c_rh = d_lr[idx_rh], c_lr[idx_rh]
        d_ft, c_ft = d_f[idx_ft], c_f[idx_ft]
        d_rest = np.concatenate([d_lr[idx_rest_lr], d_f[idx_rest_f]])[:min_trials]
        c_rest = np.concatenate([c_lr[idx_rest_lr], c_f[idx_rest_f]])[:min_trials]
        
        xb = np.concatenate([d_lh, d_rh, d_ft, d_rest])
        xc = np.concatenate([c_lh, c_rh, c_ft, c_rest])
        y = np.concatenate([np.zeros(min_trials), np.ones(min_trials), np.full(min_trials, 2), np.full(min_trials, 3)])
        subjects_arr = np.full(len(y), sub)

        win = 51
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), win, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), win, 3, axis=-1)
        du = savgol_filter(u, win, 3, deriv=1, axis=-1)
        dv = savgol_filter(v, win, 3, deriv=1, axis=-1)
        ddu = savgol_filter(u, win, 3, deriv=2, axis=-1)
        ddv = savgol_filter(v, win, 3, deriv=2, axis=-1)
        
        # [핵심 변경] 면적속도와 곡률을 곱하지 않고 완전히 독립적으로 계산
        areal_vel = (0.5 * np.abs(u * dv - v * du))[:, :, 80:400]
        denom = np.clip((du**2 + dv**2)**(1.5), 1e-6, None)
        curvature = (np.clip(np.abs(du * ddv - dv * ddu) / denom, 0, 10))[:, :, 80:400]
        
        areal_vel = np.nan_to_num(areal_vel, nan=0.0, posinf=0.0, neginf=0.0)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. 면적속도 독립 정규화 및 스파이크 발화
        vel_mean = np.mean(areal_vel, axis=(1, 2), keepdims=True)
        vel_std = np.std(areal_vel, axis=(1, 2), keepdims=True) + 1e-8
        vel_norm = (areal_vel - vel_mean) / vel_std
        th_vel = np.percentile(vel_norm, 85)
        spikes_vel = (vel_norm > th_vel).astype(np.float32) # [Trials, 64, 320]
        
        # 2. 곡률 독립 정규화 및 스파이크 발화
        curv_mean = np.mean(curvature, axis=(1, 2), keepdims=True)
        curv_std = np.std(curvature, axis=(1, 2), keepdims=True) + 1e-8
        curv_norm = (curvature - curv_mean) / curv_std
        th_curv = np.percentile(curv_norm, 85)
        spikes_curv = (curv_norm > th_curv).astype(np.float32) # [Trials, 64, 320]
        
        # 3. 면적속도(64)와 곡률(64)을 채널 축으로 이어붙임 -> 총 128채널
        kin_features = np.concatenate([spikes_vel, spikes_curv], axis=1) # [Trials, 128, 320]
        kin_spikes = kin_features.transpose(0, 2, 1) # 모델 입력을 위해 [Trials, 320, 128]로 변환
        
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

        return kin_spikes, tda_final, y, subjects_arr
    except Exception as e: 
        return None

def get_full_data_parallel(subjects):
    num_workers = min(12, cpu_count()) 
    print(f"🔥 피처 뭉개짐 해결 (면적속도/곡률 분리). 병렬 전처리 시작...")
    results = []
    with Pool(num_workers) as p:
        for i, r in enumerate(p.imap(process_single_subject, subjects)):
            if r is not None: results.append(r)
            if (i + 1) % 10 == 0: print(f"  -> 전처리 진행중: {i+1}/{len(subjects)} 명 완료...")
                
    if not results: raise ValueError("⚠️ 데이터 추출 실패.")
    Xk = np.concatenate([r[0] for r in results])
    Xt = np.concatenate([r[1] for r in results])
    y = np.concatenate([r[2] for r in results])
    subs = np.concatenate([r[3] for r in results])
    print(f"📊 최종 클래스 분포: {dict(zip(*np.unique(y, return_counts=True)))}")
    # 희소성이 128채널 전체에 걸쳐 0.15로 잘 유지되는지 확인
    print(f"🔍 입력 스파이크 희소성(Sparsity): {Xk.mean():.4f} (약 0.15 정상 복원)")
    return torch.tensor(Xk), torch.tensor(Xt), torch.tensor(y, dtype=torch.long), subs

if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    Xk, Xt, y, subs = get_full_data_parallel(VALID_SUBS)
    
    unique_subs = np.unique(subs)
    train_subs, test_subs = train_test_split(unique_subs, test_size=0.2, random_state=42)
    
    train_idx = np.isin(subs, train_subs)
    test_idx = np.isin(subs, test_subs)
    
    Xk_tr, Xt_tr, y_tr = Xk[train_idx], Xt[train_idx], y[train_idx]
    Xk_ts, Xt_ts, y_ts = Xk[test_idx], Xt[test_idx], y[test_idx]
    
    train_loader = DataLoader(TensorDataset(Xk_tr, Xt_tr, y_tr), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xk_ts, Xt_ts, y_ts), batch_size=128, shuffle=False)

    model = GeoHRSNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 분리된 128채널 기하학 피처로 Global Training 개시")
    print(f"Train 피험자: {len(train_subs)}명, Test 피험자: {len(test_subs)}명")
    best_acc = 0.0
    
    for epoch in range(1, 101):
        model.train()
        tr_loss, tr_c, tr_t = 0.0, 0, 0
        
        for kb, tb, yb in train_loader:
            kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            out = model(kb, tb)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            
            preds = out.max(1)[1]
            tr_loss += loss.item() * yb.size(0)
            tr_c += preds.eq(yb).sum().item()
            tr_t += yb.size(0)
            
        scheduler.step()
        
        model.eval()
        ts_c, ts_t = 0, 0
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                out = model(kb, tb)
                preds = out.max(1)[1]
                ts_c += preds.eq(yb).sum().item()
                ts_t += yb.size(0)
                
        epoch_loss = tr_loss / tr_t
        tr_acc = 100 * tr_c / tr_t
        ts_acc = 100 * ts_c / ts_t
        
        if ts_acc > best_acc: best_acc = ts_acc
        
        print(f"Epoch [{epoch:03d}/100] Loss: {epoch_loss:.4f} | Train Acc: {tr_acc:.1f}% | Test Acc: {ts_acc:.1f}% | Best Test Acc: {best_acc:.1f}%")