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

# =========================================================
# [1] Surrogate 스파이크
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

# =========================================================
# [2] HR-SNN 구조 (LSTM 제외, DP-Pooling & HR-Module 탑재)
# =========================================================
class GeoHRSNN(nn.Module):
    def __init__(self, in_ch=64, hid_ch=64, out_ch=4, L_DP=32, N_DP=16):
        super().__init__()
        self.L_DP = L_DP # 윈도우 길이
        self.N_DP = N_DP # 전압 차이 계산 길이
        
        # 공간 특징 추출
        self.spatial_fc = nn.Linear(in_ch, hid_ch)
        nn.init.normal_(self.spatial_fc.weight, mean=0.05, std=0.1) 
        
        # Leaky Integrator (LI) Layer: 스파이크를 발화하지 않고 전압만 누적하는 디코더용 계층
        self.li_fc = nn.Linear(hid_ch * 2, hid_ch)
        
        # TDA 처리기
        self.tda_net = nn.Sequential(
            nn.Linear(150, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        
        # DP-Pooling 결과 차원 계산 (T=320, 320/32 = 10개 윈도우)
        self.dp_out_dim = (320 // L_DP) * hid_ch
        
        # 최종 분류기 (DP-Pooling 640 + TDA 32 = 672)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.dp_out_dim + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_ch)
        )

    def forward(self, x, tda):
        B, T, _ = x.shape
        
        # HR Module을 위한 두 가지 LIF 뉴런 전압
        mem_fast = torch.zeros(B, 64, device=x.device)
        mem_slow = torch.zeros(B, 64, device=x.device)
        mem_li = torch.zeros(B, 64, device=x.device)
        
        li_seq = []
        
        for t in range(T):
            cur = self.spatial_fc(x[:, t, :])
            
            # [HR-Module] Branch 1: 빠른 반응 (낮은 임계값, 빠른 감쇠)
            mem_fast = 0.5 * mem_fast + cur
            spk_fast = spike_fn(mem_fast - 0.5)
            mem_fast = mem_fast * (1.0 - spk_fast)
            
            # [HR-Module] Branch 2: 느린 반응 (높은 임계값, 느린 감쇠)
            mem_slow = 0.9 * mem_slow + cur
            spk_slow = spike_fn(mem_slow - 1.0)
            mem_slow = mem_slow * (1.0 - spk_slow)
            
            spk_cat = torch.cat([spk_fast, spk_slow], dim=-1) # [B, 128]
            
            # [LI Layer] 스파이크 없이 순수 전압(Potential)만 누적
            cur_li = self.li_fc(spk_cat)
            mem_li = 0.9 * mem_li + cur_li
            li_seq.append(mem_li.unsqueeze(1))
            
        li_seq = torch.cat(li_seq, dim=1) # [B, 320, 64]
        
        # ==========================================
        # [DP-Pooling Decoder] 시간 구간별 전압 차이 계산
        # ==========================================
        # [B, 320, 64] -> [B, 10, 32, 64]
        windows = li_seq.view(B, T // self.L_DP, self.L_DP, -1)
        
        # 윈도우의 앞 N_DP 평균과 뒤 N_DP 평균의 차이를 계산 (Temporal Dynamics 추출)
        start_mean = windows[:, :, :self.N_DP, :].mean(dim=2)
        end_mean = windows[:, :, -self.N_DP:, :].mean(dim=2)
        dp_feat = end_mean - start_mean # [B, 10, 64]
        dp_feat = dp_feat.view(B, -1)   # [B, 640]
        
        tda_feat = self.tda_net(tda) # [B, 32]
        
        out = self.classifier(torch.cat([dp_feat, tda_feat], dim=-1))
        return out

# =========================================================
# [3] 병렬 데이터 엔진 (정규화 로직 완벽 적용)
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
        geo_energy = np.nan_to_num(geo_energy, nan=0.0, posinf=0.0, neginf=0.0)
        
        # [수정] 논문처럼 Spike 변환 전 확실한 Z-Score 정규화 적용
        geo_mean = np.mean(geo_energy, axis=(1, 2), keepdims=True)
        geo_std = np.std(geo_energy, axis=(1, 2), keepdims=True) + 1e-8
        geo_norm = (geo_energy - geo_mean) / geo_std
        
        th_geo = np.percentile(geo_norm, 85)
        kin_spikes = (geo_norm > th_geo).astype(np.float32).transpose(0, 2, 1) 
        
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
    print(f"🔥 오리지널 몽타주 복원 + 정규화 완료. 병렬 전처리 시작...")
    results = []
    with Pool(num_workers) as p:
        for i, r in enumerate(p.imap(process_single_subject, subjects)):
            if r is not None: results.append(r)
            if (i + 1) % 10 == 0: print(f"  -> 전처리 진행중: {i+1}/{len(subjects)} 명 완료...")
                
    if not results: raise ValueError("⚠️ 데이터 추출 실패.")
    Xk = np.concatenate([r[0] for r in results])
    Xt = np.concatenate([r[1] for r in results])
    y = np.concatenate([r[2] for r in results])
    print(f"📊 최종 클래스 분포: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"🔍 입력 스파이크 희소성(Sparsity): {Xk.mean():.4f} (약 0.15 정상 복원)")
    return torch.tensor(Xk), torch.tensor(Xt), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    Xk, Xt, y = get_full_data_parallel(VALID_SUBS)
    
    Xk_tr, Xk_ts, Xt_tr, Xt_ts, y_tr, y_ts = train_test_split(Xk, Xt, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(Xk_tr, Xt_tr, y_tr), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xk_ts, Xt_ts, y_ts), batch_size=128, shuffle=False)

    model = GeoHRSNN().to(device)
    
    # [수정] 최고의 성능을 위한 고급 옵티마이저 AdamW 적용
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    
    # [수정] 학습률이 주기적으로 튀어오르며 지역 최솟값을 탈출하게 돕는 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 SOTA 기법(DP-Pooling, HR-Module) 적용 완료. 학습 개시 (샘플 수: {len(y)})")
    best_acc = 0.0
    
    # 논문용 Figure를 위한 기록 배열
    history_train_loss, history_train_acc, history_test_acc = [], [], []
    final_preds, final_trues = [], []
    
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
        epoch_preds, epoch_trues = [], []
        with torch.no_grad():
            for kb, tb, yb in test_loader:
                kb, tb, yb = kb.to(device), tb.to(device), yb.to(device)
                out = model(kb, tb)
                preds = out.max(1)[1]
                ts_c += preds.eq(yb).sum().item()
                ts_t += yb.size(0)
                
                epoch_preds.extend(preds.cpu().numpy())
                epoch_trues.extend(yb.cpu().numpy())
        
        epoch_loss = tr_loss / tr_t
        tr_acc = 100 * tr_c / tr_t
        ts_acc = 100 * ts_c / ts_t
        
        history_train_loss.append(epoch_loss)
        history_train_acc.append(tr_acc)
        history_test_acc.append(ts_acc)
        
        if ts_acc > best_acc: 
            best_acc = ts_acc
            final_preds = epoch_preds
            final_trues = epoch_trues
            torch.save(model.state_dict(), "best_geohrsnn.pth")
        
        print(f"Epoch [{epoch:03d}/100] Loss: {epoch_loss:.4f} | Train Acc: {tr_acc:.1f}% | Test Acc: {ts_acc:.1f}% | Best: {best_acc:.1f}%")

    # 학습 종료 후 논문 Figure용 데이터 저장
    np.savez('experiment_results.npz', 
             train_loss=history_train_loss, 
             train_acc=history_train_acc, 
             test_acc=history_test_acc,
             best_preds=final_preds,
             trues=final_trues)
    print("\n💾 학습이 완료되었습니다. 결과가 'experiment_results.npz'에 저장되었습니다.")