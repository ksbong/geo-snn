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
# [2] Temporal Conv + HR-SNN 구조
# =========================================================
class GeoHRSNN(nn.Module):
    # temp_hid를 128로 늘려 전극당 2개의 독립적인 시간 특징을 뽑아냄
    def __init__(self, in_ch=64, temp_hid=128, hid_ch=64, out_ch=4, L_DP=32, N_DP=16):
        super().__init__()
        self.L_DP = L_DP 
        self.N_DP = N_DP 
        
        # [오버피팅 방지 핵심 1] Depthwise 1D Conv (채널 독립적 시간 패턴 추출)
        # groups=in_ch 로 설정하여 64개 채널이 절대 섞이지 않고 각자의 시간 흐름만 분석함!
        self.temp_conv = nn.Sequential(
            nn.Conv1d(in_ch, temp_hid, kernel_size=15, padding=7, groups=in_ch),
            nn.BatchNorm1d(temp_hid),
            nn.ReLU(),
            nn.Dropout(0.4) # 시간 특징에 강한 드롭아웃을 걸어 암기 방지
        )
        
        # [오버피팅 방지 핵심 2] 공간 매핑 (Pointwise 역할)
        # 여기서 비로소 각 전극의 시간 정보를 종합하여 공간 특징을 만들어냄
        self.spatial_fc = nn.Linear(temp_hid, hid_ch)
        nn.init.normal_(self.spatial_fc.weight, mean=0.05, std=0.1) 
        
        self.li_fc = nn.Linear(hid_ch * 2, hid_ch)
        
        self.tda_net = nn.Sequential(
            nn.Linear(150, 64), nn.ReLU(),
            nn.Dropout(0.4), # TDA도 피험자 특성을 타기 쉬우므로 드롭아웃 강화
            nn.Linear(64, 32), nn.ReLU()
        )
        
        self.dp_out_dim = (320 // L_DP) * hid_ch
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # 빡센 정규화
            nn.Linear(self.dp_out_dim + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_ch)
        )

    def forward(self, x, tda):
        B, T, _ = x.shape
        
        # 입력 스파이크 자체에도 임의의 노이즈(드롭아웃)를 살짝 줘서 센서 노이즈에 대한 강건함 부여
        x = nn.functional.dropout(x, p=0.1, training=self.training)
        
        x_t = x.transpose(1, 2)
        x_t = self.temp_conv(x_t) 
        x_t = x_t.transpose(1, 2) 
        
        mem_fast = torch.zeros(B, 64, device=x.device)
        mem_slow = torch.zeros(B, 64, device=x.device)
        mem_li = torch.zeros(B, 64, device=x.device)
        li_seq = []
        
        for t in range(T):
            cur = self.spatial_fc(x_t[:, t, :])
            
            mem_fast = 0.5 * mem_fast + cur
            spk_fast = spike_fn(mem_fast - 0.5)
            mem_fast = mem_fast * (1.0 - spk_fast)
            
            mem_slow = 0.9 * mem_slow + cur
            spk_slow = spike_fn(mem_slow - 1.0)
            mem_slow = mem_slow * (1.0 - spk_slow)
            
            spk_cat = torch.cat([spk_fast, spk_slow], dim=-1)
            
            cur_li = self.li_fc(spk_cat)
            mem_li = 0.9 * mem_li + cur_li
            li_seq.append(mem_li.unsqueeze(1))
            
        li_seq = torch.cat(li_seq, dim=1)
        
        windows = li_seq.view(B, T // self.L_DP, self.L_DP, -1)
        start_mean = windows[:, :, :self.N_DP, :].mean(dim=2)
        end_mean = windows[:, :, -self.N_DP:, :].mean(dim=2)
        dp_feat = end_mean - start_mean 
        dp_feat = dp_feat.view(B, -1)   
        
        tda_feat = self.tda_net(tda) 
        out = self.classifier(torch.cat([dp_feat, tda_feat], dim=-1))
        return out
    
# =========================================================
# [3] 병렬 데이터 엔진 (논문 방식대로 피험자 ID 함께 반환)
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
        
        # 논문 평가를 위한 피험자 ID 배열
        subjects_arr = np.full(len(y), sub)

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

        return kin_spikes, tda_final, y, subjects_arr
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
    subs = np.concatenate([r[3] for r in results])
    print(f"📊 최종 클래스 분포: {dict(zip(*np.unique(y, return_counts=True)))}")
    return torch.tensor(Xk), torch.tensor(Xt), torch.tensor(y, dtype=torch.long), subs

if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    Xk, Xt, y, subs = get_full_data_parallel(VALID_SUBS)
    
    # [핵심] 논문의 Global Training 방식: 피험자 기준으로 80:20 완전히 분할
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
    # [수정] 튀어오르는 현상을 없애기 위해 부드러운 코사인 스케줄러로 변경
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 피험자 독립 분할(Global Training) 방식 적용 완료. 학습 개시")
    print(f"Train 피험자: {len(train_subs)}명, Test 피험자: {len(test_subs)}명")
    best_acc = 0.0
    
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

    np.savez('experiment_results.npz', 
             train_loss=history_train_loss, 
             train_acc=history_train_acc, 
             test_acc=history_test_acc,
             best_preds=final_preds,
             trues=final_trues)
    print("\n💾 학습이 완료되었습니다. 결과가 'experiment_results.npz'에 저장되었습니다.")