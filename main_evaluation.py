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

class RobustGeoSNN(nn.Module):
    def __init__(self, num_eeg_channels=64, hidden_dim=128, num_classes=4):
        super().__init__()
        
        # [복원] 네가 예전에 썼던 CNN 기반 공간/시간 특징 추출기 (진짜 뇌 주름 생성)
        # 시간축(T)을 따라 1D Conv를 돌려서 희소한 스파이크에서 풍부한 패턴 전류를 뽑아냄
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(num_eeg_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim), # CNN에는 BN을 써도 스파이크 뇌사가 안 일어남!
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # [심층 SNN] 1층짜리 쓰레기 버리고 2단 Deep SNN으로 변경
        self.snn_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.snn_layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        # TDA 독립 처리 층
        self.tda_net = nn.Sequential(
            nn.Linear(150, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 앙상블 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(128, num_classes)
        )

    def forward(self, kin_spikes_seq, tda_features):
        B, T, _ = kin_spikes_seq.shape
        
        # CNN 처리를 위해 차원 변경: [B, T, 64] -> [B, 64, T]
        x_cnn = kin_spikes_seq.transpose(1, 2)
        
        # 희소 스파이크 -> 풍부한 연속 전류로 변환
        current_features = self.cnn_extractor(x_cnn) # [B, 128, T]
        
        # 다시 SNN 입력을 위해 시간축 복구: [B, 128, T] -> [B, T, 128]
        current_seq = current_features.transpose(1, 2)
        
        # 1층, 2층 SNN 전압 독립 관리
        mem1 = torch.zeros(B, 128, device=kin_spikes_seq.device)
        mem2 = torch.zeros(B, 128, device=kin_spikes_seq.device)
        spike_counts = torch.zeros(B, 128, device=kin_spikes_seq.device)
        
        for t in range(T):
            # 1층 SNN
            cur1 = self.snn_layer1(current_seq[:, t, :])
            mem1 = 0.9 * mem1 + cur1
            spk1 = spike_fn(mem1 - 1.0)
            mem1 = mem1 * (1.0 - spk1)
            
            # 2층 SNN (1층의 스파이크를 입력으로 받음)
            cur2 = self.snn_layer2(spk1)
            mem2 = 0.9 * mem2 + cur2
            spk2 = spike_fn(mem2 - 1.0)
            mem2 = mem2 * (1.0 - spk2)
            
            spike_counts += spk2
            
        # 발화율 정규화
        firing_rate = spike_counts / T
        
        # TDA 병합 및 최종 분류
        tda_out = self.tda_net(tda_features)
        fused_features = torch.cat([firing_rate, tda_out], dim=1) 
        out = self.classifier(fused_features)
        
        return out, spike_counts
    
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
            
            # [결자해지] 회원님의 오리지널 코드로 복원 (대문자 강제 변환 삭제)
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
            
            # [결자해지] 이제 몽타주가 정상이므로 CSD가 절대 에러를 뿜지 않음
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
    print(f"🔥 오리지널 몽타주 복원 완료. {num_workers}개 코어 병렬 전처리 시작...")
    
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
    print(f"🔍 입력 스파이크 희소성(Sparsity): {Xk.mean():.4f} (약 0.15 정상 복원)")
    return torch.tensor(Xk), torch.tensor(Xt), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    VALID_SUBS = [s for s in range(1, 110) if s not in [88, 92, 100, 104]]
    Xk, Xt, y = get_full_data_parallel(VALID_SUBS)
    
    Xk_tr, Xk_ts, Xt_tr, Xt_ts, y_tr, y_ts = train_test_split(Xk, Xt, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(Xk_tr, Xt_tr, y_tr), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xk_ts, Xt_ts, y_ts), batch_size=128, shuffle=False)

    model = RobustGeoSNN().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4) 
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 결자해지: 스파이크 활성화 모드로 학습 개시 (샘플 수: {len(y)})")
    best_acc = 0.0
    
    for epoch in range(1, 101):
        model.train()
        tr_loss, tr_c, tr_t = 0.0, 0, 0
        total_spikes = 0.0
        all_preds = []
        
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
        
        pred_unique, pred_counts = np.unique(all_preds, return_counts=True)
        pred_dist = {int(k): int(v) for k, v in zip(pred_unique, pred_counts)}
        avg_spikes = total_spikes / (tr_t * 128) 
        
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
        
        print(f"Epoch [{epoch:03d}/100] Loss: {epoch_loss:.4f} | Train Acc: {tr_acc:.1f}% | Test Acc: {ts_acc:.1f}% | Avg Spikes: {avg_spikes:.2f} | Pred Dist: {pred_dist}")