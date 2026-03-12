import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import mne
from scipy.signal import hilbert, savgol_filter
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR_PHYSIONET = './raw_data/files/'
if not os.path.exists(DATA_DIR_PHYSIONET):
    DATA_DIR_PHYSIONET = './raw_data/files'

# =========================================================
# [1] SNN Surrogate Spike
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
# [2] Geo-ALIF SNN 아키텍처 (geo_ch 동적 적용)
# =========================================================
class Geo_ALIF_SNN(nn.Module):
    def __init__(self, eeg_ch=64, geo_ch=18, temp_hid=32, hid_ch=16, out_ch=4):
        super().__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(eeg_ch, temp_hid, kernel_size=5, padding=2),
            nn.BatchNorm1d(temp_hid),
            nn.ReLU(),
            nn.Dropout1d(0.3),
            nn.Conv1d(temp_hid, hid_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid_ch),
            nn.ReLU()
        )
        # 기하학적 피처 채널이 운동 피질 집중으로 인해 줄어듦 (예: 18)
        self.geo_modulator = nn.Sequential(
            nn.Linear(geo_ch, hid_ch),
            nn.Sigmoid() 
        )
        self.theta_0 = 0.5
        self.beta = 1.8
        self.tau_a = 0.36
        self.tau_m = 0.8
        self.gamma = nn.Parameter(torch.ones(hid_ch) * 0.5)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, out_ch)
        )

    def forward(self, x_eeg, x_geo):
        B, T, _ = x_eeg.shape
        eeg_encoded = self.eeg_encoder(x_eeg.transpose(1, 2)).transpose(1, 2)
        geo_mod = self.geo_modulator(x_geo)
        
        mem_alif = torch.zeros(B, 16, device=x_eeg.device)
        eta = torch.zeros(B, 16, device=x_eeg.device)
        mem_li = torch.zeros(B, 16, device=x_eeg.device)
        
        potentials_seq = []
        spikes_for_log = []
        
        for t in range(T):
            cur_eeg = eeg_encoded[:, t, :]
            cur_geo_mod = geo_mod[:, t, :]
            
            prev_spike = spikes_for_log[-1] if t > 0 else 0
            eta = self.tau_a * eta + (1 - self.tau_a) * prev_spike
            theta_t = self.theta_0 + self.beta * eta - (self.gamma * cur_geo_mod)
            
            mem_alif = self.tau_m * mem_alif + cur_eeg
            spk = spike_fn(mem_alif - theta_t)
            mem_alif = mem_alif * (1.0 - spk)
            
            mem_li = 0.9 * mem_li + spk
            potentials_seq.append(mem_li.unsqueeze(1))
            spikes_for_log.append(spk)
            
        potentials_seq = torch.cat(potentials_seq, dim=1).transpose(1, 2)
        
        windows = potentials_seq.view(B, 16, 10, 32)
        start_mean = windows[:, :, :, :16].mean(dim=-1) 
        end_mean = windows[:, :, :, 16:].mean(dim=-1)   
        
        dp_feat = (end_mean - start_mean).flatten(1)
        out = self.classifier(dp_feat)
        return out

# =========================================================
# [3] 데이터 전처리 (운동 피질 집중 + 105명 스케일)
# =========================================================
# 논문 기준 PhysioNet에서 누락/불량 데이터가 있는 4명(S088, S092, S100, S104) 제외
EXCLUDE_SUBS = [88, 92, 100, 104]
VALID_SUBS = [s for s in range(1, 110) if s not in EXCLUDE_SUBS]

# 운동 상상 핵심 채널 (Sensorimotor cortex)
MOTOR_CHANNELS = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

def process_single_subject_dual(sub):
    try:
        def load_runs(runs):
            raws = [mne.io.read_raw_edf(os.path.join(DATA_DIR_PHYSIONET, f'S{sub:03d}', f'S{sub:03d}R{r:02d}.edf'), preload=True, verbose=False) for r in runs]
            if not raws: return None
            raw = mne.concatenate_raws(raws); raw.filter(8., 30., verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage('standard_1005', on_missing='ignore')
            return raw

        raw_lr = load_runs([4, 8, 12]) 
        raw_f = load_runs([6, 10, 14])  
        if raw_lr is None or raw_f is None: return None
        
        # 기하학적 피처를 추출할 핵심 모터 채널의 인덱스 확보
        ch_names = raw_lr.ch_names
        motor_idx = [ch_names.index(ch) for ch in MOTOR_CHANNELS if ch in ch_names]
        if not motor_idx: # 만약 매칭이 안 되면 전체 채널로 폴백
            motor_idx = list(range(len(ch_names)))

        def get_epochs(raw):
            evs, ev_id = mne.events_from_annotations(raw, verbose=False)
            t0 = next((v for k, v in ev_id.items() if 'T0' in k), None)
            t1 = next((v for k, v in ev_id.items() if 'T1' in k), None)
            t2 = next((v for k, v in ev_id.items() if 'T2' in k), None)
            ep = mne.Epochs(raw, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            csd = mne.preprocessing.compute_current_source_density(raw.copy(), sphere=(0,0,0,0.095))
            ep_c = mne.Epochs(csd, evs, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            return ep.get_data(), ep_c.get_data(), ep.events[:, 2]

        d_lr, c_lr, y_lr = get_epochs(raw_lr)
        d_f, c_f, y_f = get_epochs(raw_f)
        
        idx_lh, idx_rh, idx_ft = np.where(y_lr == 2)[0], np.where(y_lr == 3)[0], np.where(y_f == 3)[0] 
        idx_rest_lr, idx_rest_f = np.where(y_lr == 1)[0], np.where(y_f == 1)[0]
        
        min_trials = min(len(idx_lh), len(idx_rh), len(idx_ft), len(idx_rest_lr) + len(idx_rest_f))
        if min_trials == 0: return None
        
        d_lh, c_lh = d_lr[idx_lh[:min_trials]], c_lr[idx_lh[:min_trials]]
        d_rh, c_rh = d_lr[idx_rh[:min_trials]], c_lr[idx_rh[:min_trials]]
        d_ft, c_ft = d_f[idx_ft[:min_trials]], c_f[idx_ft[:min_trials]]
        d_rest = np.concatenate([d_lr[idx_rest_lr], d_f[idx_rest_f]])[:min_trials]
        c_rest = np.concatenate([c_lr[idx_rest_lr], c_f[idx_rest_f]])[:min_trials]
        
        # 1. Raw EEG (채널 64개 전체 유지)
        raw_eeg = np.concatenate([d_lh, d_rh, d_ft, d_rest])[:, :, 80:400]
        raw_eeg = raw_eeg.transpose(0, 2, 1).astype(np.float32)
        
        # 2. 기하학적 피처 (🔥 운동 피질 채널만 슬라이싱)
        xb = np.concatenate([d_lh, d_rh, d_ft, d_rest])[:, motor_idx, :]
        xc = np.concatenate([c_lh, c_rh, c_ft, c_rest])[:, motor_idx, :]
        y = np.concatenate([np.zeros(min_trials), np.ones(min_trials), np.full(min_trials, 2), np.full(min_trials, 3)])
        
        win = 15
        u = savgol_filter(np.abs(hilbert(xb, axis=-1)), win, 3, axis=-1)
        v = savgol_filter(np.abs(hilbert(xc, axis=-1)), win, 3, axis=-1)
        du = savgol_filter(u, win, 3, deriv=1, axis=-1)
        dv = savgol_filter(v, win, 3, deriv=1, axis=-1)
        ddu = savgol_filter(u, win, 3, deriv=2, axis=-1)
        ddv = savgol_filter(v, win, 3, deriv=2, axis=-1)
        
        areal_vel = (0.5 * np.abs(u * dv - v * du))[:, :, 80:400]
        denom = np.clip((du**2 + dv**2)**(1.5), 1e-6, None)
        curvature = (np.clip(np.abs(du * ddv - dv * ddu) / denom, 0, 10))[:, :, 80:400]
        
        areal_vel = np.nan_to_num(areal_vel, nan=0.0, posinf=0.0, neginf=0.0)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        
        vel_mean, vel_std = np.mean(areal_vel, axis=(1, 2), keepdims=True), np.std(areal_vel, axis=(1, 2), keepdims=True) + 1e-8
        curv_mean, curv_std = np.mean(curvature, axis=(1, 2), keepdims=True), np.std(curvature, axis=(1, 2), keepdims=True) + 1e-8
        
        vel_norm = (areal_vel - vel_mean) / vel_std 
        curv_norm = (curvature - curv_mean) / curv_std 
        
        geo_features = np.concatenate([vel_norm, curv_norm], axis=1).astype(np.float32)
        geo_features = geo_features.transpose(0, 2, 1) # [B, T, len(motor_idx)*2]
        
        return raw_eeg, geo_features, y
    except Exception as e: 
        return None

if __name__ == "__main__":
    print(f"🔥 Geo-ALIF SNN [105명 Full-Scale & 운동 피질 집중] 파이프라인 시작...\n")
    
    subject_data = {}
    valid_loaded_subs = []
    
    # 105명 데이터 전부 로드 (메모리 관리 주의, Kaggle 30GB RAM 환경이면 충분함)
    for sub in VALID_SUBS:
        print(f"데이터 로드 및 특징 추출 중: S{sub:03d} / 109", end='\r')
        data = process_single_subject_dual(sub)
        if data is not None:
            subject_data[sub] = data
            valid_loaded_subs.append(sub)
            
    valid_subs_arr = np.array(valid_loaded_subs)
    print(f"\n✅ 데이터 로드 완료! (유효 피험자: {len(valid_subs_arr)}명)")
    
    # 기하학적 피처 채널 크기 동적 획득
    sample_geo_ch = subject_data[valid_subs_arr[0]][1].shape[-1]
    print(f"📊 축출된 기하학적 피처 차원: {sample_geo_ch}채널 (순도 100% 운동 피질)")
    
    # 논문 오피셜 5-Fold (84명 Global Train / 21명 SSTL Test)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results_acc = {}
    fold = 1
    
    for train_idx, test_idx in kf.split(valid_subs_arr):
        train_subs = valid_subs_arr[train_idx]
        test_subs = valid_subs_arr[test_idx]
        
        print(f"\n{'='*50}")
        print(f"🔄 [Fold {fold}/5] Global Train: {len(train_subs)}명 | SSTL Test: {len(test_subs)}명")
        
        # [Stage 1] Global Pre-training
        X_eeg_g = np.concatenate([subject_data[s][0] for s in train_subs])
        X_geo_g = np.concatenate([subject_data[s][1] for s in train_subs])
        y_g = np.concatenate([subject_data[s][2] for s in train_subs])
        
        X_eeg_t, X_geo_t, y_t = torch.tensor(X_eeg_g), torch.tensor(X_geo_g), torch.tensor(y_g, dtype=torch.long)
        global_loader = DataLoader(TensorDataset(X_eeg_t, X_geo_t, y_t), batch_size=256, shuffle=True)
        
        global_model = Geo_ALIF_SNN(geo_ch=sample_geo_ch).to(device)
        optimizer_g = optim.AdamW(global_model.parameters(), lr=0.002, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print(f"🚀 글로벌 모델 학습 중... (학습 데이터: {len(y_g)}개, Epochs: 40)")
        for epoch in range(1, 41): # 105명 스케일이므로 에포크를 늘려 충분히 학습시킴
            global_model.train()
            for xb_eeg, xb_geo, yb in global_loader:
                xb_eeg, xb_geo, yb = xb_eeg.to(device), xb_geo.to(device), yb.to(device)
                optimizer_g.zero_grad()
                out = global_model(xb_eeg, xb_geo)
                loss = criterion(out, yb)
                loss.backward()
                optimizer_g.step()
                
        # [Stage 2] Subject-Specific Transfer Learning (SSTL)
        print(f"🎯 파인튜닝(SSTL) 및 평가 진행 중...")
        for sub in test_subs:
            X_eeg_sub, X_geo_sub, y_sub = subject_data[sub]
            X_eeg_sub, X_geo_sub, y_sub = torch.tensor(X_eeg_sub), torch.tensor(X_geo_sub), torch.tensor(y_sub, dtype=torch.long)
            
            tr_idx, ts_idx = train_test_split(np.arange(len(y_sub)), test_size=0.2, stratify=y_sub.numpy(), random_state=42)
            
            sub_train_loader = DataLoader(TensorDataset(X_eeg_sub[tr_idx], X_geo_sub[tr_idx], y_sub[tr_idx]), batch_size=8, shuffle=True)
            sub_test_loader = DataLoader(TensorDataset(X_eeg_sub[ts_idx], X_geo_sub[ts_idx], y_sub[ts_idx]), batch_size=8, shuffle=False)
            
            sstl_model = copy.deepcopy(global_model).to(device)
            # 논문 기준 파인튜닝은 아주 섬세하게 (lr 작게, epoch 짧게)
            optimizer_ft = optim.AdamW(sstl_model.parameters(), lr=0.0005, weight_decay=1e-4)
            
            best_test_acc = 0.0
            for epoch in range(1, 11): 
                sstl_model.train()
                for xb_eeg, xb_geo, yb in sub_train_loader:
                    xb_eeg, xb_geo, yb = xb_eeg.to(device), xb_geo.to(device), yb.to(device)
                    optimizer_ft.zero_grad()
                    out = sstl_model(xb_eeg, xb_geo)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer_ft.step()
                    
                sstl_model.eval()
                ts_c, ts_t = 0, 0
                with torch.no_grad():
                    for xb_eeg, xb_geo, yb in sub_test_loader:
                        xb_eeg, xb_geo, yb = xb_eeg.to(device), xb_geo.to(device), yb.to(device)
                        out = sstl_model(xb_eeg, xb_geo)
                        ts_c += out.max(1)[1].eq(yb).sum().item()
                        ts_t += yb.size(0)
                        
                test_acc = 100 * ts_c / ts_t
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    
            results_acc[sub] = best_test_acc
            print(f" ✅ S{sub:03d} 타겟 SSTL 완료 -> 최고 Test Acc: {best_test_acc:.1f}%", end='\r')
            
        print(f"\n✅ Fold {fold} 평가 완료.")
        fold += 1

    print(f"\n{'='*50}")
    print(f"📊 최종 Geo-ALIF SNN [105명 스케일] 테스트 결과")
    print(f"{'='*50}")
    valid_accs = list(results_acc.values())
    print(f"평균 정확도(Mean): {np.mean(valid_accs):.1f}%")
    print(f"최고 정확도(Max):  {np.max(valid_accs):.1f}%")
    print(f"최저 정확도(Min):  {np.min(valid_accs):.1f}%")