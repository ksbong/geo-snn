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
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR_PHYSIONET = './raw_data/files'

# =========================================================
# [1] SNN 미분 불가능성 해결: 대리 기울기(Surrogate Gradient)
# =========================================================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * torch.abs(input)) ** 2
        return grad_output * sg

spike_fn = SurrogateSpike.apply

# =========================================================
# [2] Ultimate Geo-SNN (PhysioNet 64채널 -> 4클래스 억제망)
# =========================================================
class PhysioNetGeoLIF_4Class(nn.Module):
    def __init__(self, num_eeg_channels=64, num_classes=4, leak=0.8, base_thresh=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.leak = leak
        self.base_thresh = base_thresh
        
        # 64채널 스파이크를 4개 클래스 허브로 공간 투영
        self.spatial_weights = nn.Linear(num_eeg_channels, num_classes, bias=False)
        
        # 4-Way 측면 억제망 (0:Rest, 1:Left, 2:Right, 3:Feet)
        init_inhibition = torch.tensor([
            [ 1.0, -0.4, -0.4, -0.4], # Rest는 모든 운동을 억제
            [-0.4,  1.0, -0.3, -0.3], 
            [-0.4, -0.3,  1.0, -0.3], 
            [-0.4, -0.3, -0.3,  1.0]  
        ])
        self.lateral_weights = nn.Parameter(init_inhibition, requires_grad=True)
        
        # TDA 모듈레이션 (사전지식 기반 임계값 조절)
        self.tda_to_thresh = nn.Linear(50, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        batch_size, time_steps, _ = kin_spikes_seq.shape
        mem = torch.zeros(batch_size, self.num_classes, device=kin_spikes_seq.device)
        
        # Zero-cost TDA 임계값 변조 (시간 루프 밖에서 단 1번 연산)
        tda_modulation = torch.sigmoid(self.tda_to_thresh(tda_features)) * 0.5
        dynamic_thresh = self.base_thresh + tda_modulation 
        
        spike_records = []
        
        # LIF 뉴런 시간축 적분 및 발화
        for t in range(time_steps):
            x_t = kin_spikes_seq[:, t, :] 
            projected = self.spatial_weights(x_t)
            inhibited = torch.matmul(projected, self.lateral_weights)
            
            mem = self.leak * mem + inhibited
            spikes_out = spike_fn(mem - dynamic_thresh)
            mem = mem * (1.0 - spikes_out) # 발화 후 리셋
            
            spike_records.append(spikes_out)
            
        return torch.stack(spike_records, dim=1) # (Batch, Time, 4)

# =========================================================
# [3] 로컬 데이터 로딩 함수
# =========================================================
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")
    return file_path

def extract_real_physionet_data_local(subjects):
    print(f"⏳ 로컬 파일({DATA_DIR_PHYSIONET})에서 {len(subjects)}명 데이터 읽기 시작...")
    X_kin_list, X_tda_list, y_list = [], [], []
    
    for sub in subjects:
        try:
            # --- 1. Left / Right 상상 런 (Runs 4, 8, 12) ---
            runs_LR = [4, 8, 12]
            raws_LR = []
            for r in runs_LR:
                path = get_local_file_path(sub, r)
                raws_LR.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
                
            raw_LR = mne.concatenate_raws(raws_LR)
            raw_LR.filter(8., 30., verbose=False)
            mne.datasets.eegbci.standardize(raw_LR)
            raw_LR.set_montage('standard_1005', match_case=False, on_missing='ignore')
            raw_LR_csd = mne.preprocessing.compute_current_source_density(raw_LR.copy(), sphere=(0, 0, 0, 0.095))
            
            events_LR, _ = mne.events_from_annotations(raw_LR, verbose=False)
            ep_b_LR = mne.Epochs(raw_LR, events_LR, event_id={'T0':1, 'T1':2, 'T2':3}, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c_LR = mne.Epochs(raw_LR_csd, events_LR, event_id={'T0':1, 'T1':2, 'T2':3}, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            
            X_b_LR, X_c_LR = ep_b_LR.get_data(), ep_c_LR.get_data()
            y_LR = ep_b_LR.events[:, 2] - 1 # 0(Rest), 1(Left), 2(Right)
            
            # --- 2. Feet 상상 런 (Runs 6, 10, 14) ---
            runs_F = [6, 10, 14]
            raws_F = []
            for r in runs_F:
                path = get_local_file_path(sub, r)
                raws_F.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
                
            raw_F = mne.concatenate_raws(raws_F)
            raw_F.filter(8., 30., verbose=False)
            mne.datasets.eegbci.standardize(raw_F)
            raw_F.set_montage('standard_1005', match_case=False, on_missing='ignore')
            raw_F_csd = mne.preprocessing.compute_current_source_density(raw_F.copy(), sphere=(0, 0, 0, 0.095))
            
            events_F, _ = mne.events_from_annotations(raw_F, verbose=False)
            ep_b_F = mne.Epochs(raw_F, events_F, event_id={'T0':1, 'T2':3}, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c_F = mne.Epochs(raw_F_csd, events_F, event_id={'T0':1, 'T2':3}, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            
            X_b_F, X_c_F = ep_b_F.get_data(), ep_c_F.get_data()
            y_F = ep_b_F.events[:, 2]
            y_F = np.where(y_F == 1, 0, 3) # 1은 0(Rest)으로, 3은 3(Feet)으로
            
            X_b = np.concatenate([X_b_LR, X_b_F], axis=0)
            X_c = np.concatenate([X_c_LR, X_c_F], axis=0)
            y = np.concatenate([y_LR, y_F], axis=0)
            
            # --- 3. 기하학적 피처 및 희소 스파이크 인코딩 ---
            win = 51
            u = savgol_filter(np.abs(hilbert(X_b, axis=-1)), win, 3, axis=-1)
            v = savgol_filter(np.abs(hilbert(X_c, axis=-1)), win, 3, axis=-1)
            
            du = savgol_filter(u, win, 3, deriv=1, axis=-1)
            dv = savgol_filter(v, win, 3, deriv=1, axis=-1)
            ddu = savgol_filter(u, win, 3, deriv=2, axis=-1)
            ddv = savgol_filter(v, win, 3, deriv=2, axis=-1)
            
            areal_vel = 0.5 * np.abs(u * dv - v * du)[:, :, 80:400]
            denom = np.clip((du**2 + dv**2)**(1.5), 1e-6, None)
            curvature = np.clip(np.abs(du * ddv - dv * ddu) / denom, 0, 10)[:, :, 80:400]
            
            geo_energy = areal_vel * curvature
            th_geo = np.percentile(geo_energy, 90) # 상위 10%
            kin_spikes = (geo_energy > th_geo).astype(np.float32)
            kin_spikes = np.transpose(kin_spikes, (0, 2, 1)) # (Trials, Time, 64)
            
            # --- 4. TDA 추출 ---
            ch_names = ep_b_LR.ch_names
            c3 = next(i for i, ch in enumerate(ch_names) if ch.upper() == 'C3')
            cz = next(i for i, ch in enumerate(ch_names) if ch.upper() == 'CZ')
            c4 = next(i for i, ch in enumerate(ch_names) if ch.upper() == 'C4')
            
            point_clouds = np.stack([u[:, c3, :], v[:, c3, :], u[:, cz, :], v[:, cz, :], u[:, c4, :], v[:, c4, :]], axis=-1)
            VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
            PL = PersistenceLandscape(n_bins=50)
            tda_features = PL.fit_transform(VR.fit_transform(point_clouds)).reshape(len(y), -1).astype(np.float32)
            
            X_kin_list.append(kin_spikes)
            X_tda_list.append(tda_features)
            y_list.append(y)
            print(f"✅ Sub {sub} 추출 완료")
            
        except Exception as e:
            print(f"❌ Sub {sub} 에러: {e}")
            
    if not X_kin_list:
        raise ValueError("데이터 추출 실패! 로컬 경로에 파일이 있는지 확인해줘.")
        
    return np.concatenate(X_kin_list), np.concatenate(X_tda_list), np.concatenate(y_list)

# =========================================================
# [4] 실행 및 검증 (간이 테스트)
# =========================================================
if __name__ == "__main__":
    print(f"🛠️ 디바이스 세팅: {device}")
    
    # 1번 피험자로 간이 테스트
    subjects_to_test = [1]
    X_kin_all, X_tda_all, y_all = extract_real_physionet_data_local(subjects_to_test)

    # 80:20 스플릿
    X_k_train, X_k_val, X_t_train, X_t_val, y_train, y_val = train_test_split(
        X_kin_all, X_tda_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    train_dataset = TensorDataset(torch.tensor(X_k_train), torch.tensor(X_t_train), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_k_val), torch.tensor(X_t_val), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = PhysioNetGeoLIF_4Class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 간이 로컬 데이터 모델 학습 시작 (Train: {len(y_train)}개, Val: {len(y_val)}개)")

    for epoch in range(15):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for kin_batch, tda_batch, target in train_loader:
            kin_batch, tda_batch, target = kin_batch.to(device), tda_batch.to(device), target.to(device)
            optimizer.zero_grad()
            
            spike_out = model(kin_batch, tda_batch)
            spike_sum = spike_out.sum(dim=1) 
            
            loss = criterion(spike_sum, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = spike_sum.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        train_acc = 100. * correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for kin_batch, tda_batch, target in val_loader:
                kin_batch, tda_batch, target = kin_batch.to(device), tda_batch.to(device), target.to(device)
                spike_out = model(kin_batch, tda_batch)
                spike_sum = spike_out.sum(dim=1)
                
                _, predicted = spike_sum.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/15] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("\n🎉 간이 테스트 종료. 에러 없이 돌아가면 스케일업 준비 완료!")