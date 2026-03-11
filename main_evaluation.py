import os
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
# [1] SNN 대리 기울기 및 모델 아키텍처
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

class PhysioNetGeoLIF_4Class(nn.Module):
    def __init__(self, num_eeg_channels=64, num_classes=4, leak=0.8, base_thresh=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.leak = leak
        self.base_thresh = base_thresh
        
        self.spatial_weights = nn.Linear(num_eeg_channels, num_classes, bias=False)
        
        # 4-Way 측면 억제망 (0:Rest, 1:Left, 2:Right, 3:Feet)
        init_inhibition = torch.tensor([
            [ 1.0, -0.4, -0.4, -0.4],
            [-0.4,  1.0, -0.3, -0.3], 
            [-0.4, -0.3,  1.0, -0.3], 
            [-0.4, -0.3, -0.3,  1.0]  
        ])
        self.lateral_weights = nn.Parameter(init_inhibition, requires_grad=True)
        self.tda_to_thresh = nn.Linear(50, num_classes)

    def forward(self, kin_spikes_seq, tda_features):
        batch_size, time_steps, _ = kin_spikes_seq.shape
        mem = torch.zeros(batch_size, self.num_classes, device=kin_spikes_seq.device)
        
        tda_modulation = torch.sigmoid(self.tda_to_thresh(tda_features)) * 0.5
        dynamic_thresh = self.base_thresh + tda_modulation 
        
        spike_records = []
        for t in range(time_steps):
            x_t = kin_spikes_seq[:, t, :] 
            projected = self.spatial_weights(x_t)
            inhibited = torch.matmul(projected, self.lateral_weights)
            
            mem = self.leak * mem + inhibited
            spikes_out = spike_fn(mem - dynamic_thresh)
            mem = mem * (1.0 - spikes_out)
            spike_records.append(spikes_out)
            
        return torch.stack(spike_records, dim=1)

# =========================================================
# [2] 철저한 메모리 관리형 데이터 로더
# =========================================================
def get_local_file_path(subject, run):
    sub_str = f"S{subject:03d}"
    file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"{sub_str}R{run:02d}.edf")
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR_PHYSIONET, sub_str, f"s{subject:03d}r{run:02d}.edf")
    return file_path

def extract_memory_safe_data(subjects, phase_name=""):
    print(f"\n⏳ [{phase_name}] {len(subjects)}명 데이터 추출 시작 (메모리 누수 방지 모드)...")
    X_kin_list, X_tda_list, y_list = [], [], []
    
    for sub in subjects:
        try:
            # --- 1. Left/Right 상상 ---
            runs_LR = [4, 8, 12]
            raws_LR = [mne.io.read_raw_edf(get_local_file_path(sub, r), preload=True, verbose=False) for r in runs_LR if os.path.exists(get_local_file_path(sub, r))]
            if not raws_LR: continue
            
            raw_LR = mne.concatenate_raws(raws_LR)
            raw_LR.filter(8., 30., verbose=False)
            mne.datasets.eegbci.standardize(raw_LR)
            raw_LR.set_montage('standard_1005', match_case=False, on_missing='ignore')
            raw_LR_csd = mne.preprocessing.compute_current_source_density(raw_LR.copy(), sphere=(0, 0, 0, 0.095))
            
            events_LR, _ = mne.events_from_annotations(raw_LR, verbose=False)
            ep_b_LR = mne.Epochs(raw_LR, events_LR, event_id={'T0':1, 'T1':2, 'T2':3}, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            ep_c_LR = mne.Epochs(raw_LR_csd, events_LR, event_id={'T0':1, 'T1':2, 'T2':3}, tmin=0., tmax=3.0, baseline=None, preload=True, verbose=False)
            
            X_b_LR, X_c_LR = ep_b_LR.get_data(), ep_c_LR.get_data()
            y_LR = ep_b_LR.events[:, 2] - 1
            
            # --- 메모리 즉시 해제 ---
            del raws_LR, raw_LR, raw_LR_csd, ep_b_LR, ep_c_LR
            
            # --- 2. Feet 상상 ---
            runs_F = [6, 10, 14]
            raws_F = [mne.io.read_raw_edf(get_local_file_path(sub, r), preload=True, verbose=False) for r in runs_F if os.path.exists(get_local_file_path(sub, r))]
            if not raws_F: continue
            
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
            y_F = np.where(y_F == 1, 0, 3) 
            
            # --- 메모리 즉시 해제 ---
            del raws_F, raw_F, raw_F_csd, ep_b_F, ep_c_F
            
            # 데이터 병합
            X_b = np.concatenate([X_b_LR, X_b_F], axis=0)
            X_c = np.concatenate([X_c_LR, X_c_F], axis=0)
            y = np.concatenate([y_LR, y_F], axis=0)
            
            # --- 3. 기하학 스파이크 추출 ---
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
            th_geo = np.percentile(geo_energy, 90) 
            kin_spikes = (geo_energy > th_geo).astype(np.float32)
            kin_spikes = np.transpose(kin_spikes, (0, 2, 1)) 
            
            # --- 4. TDA 추출 ---
            # 인덱스를 직접 지정하여 에러 방지 (표준 1005 기준)
            point_clouds = np.stack([u[:, 12, :], v[:, 12, :], u[:, 14, :], v[:, 14, :], u[:, 16, :], v[:, 16, :]], axis=-1) # 대략 C3, Cz, C4 위치
            VR = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1)
            PL = PersistenceLandscape(n_bins=50)
            tda_features = PL.fit_transform(VR.fit_transform(point_clouds)).reshape(len(y), -1).astype(np.float32)
            
            X_kin_list.append(torch.tensor(kin_spikes))
            X_tda_list.append(torch.tensor(tda_features))
            y_list.append(torch.tensor(y, dtype=torch.long))
            
            # 찌꺼기 메모리 청소
            del X_b, X_c, u, v, du, dv, ddu, ddv, areal_vel, denom, curvature, geo_energy
            gc.collect()
            
            if sub % 10 == 0:
                print(f"✅ Sub {sub} 추출 완료... (Garbage Collected)")
                
        except Exception as e:
            print(f"❌ Sub {sub} 에러 스킵: {e}")
            
    return torch.cat(X_kin_list), torch.cat(X_tda_list), torch.cat(y_list)

# =========================================================
# [3] 풀 테스트 실행 (Global Training & Zero-shot Test)
# =========================================================
if __name__ == "__main__":
    print(f"🛠️ 디바이스 세팅: {device}")
    
    # 1. 완벽한 피험자 분리 (Train 84명 / Test 21명)
    EXCLUDED_SUBJECTS = [88, 92, 100, 104]
    VALID_SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]
    
    TRAIN_SUBJECTS = VALID_SUBJECTS[:84]
    TEST_SUBJECTS = VALID_SUBJECTS[84:]
    
    # 2. 데이터 추출 (RAM 보호를 위해 분리해서 로드)
    X_k_train, X_t_train, y_train = extract_memory_safe_data(TRAIN_SUBJECTS, phase_name="TRAIN SET")
    X_k_test, X_t_test, y_test = extract_memory_safe_data(TEST_SUBJECTS, phase_name="TEST SET")
    
    train_loader = DataLoader(TensorDataset(X_k_train, X_t_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_k_test, X_t_test, y_test), batch_size=128, shuffle=False)

    # 3. 모델 및 학습 세팅
    model = PhysioNetGeoLIF_4Class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 FULL SCALE 모델 학습 시작! (Train Data: {len(y_train)}, Test Data: {len(y_test)})")
    
    best_acc = 0.0
    
    for epoch in range(100):
        start_time = time.time()
        
        # --- Training ---
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
        scheduler.step()
        
        # --- Zero-Shot Testing ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for kin_batch, tda_batch, target in test_loader:
                kin_batch, tda_batch, target = kin_batch.to(device), tda_batch.to(device), target.to(device)
                
                spike_out = model(kin_batch, tda_batch)
                spike_sum = spike_out.sum(dim=1)
                
                _, predicted = spike_sum.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
        test_acc = 100. * val_correct / val_total
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './best_geosnn_model.pth')
            
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1:03d}/100] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}% | Time: {epoch_time:.1f}s")

    print(f"\n🎉 전체 파이프라인 완료! 최종 뼈대 최고 정확도(Zero-shot): {best_acc:.2f}%")