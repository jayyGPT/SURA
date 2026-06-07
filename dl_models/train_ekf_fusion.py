"""
EKF-Inspired Neural Complementary Filter
Fuses noisy absolute spatial anchors (WiFi/Mag) with smooth relative motion (IMU).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, **kwargs)
    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class NeuralEKFFusion(nn.Module):
    def __init__(self, wifi_features, mag_features, imu_features, lstm_hidden=64):
        super().__init__()
        
        # --- STAGE 1: SPATIAL ANCHOR (Measurement Update) ---
        # Noisy but drift-free absolute position (X, Y)
        self.wifi_mlp = nn.Sequential(
            nn.Linear(wifi_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mag_mlp = nn.Sequential(
            nn.Linear(mag_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs (X_obs, Y_obs)
        )
        
        # --- STAGE 2: MOTION TRACKER (Prediction Step) ---
        # Smooth relative displacement (dx, dy)
        self.imu_conv = nn.Sequential(
            CausalConv1d(imu_features, 32, kernel_size=3),
            nn.ReLU()
        )
        self.imu_lstm = nn.LSTM(
            input_size=32, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=False
        )
        self.motion_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Outputs (dx, dy)
        )
        
        # --- STAGE 3: KALMAN GAIN (Alpha Gate) ---
        # Dynamically weights trust between Spatial and Motion
        self.alpha_gate = nn.Sequential(
            nn.Linear(64 + 32 + lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Outputs alpha in [0, 1]
        )
        
    def forward(self, x_wifi, x_mag, x_imu):
        B, T, _ = x_wifi.shape
        
        # 1. Compute Spatial Anchors (Frame-independent)
        w_flat = x_wifi.reshape(B * T, -1)
        m_flat = x_mag.reshape(B * T, -1)
        w_feat = self.wifi_mlp(w_flat)
        m_feat = self.mag_mlp(m_flat)
        spatial_feat = torch.cat([w_feat, m_feat], dim=1) # [B*T, 96]
        p_spatial = self.spatial_head(spatial_feat).reshape(B, T, 2) # [B, T, 2]
        
        # 2. Compute Motion Deltas (Temporal)
        imu_in = x_imu.permute(0, 2, 1) # [B, C, T]
        imu_conv = self.imu_conv(imu_in).permute(0, 2, 1) # [B, T, 32]
        imu_lstm_out, _ = self.imu_lstm(imu_conv) # [B, T, lstm_hidden]
        p_motion_delta = self.motion_head(imu_lstm_out) # [B, T, 2]
        
        # 3. Compute Alpha (Kalman Gain)
        gate_in = torch.cat([spatial_feat.reshape(B, T, -1), imu_lstm_out], dim=2)
        alpha = self.alpha_gate(gate_in) # [B, T, 1]
        
        # 4. Auto-regressive EKF Fusion
        # P_final[t] = alpha * P_spatial[t] + (1 - alpha) * (P_final[t-1] + delta[t])
        p_final = []
        for t in range(T):
            a_t = alpha[:, t, :]
            s_t = p_spatial[:, t, :]
            d_t = p_motion_delta[:, t, :]
            
            if t == 0:
                # First frame: initialize with pure spatial prediction
                f_t = s_t
            else:
                f_t = a_t * s_t + (1 - a_t) * (p_final[-1] + d_t)
            
            p_final.append(f_t)
            
        p_final = torch.stack(p_final, dim=1) # [B, T, 2]
        
        # Return all components for composite loss training
        return p_final, p_spatial, p_motion_delta, alpha

def composite_loss(p_final, p_spatial, p_motion_delta, y_true):
    criterion = nn.MSELoss()
    
    # 1. Final Loss: The fused path must match ground truth
    loss_final = criterion(p_final, y_true)
    
    # 2. Spatial Loss: The absolute anchor must be reasonably accurate on its own
    loss_spatial = criterion(p_spatial, y_true)
    
    # 3. Motion Loss: The predicted deltas must match the true instantaneous deltas
    true_deltas = y_true[:, 1:, :] - y_true[:, :-1, :]
    pred_deltas = p_motion_delta[:, 1:, :]
    loss_motion = criterion(pred_deltas, true_deltas)
    
    # Weight the losses. We strongly enforce the final path and motion smoothness.
    # Spatial is allowed to be noisy, so it gets a lower weight.
    total_loss = loss_final + 0.2 * loss_spatial + 2.0 * loss_motion
    return total_loss

def evaluate(model, wifi, mag, imu, y_true, label):
    wifi_t = torch.tensor(wifi, dtype=torch.float32)
    mag_t = torch.tensor(mag, dtype=torch.float32)
    imu_t = torch.tensor(imu, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)
    
    with torch.no_grad():
        p_final, p_spatial, p_delta, alpha = model(wifi_t, mag_t, imu_t)
        # Use the LAST frame of each sequence for the trajectory
        pred_final = p_final[:, -1, :].numpy()
        true_final = y_t[:, -1, :].numpy()
        errors = np.sqrt(np.sum((pred_final - true_final)**2, axis=1))
        
        # Mean alpha value
        mean_alpha = alpha.mean().item()
        
    print(f"\n--- {label} ---")
    print(f"Mean Error: {errors.mean():.2f} meters")
    print(f"Max Error:  {errors.max():.2f} meters")
    print(f"Median:     {np.median(errors):.2f} meters")
    print(f"Avg Alpha Gate (0=IMU, 1=WiFi): {mean_alpha:.3f}")
    
    return pred_final, true_final, errors

def main():
    print("=" * 60)
    print("Training Neural EKF (Complementary Filter) Model")
    print("=" * 60)
    
    base_dir = '../Datasets/dl_processed_v2'
    
    X_train_wifi = np.load(os.path.join(base_dir, 'X_train_wifi.npy'))
    X_train_mag = np.load(os.path.join(base_dir, 'X_train_mag.npy'))
    X_train_imu = np.load(os.path.join(base_dir, 'X_train_imu.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    
    X_test_wifi = np.load(os.path.join(base_dir, 'X_test_wifi.npy'))
    X_test_mag = np.load(os.path.join(base_dir, 'X_test_mag.npy'))
    X_test_imu = np.load(os.path.join(base_dir, 'X_test_imu.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_wifi, dtype=torch.float32),
        torch.tensor(X_train_mag, dtype=torch.float32),
        torch.tensor(X_train_imu, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_wifi, dtype=torch.float32),
        torch.tensor(X_test_mag, dtype=torch.float32),
        torch.tensor(X_test_imu, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = NeuralEKFFusion(
        wifi_features=X_train_wifi.shape[2],
        mag_features=X_train_mag.shape[2],
        imu_features=X_train_imu.shape[2]
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 120
    best_loss = float('inf')
    patience = 15
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_w, b_m, b_i, b_y in train_loader:
            optimizer.zero_grad()
            p_final, p_spatial, p_delta, _ = model(b_w, b_m, b_i)
            loss = composite_loss(p_final, p_spatial, p_delta, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_w.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for b_w, b_m, b_i, b_y in test_loader:
                p_final, p_spatial, p_delta, _ = model(b_w, b_m, b_i)
                loss = composite_loss(p_final, p_spatial, p_delta, b_y)
                test_loss += loss.item() * b_w.size(0)
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1:02d} | Train CompLoss: {train_loss:.4f} | Test CompLoss: {test_loss:.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), 'best_ekf_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    # ========== EVALUATION ==========
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(torch.load('best_ekf_model.pth', weights_only=True))
    model.eval()
    
    # 1. Forward
    pred_fwd, true_fwd, err_fwd = evaluate(
        model, X_test_wifi, X_test_mag, X_test_imu, y_test, "FORWARD S9+")
        
    # PLOTS
    fig, axes = plt.subplots(1, 1, figsize=(9, 7))
    
    axes.plot(true_fwd[:, 0], true_fwd[:, 1], 'b-o', markersize=2, label='Ground Truth', alpha=0.6)
    axes.plot(pred_fwd[:, 0], pred_fwd[:, 1], 'g-', linewidth=2, 
                 label=f'EKF Fusion (MAE={err_fwd.mean():.2f}m)', alpha=0.8)
    axes.scatter([true_fwd[0, 0]], [true_fwd[0, 1]], color='gold', s=150, 
                     edgecolors='black', zorder=5, label='Start')
    axes.set_title('Neural EKF: FORWARD Path (S9+)', fontsize=14)
    axes.set_xlabel('X (m)')
    axes.set_ylabel('Y (m)')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    plt.suptitle('Neural Complementary Filter (EKF Fusion)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../Datasets/ekf_fusion_results.png', dpi=300, bbox_inches='tight')
    print("\\nSaved EKF fusion plot.")

if __name__ == '__main__':
    main()
