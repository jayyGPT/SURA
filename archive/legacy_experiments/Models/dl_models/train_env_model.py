"""
Three-Branch Environment-Aware Model v2
- WiFi branch: Per-frame spatial fingerprint (NO temporal conv - direction independent)
- Magnetic branch: Per-frame spatial fingerprint (NO temporal conv - direction independent)
- IMU branch: Temporal motion processing (CausalConv1d + LSTM)
- Predicts ABSOLUTE position (X, Y) per frame, NOT deltas
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

class EnvironmentModel(nn.Module):
    """
    Three-branch architecture:
    - WiFi: per-frame MLP (spatial fingerprint, direction-independent)
    - Magnetic: per-frame MLP (spatial fingerprint, direction-independent)
    - IMU: CausalConv1d + LSTM (temporal motion context)
    
    WiFi and Magnetic branches have NO temporal processing, so they produce
    the same output regardless of walking direction. This forces them to learn
    the environment's spatial fingerprint map.
    """
    def __init__(self, wifi_features, mag_features, imu_features, lstm_hidden=128):
        super().__init__()
        
        # SPATIAL BRANCH 1: WiFi fingerprint -> position features
        # Per-frame MLP, no convolution, no temporal dependency
        self.wifi_branch = nn.Sequential(
            nn.Linear(wifi_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # SPATIAL BRANCH 2: Magnetic fingerprint -> position features
        # Per-frame MLP, direction-independent
        self.mag_branch = nn.Sequential(
            nn.Linear(mag_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # MOTION BRANCH: IMU temporal processing
        # CausalConv1d for local pattern extraction + LSTM for temporal context
        self.imu_conv = nn.Sequential(
            CausalConv1d(imu_features, 32, kernel_size=3),
            nn.ReLU(),
        )
        self.imu_lstm = nn.LSTM(
            input_size=32, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=False
        )
        
        # IMU temporal output projection
        self.imu_proj = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
        )
        
        # FUSION HEAD: Combine spatial + temporal features -> absolute (X, Y)
        # 64 (wifi) + 32 (mag) + 32 (imu) = 128
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: absolute (X, Y)
        )
        
    def forward(self, x_wifi, x_mag, x_imu):
        B, T, _ = x_wifi.shape
        
        # Spatial branches: process each frame independently
        # Reshape to (B*T, features), apply MLP, reshape back
        wifi_flat = x_wifi.reshape(B * T, -1)
        wifi_out = self.wifi_branch(wifi_flat).reshape(B, T, -1)  # [B, T, 64]
        
        mag_flat = x_mag.reshape(B * T, -1)
        mag_out = self.mag_branch(mag_flat).reshape(B, T, -1)    # [B, T, 32]
        
        # Motion branch: temporal processing
        imu_conv_in = x_imu.permute(0, 2, 1)  # [B, C, T]
        imu_conv_out = self.imu_conv(imu_conv_in).permute(0, 2, 1)  # [B, T, 32]
        imu_lstm_out, _ = self.imu_lstm(imu_conv_out)  # [B, T, lstm_hidden]
        imu_out = self.imu_proj(imu_lstm_out)  # [B, T, 32]
        
        # Fuse all three branches
        fused = torch.cat([wifi_out, mag_out, imu_out], dim=2)  # [B, T, 128]
        
        # Predict absolute position per frame
        positions = self.fusion(fused)  # [B, T, 2]
        return positions

def main():
    print("=" * 60)
    print("Training Three-Branch Environment-Aware Model v2")
    print("=" * 60)
    
    base_dir = '../Datasets/dl_processed_v2'
    
    # Load data
    X_train_wifi = np.load(os.path.join(base_dir, 'X_train_wifi.npy'))
    X_train_mag = np.load(os.path.join(base_dir, 'X_train_mag.npy'))
    X_train_imu = np.load(os.path.join(base_dir, 'X_train_imu.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    
    X_test_wifi = np.load(os.path.join(base_dir, 'X_test_wifi.npy'))
    X_test_mag = np.load(os.path.join(base_dir, 'X_test_mag.npy'))
    X_test_imu = np.load(os.path.join(base_dir, 'X_test_imu.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    
    print(f"Train: WiFi={X_train_wifi.shape}, Mag={X_train_mag.shape}, IMU={X_train_imu.shape}, Y={y_train.shape}")
    print(f"Test:  WiFi={X_test_wifi.shape}, Mag={X_test_mag.shape}, IMU={X_test_imu.shape}, Y={y_test.shape}")
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_wifi, dtype=torch.float32),
        torch.tensor(X_train_mag, dtype=torch.float32),
        torch.tensor(X_train_imu, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_wifi, dtype=torch.float32),
        torch.tensor(X_test_mag, dtype=torch.float32),
        torch.tensor(X_test_imu, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = EnvironmentModel(
        wifi_features=X_train_wifi.shape[2],
        mag_features=X_train_mag.shape[2],
        imu_features=X_train_imu.shape[2],
    )
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 100
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    print("\nTraining on ABSOLUTE position prediction...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_w, b_m, b_i, b_y in train_loader:
            optimizer.zero_grad()
            pred_pos = model(b_w, b_m, b_i)  # [B, T, 2]
            loss = criterion(pred_pos, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_w.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for b_w, b_m, b_i, b_y in test_loader:
                pred_pos = model(b_w, b_m, b_i)
                loss = criterion(pred_pos, b_y)
                test_loss += loss.item() * b_w.size(0)
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1:02d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), 'best_env_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
    # ========== EVALUATION ==========
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(torch.load('best_env_model.pth', weights_only=True))
    model.eval()
    
    def evaluate(model, wifi, mag, imu, y_true, label):
        wifi_t = torch.tensor(wifi, dtype=torch.float32)
        mag_t = torch.tensor(mag, dtype=torch.float32)
        imu_t = torch.tensor(imu, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(wifi_t, mag_t, imu_t)
            # Use the LAST frame of each window for trajectory
            pred_final = pred[:, -1, :].numpy()
            true_final = y_t[:, -1, :].numpy()
            errors = np.sqrt(np.sum((pred_final - true_final)**2, axis=1))
            
        print(f"\n--- {label} ---")
        print(f"Mean Error: {errors.mean():.2f} meters")
        print(f"Max Error:  {errors.max():.2f} meters")
        print(f"Median:     {np.median(errors):.2f} meters")
        
        return pred_final, true_final, errors
    
    # Test 1: Forward S9+
    pred_fwd, true_fwd, err_fwd = evaluate(
        model, X_test_wifi, X_test_mag, X_test_imu, y_test, "FORWARD S9+")
    
    # Test 2: Reversed S9+
    X_rev_wifi = np.load(os.path.join(base_dir, 'X_test_rev_wifi.npy'))
    X_rev_mag = np.load(os.path.join(base_dir, 'X_test_rev_mag.npy'))
    X_rev_imu = np.load(os.path.join(base_dir, 'X_test_rev_imu.npy'))
    y_rev = np.load(os.path.join(base_dir, 'y_test_rev.npy'))
    
    pred_rev, true_rev, err_rev = evaluate(
        model, X_rev_wifi, X_rev_mag, X_rev_imu, y_rev, "REVERSED S9+")
    
    # Test 3: Ablation - WiFi only
    pred_wonly, _, err_wonly = evaluate(
        model, X_test_wifi, X_test_mag * 0, X_test_imu * 0, y_test, "WiFi ONLY (Mag+IMU zeroed)")
    
    # Test 4: Ablation - Magnetic only
    pred_monly, _, err_monly = evaluate(
        model, X_test_wifi * 0, X_test_mag, X_test_imu * 0, y_test, "Magnetic ONLY (WiFi+IMU zeroed)")
    
    # Test 5: Ablation - WiFi+Mag (no IMU)
    pred_spatial, _, err_spatial = evaluate(
        model, X_test_wifi, X_test_mag, X_test_imu * 0, y_test, "WiFi+Mag ONLY (no IMU)")
    
    # Save metrics
    results_df = pd.DataFrame({
        'True_X': true_fwd[:, 0], 'True_Y': true_fwd[:, 1],
        'Pred_X': pred_fwd[:, 0], 'Pred_Y': pred_fwd[:, 1],
        'Error': err_fwd
    })
    results_df.to_csv('../Datasets/Metrics_S9_EnvModel.csv', index=False)
    
    results_rev_df = pd.DataFrame({
        'True_X': true_rev[:, 0], 'True_Y': true_rev[:, 1],
        'Pred_X': pred_rev[:, 0], 'Pred_Y': pred_rev[:, 1],
        'Error': err_rev
    })
    results_rev_df.to_csv('../Datasets/Metrics_S9_EnvModel_Reversed.csv', index=False)
    
    # ========== PLOTS ==========
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    axes[0].plot(true_fwd[:, 0], true_fwd[:, 1], 'b-o', markersize=2, label='Ground Truth', alpha=0.6)
    axes[0].plot(pred_fwd[:, 0], pred_fwd[:, 1], 'g--', linewidth=2, 
                 label=f'Prediction (MAE={err_fwd.mean():.2f}m)', alpha=0.8)
    axes[0].scatter([true_fwd[0, 0]], [true_fwd[0, 1]], color='gold', s=150, 
                     edgecolors='black', zorder=5, label='Start')
    axes[0].set_title('FORWARD Path (S9+)', fontsize=14)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(true_rev[:, 0], true_rev[:, 1], 'b-o', markersize=2, label='Ground Truth (Reversed)', alpha=0.6)
    axes[1].plot(pred_rev[:, 0], pred_rev[:, 1], 'r--', linewidth=2, 
                 label=f'Prediction (MAE={err_rev.mean():.2f}m)', alpha=0.8)
    axes[1].scatter([true_rev[0, 0]], [true_rev[0, 1]], color='gold', s=150, 
                     edgecolors='black', zorder=5, label='Start')
    axes[1].set_title('REVERSED Path (S9+)', fontsize=14)
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Environment-Aware Model: Forward vs Reversed', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../Datasets/env_model_forward_vs_reversed.png', dpi=300, bbox_inches='tight')
    print("\nSaved forward vs reversed plot.")
    
    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ['WiFi+Mag+IMU\n(Full)', 'WiFi+Mag\n(no IMU)', 'WiFi only', 'Mag only', 'Reversed\n(Full)']
    means = [err_fwd.mean(), err_spatial.mean(), err_wonly.mean(), err_monly.mean(), err_rev.mean()]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(configs, means, color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}m', 
                ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (meters)', fontsize=12)
    ax.set_title('Feature Contribution Ablation Study', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../Datasets/env_model_ablation.png', dpi=300, bbox_inches='tight')
    print("Saved ablation plot.")

if __name__ == '__main__':
    main()
