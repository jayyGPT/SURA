"""
CRITICAL TEST: Can the model localize a REVERSED path?
If the model learned the environment (WiFi -> position), reversing the path should work.
If the model learned the route sequence, reversing will fail catastrophically.
"""
import torch
import torch.nn as nn
import numpy as np
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

class HybridCausalModel(nn.Module):
    def __init__(self, wifi_features, imu_features, lstm_hidden=128):
        super().__init__()
        self.wifi_conv = nn.Sequential(CausalConv1d(wifi_features, 64, kernel_size=3), nn.ReLU())
        self.imu_conv = nn.Sequential(CausalConv1d(imu_features, 32, kernel_size=3), nn.ReLU())
        self.lstm = nn.LSTM(input_size=96, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x_wifi, x_imu):
        x_wifi = x_wifi.permute(0, 2, 1)
        x_imu = x_imu.permute(0, 2, 1)
        out_wifi = self.wifi_conv(x_wifi)
        out_imu = self.imu_conv(x_imu)
        out = torch.cat((out_wifi, out_imu), dim=1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        return self.fc(lstm_out)

def create_sequences(wifi_data, imu_data, labels, time_steps=10):
    X_w, X_i, y = [], [], []
    if len(wifi_data) <= time_steps:
        return np.array([]), np.array([]), np.array([])
    for i in range(len(wifi_data) - time_steps):
        X_w.append(wifi_data[i : i + time_steps])
        X_i.append(imu_data[i : i + time_steps])
        y.append(labels[i : i + time_steps])
    return np.array(X_w), np.array(X_i), np.array(y)

# Load the raw test data
X_test_wifi = np.load('../Datasets/dl_processed/X_test_wifi.npy')
X_test_imu = np.load('../Datasets/dl_processed/X_test_imu.npy')
y_test = np.load('../Datasets/dl_processed/y_test.npy')

# Reconstruct the full frame-level arrays from the sliding windows
# Window 0 has frames 0-9, window 1 has frames 1-10, etc.
# So the full sequence is: window[0] full + last frame of each subsequent window
n_windows = X_test_wifi.shape[0]
seq_len = X_test_wifi.shape[1]
total_frames = n_windows + seq_len - 1

wifi_full = np.zeros((total_frames, X_test_wifi.shape[2]))
imu_full = np.zeros((total_frames, X_test_imu.shape[2]))
labels_full = np.zeros((total_frames, 2))

# Fill from first window
wifi_full[:seq_len] = X_test_wifi[0]
imu_full[:seq_len] = X_test_imu[0]
labels_full[:seq_len] = y_test[0]

# Fill remaining unique frames
for i in range(1, n_windows):
    wifi_full[seq_len - 1 + i] = X_test_wifi[i, -1]
    imu_full[seq_len - 1 + i] = X_test_imu[i, -1]
    labels_full[seq_len - 1 + i] = y_test[i, -1]

print(f"Reconstructed full sequence: {total_frames} frames")
print(f"Forward path: ({labels_full[0,0]:.1f}, {labels_full[0,1]:.1f}) -> ({labels_full[-1,0]:.1f}, {labels_full[-1,1]:.1f})")

# REVERSE the entire sequence
wifi_reversed = wifi_full[::-1].copy()
imu_reversed = imu_full[::-1].copy()
labels_reversed = labels_full[::-1].copy()

print(f"Reversed path: ({labels_reversed[0,0]:.1f}, {labels_reversed[0,1]:.1f}) -> ({labels_reversed[-1,0]:.1f}, {labels_reversed[-1,1]:.1f})")

# Create sliding windows from the reversed data
X_w_rev, X_i_rev, y_rev = create_sequences(wifi_reversed, imu_reversed, labels_reversed, time_steps=10)
print(f"Reversed windows: {X_w_rev.shape[0]}")

# Load model
model = HybridCausalModel(X_test_wifi.shape[2], X_test_imu.shape[2])
model.load_state_dict(torch.load('best_causal_model.pth', weights_only=True))
model.eval()

# Evaluate on FORWARD path (normal)
wifi_t = torch.tensor(X_test_wifi, dtype=torch.float32)
imu_t = torch.tensor(X_test_imu, dtype=torch.float32)
y_t = torch.tensor(y_test, dtype=torch.float32)

# Evaluate on REVERSED path
wifi_r = torch.tensor(X_w_rev, dtype=torch.float32)
imu_r = torch.tensor(X_i_rev, dtype=torch.float32)
y_r = torch.tensor(y_rev, dtype=torch.float32)

def evaluate_trajectory(model, wifi, imu, y_true, label):
    with torch.no_grad():
        pred_deltas = model(wifi, imu)
        pred_sliced = pred_deltas[:, 1:, :]
        p_start = y_true[:, 0:1, :]
        p_pred_final = p_start[:, 0, :] + torch.sum(pred_sliced, dim=1)
        p_true_final = y_true[:, -1, :]
        errors = torch.sqrt(torch.sum((p_pred_final - p_true_final)**2, dim=1))
        
        print(f"\n=== {label} ===")
        print(f"Mean Error: {errors.mean().item():.2f}m")
        print(f"Max Error: {errors.max().item():.2f}m")
        
        return p_pred_final.numpy(), p_true_final.numpy(), errors.numpy()

pred_fwd, true_fwd, err_fwd = evaluate_trajectory(model, wifi_t, imu_t, y_t, "FORWARD (normal)")
pred_rev, true_rev, err_rev = evaluate_trajectory(model, wifi_r, imu_r, y_r, "REVERSED path")

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

axes[0].plot(true_fwd[:, 0], true_fwd[:, 1], 'b-o', markersize=2, label='Ground Truth', alpha=0.6)
axes[0].plot(pred_fwd[:, 0], pred_fwd[:, 1], 'g--', linewidth=2, label=f'Prediction (MAE={err_fwd.mean():.2f}m)', alpha=0.8)
axes[0].set_title('FORWARD Path (Normal Test)', fontsize=14)
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(true_rev[:, 0], true_rev[:, 1], 'b-o', markersize=2, label='Ground Truth (Reversed)', alpha=0.6)
axes[1].plot(pred_rev[:, 0], pred_rev[:, 1], 'r--', linewidth=2, label=f'Prediction (MAE={err_rev.mean():.2f}m)', alpha=0.8)
axes[1].set_title('REVERSED Path (Same Corridor, Opposite Direction)', fontsize=14)
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Y (m)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Does the Model Learn the Environment or the Route?', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../Datasets/forward_vs_reversed_test.png', dpi=300, bbox_inches='tight')
print("\nSaved comparison plot.")
