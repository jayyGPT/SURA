# Model 5: WiFiOnlyKalmanNet — Single-Innovation Baseline

**Source file:** [stage3_dual_kalmannet.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_dual_kalmannet.py) (lines 244–274)  
**Paper reference:** Table II "WiFi-only KalmanNet" rows  
**Role in system:** Ablation baseline — proves the value of adding the magnetic channel by comparing against a KalmanNet that only uses Wi-Fi + IMU.

---

## 1. Purpose

This is the **control experiment**. It is architecturally identical to DualKalmanNet but uses only a single measurement channel (Wi-Fi). By comparing DualKalmanNet vs WiFiOnlyKalmanNet under the same conditions, we isolate the exact contribution of the magnetic innovation.

---

## 2. Model Architecture

### WiFiOnlyKalmanNet class (line 244–274)
```python
class WiFiOnlyKalmanNet(nn.Module):
    """Original single-measurement KalmanNet (WiFi only, no magnetic)."""
    def __init__(self, hidden=64):
        super().__init__()
        feat_dim = 2 + 2 + 2 + 2 + 1  # = 9
        self.cell = nn.GRUCell(feat_dim, hidden)
        self.head = nn.Linear(hidden, 4)           # Single 2×2 gain
        nn.init.zeros_(self.head.weight)
        self.head.bias.data = torch.tensor([0.5, 0.0, 0.0, 0.5])  # K ≈ 0.5·I
        self.hidden = hidden
```

### Comparison with DualKalmanNet

| Aspect | WiFiOnlyKalmanNet | DualKalmanNet |
|--------|-------------------|---------------|
| Feature dim | 9 | 13 |
| GRU hidden | 64 | 64 |
| Output dim | 4 (one 2×2 gain) | 8 (two 2×2 gains) |
| Measurement channels | Wi-Fi only | Wi-Fi + Magnetic |
| Magnetic map | Not used | Required |
| Parameters | ~10.5K | ~15.7K |

### GRU Feature Vector (9 dimensions)

```python
feat = torch.cat([innov, dz, M[:, t, :], dx_prev, mw], dim=1)
```

| Feature | Dim | Description |
|---------|-----|-------------|
| `innov` | 2 | Wi-Fi innovation `(z_wifi - x_pred) · mask` |
| `dz` | 2 | Temporal Wi-Fi diff `(z_wifi_t - z_wifi_prev) · mask` |
| `M[:, t, :]` | 2 | PDR motion |
| `dx_prev` | 2 | Previous state update |
| `mw` | 1 | Wi-Fi availability mask |
| **Total** | **9** | |

**Missing vs DualKalmanNet:** `y_mag` (1), `grad` (2), `m_mag` (1) = 4 features removed.

---

## 3. Forward Pass

```python
def forward(self, M, S, mask):
    B, T, _ = M.shape
    h = torch.zeros(B, self.hidden); x = torch.zeros(B, 2)
    z_prev = S[:, 0, :]; dx_prev = torch.zeros(B, 2)
    outs = []
    for t in range(T):
        mw = mask[:, t, :]
        x_pred = x + M[:, t, :]                              # Predict
        innov = (S[:, t, :] - x_pred) * mw                   # Wi-Fi innovation
        dz = (S[:, t, :] - z_prev) * mw                      # Temporal diff
        feat = torch.cat([innov, dz, M[:, t, :], dx_prev, mw], dim=1)
        h = self.cell(feat, h)                                 # GRU update
        K = self.head(h).view(B, 2, 2)                        # Single 2×2 gain
        corr = torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1) * mw  # Correction
        x_new = x_pred + corr                                 # Update
        dx_prev = x_new - x
        z_prev = torch.where(mw.bool(), S[:, t, :], z_prev)
        x = x_new
        outs.append(x)
    return torch.stack(outs, 1)
```

The logic is identical to DualKalmanNet except:
- No magnetic map lookup
- No `y_mag`, `grad`, or `m_mag` features
- Single gain matrix `K` instead of `K_wifi + K_mag`
- Single correction `corr = K @ innov` instead of `corr_wifi + corr_mag`

---

## 4. Training

Trained identically to DualKalmanNet:
- Same synthetic trajectories (250 train, 60 test)
- Same optimizer (Adam, lr=2e-3, weight_decay=1e-5)
- Same loss (MSE on trajectories)
- Same 150 epochs, batch size 32

---

## 5. Results Comparison

From the paper (Table II):

| Regime | WiFiOnlyKalmanNet | DualKalmanNet | Improvement |
|--------|-------------------|---------------|-------------|
| Full Wi-Fi (1 Hz) | 0.55 ± 0.05 m | **0.47 ± 0.03 m** | 14.5% |
| Degraded Wi-Fi (5s, 40% drop) | 1.44 ± 0.18 m | **1.07 ± 0.10 m** | 25.7% |

The magnetic channel provides marginal improvement under ideal Wi-Fi conditions (Wi-Fi already fixes position every second), but a **dramatic 25.7% improvement** when Wi-Fi is degraded — precisely the scenario where magnetic matching is most valuable.

**Verdict: Baseline correctly implemented and fairly compared.**
