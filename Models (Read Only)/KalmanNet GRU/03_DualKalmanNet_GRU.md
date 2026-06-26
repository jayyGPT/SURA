# Model 3: DualKalmanNet — Dual-Innovation Neural Kalman Filter

**Source file:** [stage3_dual_kalmannet.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_dual_kalmannet.py)  
**Paper reference:** Section II.E "Dual-Innovation KalmanNet Fusion", Figure 1, Equations (4)–(8)  
**Role in system:** The temporal fusion engine — replaces the analytical Kalman gain with a GRU-learned dual gain that fuses Wi-Fi, Magnetic, and IMU measurements into a single trajectory estimate.

---

## 1. Purpose and Design Philosophy

In a classical Extended Kalman Filter (EKF), the Kalman gain `K` is computed analytically from fixed covariance matrices `Q` (process noise) and `R` (measurement noise). These matrices are assumed constant and must be hand-tuned. In highly variable multi-modal environments (e.g., corridors with varying AP density, magnetically featureless zones), this assumption breaks.

**KalmanNet** (Revach et al., 2022) replaces this fixed gain with a learned one: a GRU observes the filter's innovation sequence and outputs an optimal gain matrix at each time step, adapting to the current signal quality.

This implementation **extends** the original KalmanNet with a **dual-innovation** architecture: instead of one measurement channel, the GRU receives innovations from both Wi-Fi (sparse, ~1 Hz) and Magnetic (dense, every frame), and outputs **two separate 2×2 gain matrices**. Binary availability masks allow the system to operate seamlessly even when Wi-Fi drops out entirely.

---

## 2. Coordinate System

The model operates in a **centred coordinate system**: all positions are relative to the starting point of each trajectory. This is critical:

```python
# In build_sequence (line 127):
start = true_xy[0].copy()

# In train_dual (line 308-309):
St = t(S - start[:, None, :])   # Wi-Fi fixes centred to origin
Yt = t(Y - start[:, None, :])   # Ground truth centred to origin
```

The state `x` starts at `[0, 0]` and tracks displacement from the initial position. This makes the model translation-invariant: it learns *how to fuse*, not *where the building is*.

For magnetic map lookups (which require absolute coordinates), the start position is passed separately:

```python
# In forward() (line 207):
xp_abs = (x_pred + start_abs).detach()  # convert back to absolute for map lookup
```

---

## 3. Scalar Magnetic Anomaly Map

Unlike the 4-channel CNN magnetic model, the KalmanNet uses a **single-channel** scalar anomaly map for its innovation features:

### build_scalar_mag_map (line 55–72)
```python
def build_scalar_mag_map(grid):
    df["anom"] = df["magN_mean"] - df.groupby("phone")["magN_mean"].transform("mean")
    # ... interpolation ...
    vals = np.where(np.isnan(lin), nn_fill, lin).reshape(grid.nx, grid.ny)
    gx, gy = np.gradient(vals, grid.cell)  # spatial gradients via finite differences

    return (torch.tensor(vals, ...),   # [nx, ny] anomaly values
            torch.tensor(gx, ...),     # [nx, ny] ∂A/∂x gradient
            torch.tensor(gy, ...),     # [nx, ny] ∂A/∂y gradient
            float(nstd),               # measurement noise std
            grid)
```

This builds:
1. **`vals`** — The scalar anomaly field `A(x, y)` = device-invariant magnetic norm anomaly
2. **`gx`** — `∂A/∂x` spatial gradient (computed via `np.gradient`)
3. **`gy`** — `∂A/∂y` spatial gradient
4. **`nstd`** — within-node measurement noise (used to add realistic noise during training)

### bilinear_scalar helper (line 75–85)
```python
def bilinear_scalar(gridT, x, grid):
    """Sample a [nx, ny] grid at world coords x [B, 2]. Returns [B]."""
```

Performs bilinear interpolation to look up `A(x_pred)`, `∂A/∂x(x_pred)`, and `∂A/∂y(x_pred)` at the current predicted position. This is differentiable-friendly (though `.detach()` is used on the position to prevent gradients flowing through map lookups).

---

## 4. Temporal Binning (build_sequence)

Raw sensor data at 16.7 Hz is too fine-grained for the GRU. The data is **binned** into `T_BINS = 160` time bins per trajectory:

### build_sequence function (line 91–141)
```python
edges = np.linspace(0, n, T + 1).astype(int)   # T=160 equal bins

for i in range(T):
    a, b = edges[i], edges[i + 1]
    M[i] = u[a:b].sum(0)           # PDR motion: sum of step displacements in bin
    # Wi-Fi: carry-forward latest fix
    bf = [fixes[t] for t in fixes if a <= t < b]
    if bf:
        last_wifi = np.mean(bf, axis=0)
        mask_wifi[i] = 1.0           # Wi-Fi available in this bin
    S_wifi[i] = last_wifi            # Always stores the most recent fix
    Y[i] = true_xy[min(b, n) - 1]   # Ground truth at end of bin
    mag_obs[i] = mag_obs_frames[min(b, n) - 1]  # Mag observation at end of bin
```

**Key design decisions:**
- **PDR motion is summed** (not averaged) across frames in each bin — displacement is additive.
- **Wi-Fi is carry-forward**: if no scan arrives in a bin, the mask is 0 and the previous fix is held. This correctly models Wi-Fi sparsity.
- **Magnetic is available every bin** — unlike Wi-Fi, the magnetometer runs continuously at 16.7 Hz.

**Output tensors per trajectory:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `M` | `[T, 2]` | PDR motion (summed step displacements per bin) |
| `S_wifi` | `[T, 2]` | Wi-Fi spatial fix (carry-forward) |
| `mask_wifi` | `[T, 1]` | Binary: 1 if Wi-Fi scan arrived in this bin |
| `mag_obs` | `[T, 1]` | Scalar magnetic anomaly observation |
| `Y` | `[T, 2]` | Ground truth position (centred) |
| `start` | `[2]` | Absolute starting position |

---

## 5. Model Architecture

### DualKalmanNet class (line 147–238)

```python
class DualKalmanNet(nn.Module):
    def __init__(self, magmap, hidden=64):
        super().__init__()
        self.vals, self.gx, self.gy, _, self.grid = magmap

        feat_dim = 2 + 1 + 2 + 2 + 2 + 2 + 1 + 1  # = 13
        self.cell = nn.GRUCell(feat_dim, hidden)
        self.head = nn.Linear(hidden, 8)  # 4 for K_wifi + 4 for K_mag
        self.hidden = hidden

        # Smart initialization
        nn.init.zeros_(self.head.weight)
        self.head.bias.data = torch.tensor([0.5, 0.0, 0.0, 0.5,   # K_wifi ≈ 0.5·I
                                            0.0, 0.0, 0.0, 0.0])  # K_mag starts at 0
```

### Architecture Components

| Component | Type | Input dim | Output dim | Purpose |
|-----------|------|-----------|------------|---------|
| `self.cell` | `nn.GRUCell` | 13 | 64 | Recurrent unit that tracks filter state across time steps |
| `self.head` | `nn.Linear` | 64 | 8 | Decodes GRU hidden state into two 2×2 gain matrices |

**Why GRUCell (not GRU)?** The filter loop must be unrolled manually because each time step requires map lookups and innovation computations that depend on the *current predicted state* — these can't be batched across time.

### Bias Initialization Rationale

```python
# K_wifi bias: [0.5, 0.0, 0.0, 0.5] = 0.5 * I₂
# K_mag bias:  [0.0, 0.0, 0.0, 0.0] = 0₂
```

- **K_wifi starts at 0.5·I:** This means at initialization, the filter already trusts Wi-Fi measurements at 50% weight. This is a reasonable starting point that prevents the filter from diverging early in training.
- **K_mag starts at 0:** Conservative — the magnetic channel starts "off" and the GRU must learn to activate it. This prevents untrained magnetic corrections from corrupting the trajectory.
- **Weight matrix is zeroed:** Combined with the biases, the GRU output starts at a fixed, reasonable gain regardless of input features. The network then learns deviations from this baseline.

---

## 6. Forward Pass — Step-by-Step

### Full forward method (line 183–238)

```python
def forward(self, M, S_wifi, mask_wifi, mag_obs, start_abs):
    B, T, _ = M.shape
    h = torch.zeros(B, self.hidden, device=M.device)    # GRU hidden state
    x = torch.zeros(B, 2, device=M.device)               # State (centred)
    z_prev = S_wifi[:, 0, :]                              # Previous Wi-Fi fix
    dx_prev = torch.zeros(B, 2, device=M.device)          # Previous state update
    outs = []
```

For each time step `t = 0, 1, ..., T-1`:

### Step 1: Prediction
```python
x_pred = x + M[:, t, :]   # Apply PDR motion
```
Standard Kalman predict: add the PDR displacement to the current state.

### Step 2: Wi-Fi Innovation
```python
mw = mask_wifi[:, t, :]                         # [B, 1] — binary mask
y_wifi = (S_wifi[:, t, :] - x_pred) * mw        # [B, 2] — masked innovation
dz_wifi = (S_wifi[:, t, :] - z_prev) * mw       # [B, 2] — temporal Wi-Fi diff
```
- **`y_wifi`**: The spatial residual between the Wi-Fi fix and the predicted position. Masked to zero when no Wi-Fi is available.
- **`dz_wifi`**: How much the Wi-Fi fix changed since the last scan. Gives the GRU velocity information from the Wi-Fi channel.

### Step 3: Magnetic Innovation
```python
xp_abs = (x_pred + start_abs).detach()                    # Convert to absolute coords
map_val = bilinear_scalar(self.vals, xp_abs, self.grid).unsqueeze(-1)  # A(x_pred), [B,1]
grad = torch.stack([
    bilinear_scalar(self.gx, xp_abs, self.grid),           # ∂A/∂x, [B]
    bilinear_scalar(self.gy, xp_abs, self.grid),           # ∂A/∂y, [B]
], dim=1)                                                   # [B, 2]
y_mag = mag_obs[:, t, :] - map_val                         # [B, 1] — scalar innovation
m_mag = torch.ones(B, 1, device=M.device)                  # Always available
```

**This is the key difference from the paper's original (incorrect) equation.** The magnetic innovation is **scalar**: `y_mag = A_obs - A(x_pred)`, where:
- `A_obs` is the magnetic anomaly value actually measured by the phone
- `A(x_pred)` is the map's predicted anomaly value at the current estimated position

If the filter's position estimate is correct, `y_mag ≈ 0`. If the estimate is wrong, `y_mag` indicates the magnitude of mismatch.

The `.detach()` on `xp_abs` prevents gradients from flowing through the map lookup — only the gain matrices are learned, not the map itself.

### Step 4: GRU Feature Assembly
```python
feat = torch.cat([y_wifi, y_mag, grad, dz_wifi, M[:, t, :],
                  dx_prev, mw, m_mag], dim=1)  # [B, 13]
```

| Feature | Dim | Type | Content |
|---------|-----|------|---------|
| `y_wifi` | 2 | Innovation | Wi-Fi spatial residual (masked to 0 when absent) |
| `y_mag` | 1 | Innovation | Magnetic scalar field residual |
| `grad` | 2 | Context | Spatial gradient `[∂A/∂x, ∂A/∂y]` at current position |
| `dz_wifi` | 2 | Velocity | Temporal difference of consecutive Wi-Fi fixes |
| `M[:, t, :]` | 2 | Motion | PDR displacement (step direction and magnitude) |
| `dx_prev` | 2 | History | Previous state update (tells GRU how much it corrected last step) |
| `mw` | 1 | Mask | Wi-Fi availability (0 or 1) |
| `m_mag` | 1 | Mask | Magnetic availability (always 1) |
| **Total** | **13** | | |

### Step 5: GRU Update
```python
h = self.cell(feat, h)         # [B, 64] — updated hidden state
o = self.head(h)               # [B, 8] — raw gain values
```

### Step 6: Decode Dual Gains
```python
K_wifi = o[:, :4].view(B, 2, 2)     # First 4 values → 2×2 Wi-Fi gain
K_mag = o[:, 4:8].view(B, 2, 2)     # Last 4 values → 2×2 Magnetic gain
```

Each gain is a full 2×2 matrix (not diagonal), allowing cross-coupling between x and y corrections.

### Step 7: Dual State Correction
```python
corr_wifi = mw * torch.bmm(K_wifi, y_wifi.unsqueeze(-1)).squeeze(-1)    # [B, 2]
corr_mag = torch.bmm(K_mag, (y_mag * grad).unsqueeze(-1)).squeeze(-1)   # [B, 2]

x_new = x_pred + corr_wifi + corr_mag
```

**Wi-Fi correction:** `m_wifi · K_wifi @ y_wifi` — standard matrix-vector product, masked by availability.

**Magnetic correction:** `K_mag @ (y_mag · ∇A)` — the scalar innovation `y_mag` is broadcast-multiplied by the 2D gradient `∇A`, converting it into a 2D vector pointing in the direction of steepest magnetic change. Then `K_mag` transforms this into the final correction.

**Intuition:** If `y_mag > 0` (measured field is stronger than expected), and `∇A` points North (field gets stronger going North), then the correction pushes the estimate North — exactly where the true position must be.

### Step 8: Bookkeeping
```python
dx_prev = x_new - x                                                    # State update for next step
z_prev = torch.where(mw.bool(), S_wifi[:, t, :], z_prev)              # Update last Wi-Fi fix
x = x_new
outs.append(x)
```

---

## 7. Training Configuration

### train_dual function (line 304–338)

```python
opt = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
mse = nn.MSELoss()
# 150 epochs, batch size 32
# 250 training walks, 60 test walks
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 2e-3 | Slightly higher than typical — the GRUCell has few parameters (13×64 + 64×64 + 64×8 = ~5.4K) |
| Weight decay | 1e-5 | Very light L2 — GRU training is sensitive to over-regularization |
| Batch size | 32 | Small batch due to large sequence length (T=160 steps per walk) |
| Epochs | 150 | Long training to allow GRU to learn complex gain scheduling patterns |
| Loss | MSE | Direct supervision on trajectory position vs. ground truth |
| Temporal bins | T=160 | Each trajectory is discretized into 160 equal-time bins |

### Loss function
```python
mse(out, Yt).backward()  # MSE between predicted trajectory and ground truth
```
The loss is computed over the **entire trajectory** (not per-step), forcing the GRU to learn long-horizon gain scheduling that minimizes cumulative drift.

---

## 8. Evaluation Regimes

Two signal quality regimes are tested:

| Regime | Wi-Fi Period | AP Dropout | Purpose |
|--------|-------------|------------|---------|
| Full Wi-Fi | 1.0 s (1 Hz) | 0% | Normal operating conditions |
| Degraded Wi-Fi | 5.0 s | 40% | Stress test: long gaps + missing APs |

```python
regimes = {
    "Full WiFi (1 Hz, 0% AP drop)": dict(wifi_period=1.0, ap_dropout=0.0),
    "Degraded WiFi (5s, 40% AP drop)": dict(wifi_period=5.0, ap_dropout=0.4),
}
```

In degraded mode, Wi-Fi scans arrive 5× less frequently AND 40% of AP readings are randomly zeroed. This simulates real-world signal attenuation.

---

## 9. Paper Figure Verification

**Figure 1 in Paper.tex** shows:
- PDR Model (u_t) → Predict (x_pred) ✅ (line 200)
- Wi-Fi MLP (z_wifi) → Innovations ✅ (line 203)
- Mag CNN (z_mag) → Innovations ✅ (line 213: y_mag = mag_obs - map_val)
- Predict → Innovations ✅ (x_pred is used in both innovation computations)
- Innovations → Dual GRU (K_wifi, K_mag) ✅ (lines 217–224)
- Dual GRU → Update ✅ (lines 227–230)
- Innovations → Update ✅ (corr_wifi and corr_mag use innovations)
- Predict → Update ✅ (x_new = x_pred + corrections)

**Note on data flow:** The arrow from Mag CNN into Innovations is now correct after our audit fix. Previously it incorrectly showed Mag → GRU (bypassing innovations).

**Verdict: Figure 1 is accurate.**

---

## 10. Complete Parameter Count

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| GRUCell input→hidden weights | 13 × 64 × 3 = 2,496 | 3 gates (reset, update, new) |
| GRUCell hidden→hidden weights | 64 × 64 × 3 = 12,288 | 3 gates |
| GRUCell biases | 64 × 6 = 384 | 3 input biases + 3 hidden biases |
| Linear head weights | 64 × 8 = 512 | |
| Linear head bias | 8 | |
| **Total** | **15,688** | |

This is extremely lightweight — fewer than 16K parameters. The entire KalmanNet can be deployed on any smartphone with negligible compute.
