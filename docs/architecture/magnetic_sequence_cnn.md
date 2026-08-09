# Model 2: MagSequenceMatcher — 1D-CNN Magnetic Sequence Matcher

**Source file:** [stage2_mag_sequence.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_mag_sequence.py)  
**Paper reference:** Section II.C "Magnetic Sequence Matcher", Figure 3  
**Role in system:** Maps a temporal sliding window of rotation-invariant magnetic features to a 2D spatial coordinate and a calibrated uncertainty estimate.

---

## 1. Purpose and Design Philosophy

Single-point magnetic matching is highly ambiguous — many locations in a building may have similar field magnitude. However, the *sequence* of magnetic anomalies along a walked path is far more discriminative, because the spatial anomaly profile is unique to each corridor segment.

This model exploits temporal context by processing a sliding window of `T` frames (optimal: T=84 ≈ 5 seconds) through a 1D-CNN. The architecture outputs both a position estimate and a learned uncertainty, enabling the downstream Kalman filter to appropriately weight magnetic evidence.

---

## 2. Input Feature Engineering

### Rotation-Invariant Features

**Constants (line 36):**
```python
MAG_FEATS = ["magN", "magV", "magH", "dip"]   # 4 rotation-invariant channels
```

| Channel | Feature | Definition | Why rotation-invariant? |
|---------|---------|------------|------------------------|
| `magN` | Magnetic norm (magnitude) | `‖M_t‖` = total field strength | Scalar — independent of device orientation |
| `magV` | Vertical projection | Projection of M onto estimated gravity vector | Uses gravity to define "up" regardless of device pose |
| `magH` | Horizontal component | Magnitude of M projected onto horizontal plane | Gravity-referenced, orientation-independent |
| `dip` | Dip angle | Angle between M and horizontal plane | Angle — invariant to rotation about vertical axis |

All 4 features are scalar values that remain identical regardless of how the user holds the device. This is critical because the model must work across Navigation, Call listening, and Swinging postures.

### Device-Invariant Anomaly Map

The raw magnetic features vary between different phone models (different magnetometer calibrations). To achieve device invariance:

```python
# line 58: subtract per-phone building-wide mean
df[f"anom_{f}"] = df[col] - df.groupby("phone")[col].transform("mean")
```

For each phone, the building-wide average of each feature is subtracted. This removes the phone-specific bias, leaving only the spatial *anomaly* pattern — the deviations from the mean that are caused by structural steel, electrical wiring, etc.

### Anomaly Map Interpolation (line 42–72)

The anomaly values at discrete survey nodes are interpolated onto a continuous grid using:
1. **Linear interpolation** (`griddata(..., method="linear")`) for smooth regions
2. **Nearest-neighbor fill** (`griddata(..., method="nearest")`) for extrapolation at grid edges

```python
lin = griddata(nxy, nval, grid.coords, method="linear")
nn_fill = griddata(nxy, nval, grid.coords, method="nearest")
vals = np.where(np.isnan(lin), nn_fill, lin).reshape(grid.nx, grid.ny)
```

This produces a tensor of shape `[C, nx, ny]` (4 channels × grid width × grid height).

### Per-Channel Noise Standard Deviation (line 68–69)
```python
node_std = df.groupby([df["x"].round(1), df["y"].round(1)])[f"anom_{f}"].std().median()
stds.append(float(node_std) if np.isfinite(node_std) else 1.0)
```

Within-node measurement noise is estimated as the median standard deviation across all survey nodes. This is used to add realistic noise during synthetic training data generation.

---

## 3. Training Data Generation

### Sliding Windows (line 97–133)

Training data comes from **synthetic walks** (not raw measurements). For each walk:

```python
# Sample magnetic map values at each true position
mag_clean = bilinear_mc(maps_t, tx, grid).numpy()  # [n_frames, 4]

# Add realistic measurement noise
noise = rng.normal(0, 1, size=(n, C)) * np.array(stds)
mag_obs = mag_clean + noise
```

Then sliding windows of length `T` are extracted:
```python
for t in range(window_size, n, stride):
    windows.append(mag_obs[t - window_size: t])  # [T, 4]
    targets.append(true_xy[t - 1])                # [2]
```

The target is the **last frame's position** — strictly causal (the model predicts "where am I now?" from the past `T` frames).

**Data scale:** 300 walks for training, 60 for testing, each producing ~40 windows → ~12,000 training windows.

---

## 4. Model Architecture

### MagSequenceMatcher class (line 139–169)

```python
class MagSequenceMatcher(nn.Module):
    def __init__(self, in_channels=4, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
        self.var_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1),  # outputs log(sigma^2)
        )
```

### Encoder: Layer-by-layer breakdown

The input `[B, T, 4]` is transposed to `[B, 4, T]` (Conv1d expects channels-first):

| # | Layer | Input shape | Output shape | Kernel | Padding | Purpose |
|---|-------|-------------|--------------|--------|---------|---------|
| 1 | `Conv1d(4, 32, k=7, pad=3)` | [B, 4, T] | [B, 32, T] | 7 | 3 (same) | Large receptive field captures broad anomaly trends over ~0.4s. 32 filters learn basic magnetic patterns. |
| 2 | `BatchNorm1d(32)` | [B, 32, T] | [B, 32, T] | — | — | Stabilizes training by normalizing activations per channel. |
| 3 | `ReLU()` | [B, 32, T] | [B, 32, T] | — | — | Non-linearity. |
| 4 | `MaxPool1d(2)` | [B, 32, T] | [B, 32, T/2] | 2 | — | Halves temporal resolution. Forces feature invariance to small time shifts. |
| 5 | `Conv1d(32, 64, k=5, pad=2)` | [B, 32, T/2] | [B, 64, T/2] | 5 | 2 (same) | Medium kernel captures mid-scale anomaly patterns. Channel expansion to 64. |
| 6 | `BatchNorm1d(64)` | [B, 64, T/2] | [B, 64, T/2] | — | — | Normalization. |
| 7 | `ReLU()` | [B, 64, T/2] | [B, 64, T/2] | — | — | Non-linearity. |
| 8 | `MaxPool1d(2)` | [B, 64, T/2] | [B, 64, T/4] | 2 | — | Further halves temporal resolution. |
| 9 | `Conv1d(64, 128, k=3, pad=1)` | [B, 64, T/4] | [B, 128, T/4] | 3 | 1 (same) | Small kernel captures fine-grained local anomaly structure. Channel expansion to 128. |
| 10 | `BatchNorm1d(128)` | [B, 128, T/4] | [B, 128, T/4] | — | — | Normalization. |
| 11 | `ReLU()` | [B, 128, T/4] | [B, 128, T/4] | — | — | Non-linearity. |
| 12 | `AdaptiveAvgPool1d(1)` | [B, 128, T/4] | [B, 128, 1] | — | — | Global average pooling collapses the temporal dimension to a single 128-d feature vector. Makes output independent of exact window length. |

After `.squeeze(-1)`: **encoder output = `[B, 128]`**

### Design decisions:
- **Decreasing kernel sizes (7→5→3):** Captures progressively finer spatial detail. The first layer sees ~0.4s of context; by layer 3, each neuron has an effective receptive field spanning the entire window.
- **BatchNorm after every conv:** Essential for stable 1D-CNN training. Without it, the varying scale of magnetic anomalies across different corridors causes gradient instability.
- **AdaptiveAvgPool1d(1) instead of Flatten:** Makes the architecture agnostic to the exact window size T. During the window-size sweep (T ∈ {50, 84, 134, 167}), the same architecture works for all.

### Position Head (line 155–158)

```python
self.pos_head = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2),
)
```

| # | Layer | Input | Output | Purpose |
|---|-------|-------|--------|---------|
| 1 | `Linear(128, 64)` | 128 | 64 | Compress encoder features |
| 2 | `ReLU()` | 64 | 64 | Non-linearity |
| 3 | `Dropout(0.2)` | 64 | 64 | Regularization — prevents the position head from overfitting to specific corridors |
| 4 | `Linear(64, 2)` | 64 | 2 | Output: 2D spatial coordinate `z_mag = [x, y]` |

### Variance Head (line 159–162)

```python
self.var_head = nn.Sequential(
    nn.Linear(128, 32), nn.ReLU(),
    nn.Linear(32, 1),  # outputs log(sigma^2)
)
```

| # | Layer | Input | Output | Purpose |
|---|-------|-------|--------|---------|
| 1 | `Linear(128, 32)` | 128 | 32 | Smaller head — uncertainty requires less capacity than position |
| 2 | `ReLU()` | 32 | 32 | Non-linearity |
| 3 | `Linear(32, 1)` | 32 | 1 | Output: `log(σ²_mag)` — log-variance (not variance directly) |

**Why log-variance?** The variance must be strictly positive. By predicting `log(σ²)`, the model can output any real number, and `exp()` guarantees positivity. This is standard practice for heteroscedastic uncertainty estimation.

**No Dropout in var_head:** Intentional. The variance head should learn a stable, smooth uncertainty surface. Dropout would inject noise into uncertainty estimates, making them unreliable for downstream filtering.

---

## 5. Forward Pass

```python
def forward(self, x):
    # x: [B, W, C] -> conv expects [B, C, W]
    feat = self.encoder(x.transpose(1, 2)).squeeze(-1)  # [B, 128]
    pos = self.pos_head(feat)        # [B, 2]
    logvar = self.var_head(feat)     # [B, 1]
    return pos, logvar
```

1. **Transpose:** Input `[B, T, 4]` → `[B, 4, T]` for Conv1d
2. **Encode:** Extract 128-d feature vector
3. **Branch:** Shared features → two independent heads
4. **Return:** Position estimate + log-variance

---

## 6. Loss Function

### nll_loss (line 172–176)

```python
def nll_loss(pred_pos, logvar, true_pos):
    """Heteroscedastic Gaussian NLL: encourages calibrated uncertainty."""
    var = torch.exp(logvar).clamp(min=0.01)  # [B, 1]
    sq_err = ((pred_pos - true_pos) ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    return (0.5 * sq_err / var + 0.5 * logvar).mean()
```

This is the negative log-likelihood of a Gaussian: `L = (1/2) · ‖z - z_true‖² / σ² + (1/2) · log(σ²)`

**Two competing terms:**
- `sq_err / var`: Penalizes large errors. If `σ²` is small (high confidence), errors are penalized heavily.
- `log(σ²)`: Penalizes large variance. Prevents the model from trivially setting `σ² → ∞` to make the first term zero.

**The `.clamp(min=0.01)` on variance:** Prevents numerical instability when the model predicts very small variance (which would cause the first term to explode).

---

## 7. Training Configuration

```python
model = MagSequenceMatcher(in_channels=len(MAG_FEATS))  # in_channels=4
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
# 60 epochs, batch size 128
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 1e-3 | Standard Adam default |
| Weight decay | 1e-4 | L2 regularization |
| Scheduler | ReduceLROnPlateau | Halves LR if test MAE doesn't improve for 8 epochs |
| Patience | 8 | Gives the model time to escape local minima |
| Factor | 0.5 | Conservative LR reduction |
| Epochs | 60 | Sufficient for convergence |
| Batch size | 128 | Larger batch for stable gradient estimates |

### Window Size Sweep (line 248–262)

```python
candidates = [50, 84, 134, 167]
# 50 frames  = ~3.0 s
# 84 frames  = ~5.0 s  <-- OPTIMAL
# 134 frames = ~8.0 s
# 167 frames = ~10.0 s
```

The sweep tests 4 window sizes. **T=84 (5.0 s)** was found optimal: short enough to avoid overfitting to long route patterns, long enough to accumulate sufficient spatial variation for unambiguous matching.

---

## 8. Paper Figure Verification

**Figure 3 in Paper.tex** shows:
- Input: `ℝ^{T × 4}` ✅ (matches `in_channels=4`)
- Conv1D(32), k=7, BN, ReLU → MaxPool ✅ (matches lines 147–148)
- Conv1D(64), k=5, BN, ReLU → MaxPool ✅ (matches lines 149–150)
- Conv1D(128), k=3, BN, ReLU → Adaptive AvgPool ✅ (matches lines 151–153)
- Position head: FC(64), ReLU, Drop(0.2), FC(2) → z_mag ✅ (matches lines 155–158)
- Variance head: FC(32), ReLU, FC(1) → log σ²_mag ✅ (matches lines 159–162)
- Dashed encoder group box ✅

**Verdict: Figure 3 is accurate.**
