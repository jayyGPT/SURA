# Model 1: WifiHeatmapNet — Wi-Fi Probability Heatmap MLP

**Source file:** [stage2_wifi_heatmap.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py)  
**Paper reference:** Section II.B "Wi-Fi Processing and Heatmap Model", Figure 2  
**Role in system:** Environment model — maps a single Wi-Fi RSSI scan to a spatial probability distribution over the floor grid.

---

## 1. Purpose and Design Philosophy

This model is deliberately **per-frame and Wi-Fi-only**, making it trivially direction-invariant. It does not see IMU, magnetic, or any temporal context. It learns the *environment's* RF fingerprint structure — not any specific trajectory. This prevents trajectory memorization (a key failure mode in end-to-end models).

The output is a probability heatmap, not a single point. This honestly represents fingerprint ambiguity: one scan can match several places. The downstream Kalman filter resolves ambiguity over time.

---

## 2. Input Preprocessing

### Constants (from code)
```python
WIFI_FLOOR = -100.0    # dBm value representing "AP not seen"
RSS_CLIP   = -90.0     # clip weak signals below this
KEEP_MODES = {"Navigation", "Call listening", "Swinging"}
```

### encode_wifi function (line 63–68)
```python
def encode_wifi(rss_mat):
    """RSS (N, A) with -100 floor -> [0,1] strength, missing -> 0."""
    x = np.clip(rss_mat, RSS_CLIP, -30.0)        # Step 1: Clip to [-90, -30]
    x = (x - RSS_CLIP) / (-30.0 - RSS_CLIP)      # Step 2: Rescale to [0, 1]
    x[rss_mat <= WIFI_FLOOR] = 0.0                # Step 3: Absent APs → 0
    return x.astype(np.float32)
```

**Step-by-step:**
1. **Clip:** Any signal weaker than -90 dBm is treated as -90; anything stronger than -30 dBm stays at -30. This removes extreme noise in very weak signals.
2. **Rescale:** Linear mapping from [-90, -30] dBm → [0.0, 1.0]. A signal at -90 dBm maps to 0.0 (very weak); a signal at -30 dBm maps to 1.0 (very strong).
3. **Absent APs:** If the raw value was ≤ -100 dBm (the floor constant), the AP was not detected at all. These are set to exactly 0.0, distinguishing "not seen" from "seen but very weak."

**Input shape:** `[B, N]` where `N` = number of unique APs in the vocabulary (determined by `bssid_vocab.json`).

---

## 3. Spatial Grid

### Grid class (line 43–54)
```python
class Grid:
    def __init__(self, xs, ys, cell=CELL):  # CELL = 1.0 m
        self.x0, self.x1 = float(np.floor(xs.min())), float(np.ceil(xs.max()))
        self.y0, self.y1 = float(np.floor(ys.min())), float(np.ceil(ys.max()))
        self.cell = cell
        self.nx = int(round((self.x1 - self.x0) / cell)) + 1
        self.ny = int(round((self.y1 - self.y0) / cell)) + 1
        # ... meshgrid creation ...
        self.coords = np.stack([self.gxx.ravel(), self.gyy.ravel()], axis=1)  # (M, 2)
        self.n_cells = self.nx * self.ny  # = M
```

The grid is a regular 1.0 m × 1.0 m lattice covering the bounding box of all surveyed nodes. `M = nx × ny` is the total number of grid cells. Each cell has physical coordinates `C_c ∈ ℝ²`.

---

## 4. Model Architecture

### WifiHeatmapNet class (line 74–84)
```python
class WifiHeatmapNet(nn.Module):
    def __init__(self, n_ap, n_cells):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ap, 256), nn.ReLU(), nn.Dropout(0.3),  # Layer 1
            nn.Linear(256, 256),  nn.ReLU(), nn.Dropout(0.3),  # Layer 2
            nn.Linear(256, n_cells),                            # Layer 3 (output)
        )

    def forward(self, x):
        return self.net(x)  # Returns raw logits, NOT probabilities
```

### Layer-by-layer breakdown

| # | Layer | Input dim | Output dim | Activation | Regularization | Purpose |
|---|-------|-----------|------------|------------|----------------|---------|
| 1 | `nn.Linear(n_ap, 256)` | N | 256 | ReLU | Dropout(0.3) | Feature extraction: compress the sparse, high-dimensional RSSI vector into a dense 256-d representation. ReLU introduces non-linearity. Dropout prevents co-adaptation of neurons. |
| 2 | `nn.Linear(256, 256)` | 256 | 256 | ReLU | Dropout(0.3) | Second hidden layer: further non-linear transformation. Same width maintains representational capacity. |
| 3 | `nn.Linear(256, n_cells)` | 256 | M | None | None | Output: raw logits for each grid cell. No activation here because Softmax is applied externally during inference. |

### Key design decisions:
- **Width 256:** Empirically chosen. Two hidden layers of 256 provide sufficient capacity for fingerprint discrimination without overfitting.
- **Dropout 0.3:** Aggressive enough to provide meaningful regularization (30% of neurons zeroed each forward pass), preventing the MLP from memorizing specific scan vectors.
- **No Softmax in forward():** The model outputs raw logits. Softmax is applied externally in `soft_argmax()` (line 87–89). This is intentional: the KL divergence loss uses `log_softmax` internally, which is numerically more stable than `softmax → log`.

---

## 5. Training Target

### gaussian_target method (line 56–60)
```python
def gaussian_target(self, x, y):
    d2 = (self.gxx - x) ** 2 + (self.gyy - y) ** 2
    t = np.exp(-d2 / (2 * SIGMA ** 2)).ravel()  # SIGMA = 2.0 m
    s = t.sum()
    return (t / s).astype(np.float32) if s > 0 else t.astype(np.float32)
```

For each training sample at true coordinates `(x, y)`, the target is a normalized 2D Gaussian blob centered on that point with **σ = 2.0 m**, evaluated at every grid cell. This is a soft label: cells near the true position get high probability, cells far away get near-zero.

**Why soft labels instead of one-hot?** Because multiple grid cells may be equidistant from the true position (e.g., if the user is between two cells). Soft labels provide gradient signal to all nearby cells, not just the nearest one. The width σ = 2.0 m means cells within ~4 m of the true position receive meaningful gradient.

---

## 6. Loss Function

### KL Divergence (line 126–128)
```python
def kl(logits, target):
    logp = torch.log_softmax(logits, dim=1)
    return torch.sum(target * (torch.log(target + 1e-9) - logp), dim=1).mean()
```

This computes `D_KL(q || p) = Σ_c q_c · log(q_c / p_c)` where:
- `q` = Gaussian target distribution (soft label)
- `p` = model's predicted distribution (via softmax of logits)
- `1e-9` prevents `log(0)` when `q_c = 0`

**Why KL divergence instead of cross-entropy?** They differ by the entropy of `q` (a constant). KL divergence is preferred here because it explicitly measures how far the predicted distribution is from the target distribution, and the `target * log(target)` term ensures the loss is exactly 0 when `p = q`.

---

## 7. Inference (Soft-Argmax)

### soft_argmax function (line 87–89)
```python
def soft_argmax(logits, coords_t):
    p = torch.softmax(logits, dim=1)   # [B, M] — probability over cells
    return p @ coords_t                 # [B, 2] — weighted sum of coordinates
```

This computes: `z_wifi = Σ_c p_c · C_c` — the probability-weighted centroid of the heatmap. It is differentiable (unlike hard argmax) and produces a continuous 2D coordinate estimate.

### Covariance computation (in stage3_ekf_fusion.py, heatmap_fix, line 79–88)
```python
mu = (p[:, None] * c).sum(0)        # centroid
d = c - mu                           # deviations
R = (p[:, None, None] * np.einsum("ni,nj->nij", d, d)).sum(0)  # covariance
```

This computes `R_hm = Σ_c p_c · (C_c - z_wifi)(C_c - z_wifi)^T` — the probability-weighted covariance. A sharp, concentrated heatmap yields a small R (high confidence); a spread-out heatmap yields a large R (low confidence). This provides an honest uncertainty estimate to the downstream Kalman filter.

---

## 8. Training Configuration

```python
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 80 epochs, batch size 64
# Train/test split: 80/20 random OR held-out phone (S9+)
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 1e-3 | Standard Adam default |
| Weight decay | 1e-4 | L2 regularization to prevent overfitting |
| Batch size | 64 | Standard for tabular data |
| Epochs | 80 | Empirically sufficient for convergence |
| Train split | 80% random visits | For random split eval |
| Test split | S9+ device only | For cross-device generalization eval |

---

## 9. Paper Figure Verification

**Figure 2 in Paper.tex** shows:
- Input: `x_wifi ∈ ℝ^N` ✅ (matches `n_ap`)
- FC(256), ReLU, Dropout(0.3) ✅ (matches line 78)
- FC(256), ReLU, Dropout(0.3) ✅ (matches line 79)
- FC(M) ✅ (matches line 80)
- Softmax → p ∈ ℝ^M ✅ (matches soft_argmax)

**Verdict: Figure 2 is accurate.**
