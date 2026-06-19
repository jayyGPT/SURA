# Dual-Update KalmanNet: Technical Documentation

## Overview

This document describes the Dual-Update KalmanNet architecture, which extends the
original Wi-Fi-only KalmanNet with a second, continuous magnetic measurement channel.
The key design principle is **sensor independence**: the filter learns to dynamically
allocate trust between Wi-Fi (sparse but absolute) and magnetic (dense but weak) via
two independent learned matrix gains. This is **not** a naive fusion; the GRU hidden
state learns a context-dependent trust schedule that slides between sensors based on
the current spatial and temporal conditions.

---

## 1. Motivation and Background

### 1.1 The Limitation of Wi-Fi-Only Fusion
The original KalmanNet (in `stage3_kalmannet.py`) achieves 0.54m MAE on synthetic
trajectories by fusing:
- **IMU PDR** (16.7 Hz, smooth but drifts)
- **Wi-Fi heatmap** (~1 Hz, absolute but jumpy/sparse)

Between two consecutive Wi-Fi scans (roughly 16 frames), the filter relies entirely
on the drifting IMU dead-reckoning with no correction whatsoever. The position error
during these inter-scan gaps accumulates linearly with step noise.

### 1.2 What Magnetic Data Offers
The geomagnetic field varies spatially inside buildings (due to structural steel,
electrical wiring, etc.) and is measured at the full IMU rate of 16.7 Hz. The
rotation-invariant features computed by `mag_normalizer.py` are:

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `magN`  | \|M\|  | Total field magnitude |
| `magV`  | M · ĝ  | Vertical component (along gravity) |
| `magH`  | √(\|M\|² - magV²) | Horizontal magnitude |
| `dip`   | atan2(magV, magH) | Inclination angle |

These are **device-independent** after the online calibration pipeline removes
hard/soft-iron bias. The per-building anomaly field (deviation from building-wide
mean) forms a spatial fingerprint that varies along corridors.

### 1.3 Why Not Fuse Them Into a Single Heatmap?
Multiplying Wi-Fi and magnetic heatmaps together (Hadamard product) would force
a fusion decision at the measurement level. If one sensor is noisy at a given
moment, it corrupts the joint belief. Our dual-update approach instead defers the
fusion to the **filter level**, where the GRU can learn to suppress one channel
contextually.

---

## 2. Architecture

### 2.1 System Overview

```
                  ┌──────────────────────────┐
                  │      IMU Dead Reckoning   │
                  │   (step + heading → u_t)  │
                  └────────────┬─────────────┘
                               │ predict
                  ┌────────────▼─────────────┐
   Wi-Fi scan     │    x_pred = x_{t-1} + u  │     Magnetic stream
   (~1 Hz)        └────────────┬─────────────┘     (16.7 Hz)
       │                       │                        │
       ▼                       │                        ▼
  ┌─────────┐                  │                ┌─────────────┐
  │ Heatmap │                  │                │  Anomaly Map │
  │   MLP   │                  │                │   Lookup     │
  └────┬────┘                  │                └──────┬──────┘
       │ z_wifi                │                       │ mag_obs
       ▼                       │                       ▼
  y_wifi = z_wifi - x_pred     │          y_mag = mag_obs - map(x_pred)
       │                       │                       │
       └───────────┬───────────┘───────────────────────┘
                   │
            ┌──────▼──────┐
            │  GRU Cell   │
            │  (64 units) │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ Linear Head │
            │  → 8 values │
            └──────┬──────┘
            ┌──────┴──────┐
            │             │
      ┌─────▼─────┐ ┌────▼────┐
      │  K_wifi   │ │  K_mag  │
      │  (2×2)    │ │  (2×2)  │
      └─────┬─────┘ └────┬───┘
            │             │
            ▼             ▼
  x_t = x_pred + m_wifi·(K_wifi · y_wifi) + (K_mag · (y_mag · grad))
```

### 2.2 GRU Input Features (13-dimensional)

| Feature       | Dim | Description                                    |
|---------------|-----|------------------------------------------------|
| `y_wifi`      |  2  | Wi-Fi innovation (z_wifi - x_pred), zero-masked |
| `y_mag`       |  1  | Magnetic innovation (obs - map(x_pred))         |
| `grad`        |  2  | Spatial gradient of the magnetic map at x_pred  |
| `dz_wifi`     |  2  | Temporal difference of Wi-Fi fix                |
| `u_t`         |  2  | PDR motion displacement                         |
| `dx_prev`     |  2  | Previous state update (for momentum tracking)   |
| `m_wifi`      |  1  | Binary Wi-Fi availability mask                  |
| `m_mag`       |  1  | Binary magnetic availability mask (always 1)    |

### 2.3 Output: Two Independent Matrix Gains

The linear head outputs 8 values, reshaped into two 2×2 matrices:

- **K_wifi** (indices 0-3): Applied to the 2D Wi-Fi innovation vector.
  - Initialized to `diag(0.5, 0.5)` for sane KF-like initial behaviour.
  - Masked to zero when no Wi-Fi scan is present (`m_wifi = 0`).

- **K_mag** (indices 4-7): Applied to the gradient-projected magnetic correction.
  - Initialized to zeros (conservative start -- the network must learn to trust it).
  - The correction is `K_mag @ (y_mag * grad)`, which maps the scalar magnetic
    mismatch into a 2D spatial displacement via the field gradient. This is the
    key mathematical insight: the gradient tells the filter *which direction* to
    move to reduce the magnetic error.

### 2.4 State Update Equation

```
x_t = x_pred + m_wifi · (K_wifi · y_wifi) + K_mag · (y_mag · grad)
```

This equation runs **every frame**. Between Wi-Fi scans (15 out of 16 frames),
`m_wifi = 0` and the correction reduces to:

```
x_t = x_pred + K_mag · (y_mag · grad)
```

The filter is no longer blind between Wi-Fi scans. If the magnetic innovation is
strong (i.e., the predicted position disagrees with the observed field), K_mag
nudges the state in the gradient direction. If the field is flat (featureless
corridor), the gradient is near zero and the correction vanishes naturally.

---

## 3. New Files Created

### 3.1 `stage2_mag_sequence.py` — Magnetic Sequence Matcher

**Purpose:** Train a 1D-CNN to map a sliding window of magnetic features to a
spatial coordinate. This is a standalone stage-2 model that proves magnetic
sequences carry spatial information, and finds the optimal temporal window size.

**Architecture:**
```
Input [B, W, 4] → Conv1d(4→32, k=7) → BN → ReLU → MaxPool(2)
                → Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2)
                → Conv1d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool(1)
                → Linear(128→64) → ReLU → Dropout(0.2)
                → Position Head: Linear(64→2)
                → Variance Head: Linear(32→1)  (log σ²)
```

**Key design decisions:**
- **Heteroscedastic NLL loss** instead of MSE. This trains the variance head to
  output calibrated uncertainty. In featureless corridors, the model learns to
  report high variance; near distinctive anomaly sequences, it reports low variance.
- **Window-size sweep** across 50/84/134/167 frames (3/5/8/10 seconds). Longer
  windows capture more spatial context but introduce latency.
- **BatchNorm + LR scheduling** for stable convergence on the noisy magnetic signal.

### 3.2 `stage3_dual_kalmannet.py` — Dual-Update KalmanNet

**Purpose:** The main fusion script. Trains and evaluates the DualKalmanNet against
the WiFi-only KalmanNet baseline under two regimes:
1. **Full Wi-Fi** (1 Hz, 0% AP dropout)
2. **Degraded Wi-Fi** (5s period, 40% AP dropout)

**Key design decisions:**
- **Gradient-projected correction:** The magnetic innovation `y_mag` is a scalar
  (mismatch between observed field and map value at the predicted position). To
  convert this into a 2D position correction, we multiply by the spatial gradient
  `grad = [∂A/∂x, ∂A/∂y]` of the anomaly map. This gives the GRU a physically
  meaningful direction to correct along.
- **Conservative initialization:** K_mag starts at zero. The network must learn
  during training that magnetic corrections are beneficial. This prevents early
  training instability from noisy magnetic data.
- **Shared GRU state:** Both sensors share the same 64-unit GRU hidden state.
  This allows cross-sensor reasoning (e.g., "Wi-Fi just corrected 3m to the left,
  so the magnetic innovation should be adjusted accordingly").

---

## 4. Mathematical Details

### 4.1 Magnetic Anomaly Map Construction

For each of the 168 surveyed static nodes, we compute the device-invariant anomaly:

```
anomaly_i = magN_mean_i - mean(magN_mean | phone_i)
```

This subtracts the phone-specific building-wide DC offset (hard-iron bias), leaving
only the spatially varying magnetic signature. The anomaly values are then:
1. Aggregated per unique (X, Y) node by averaging across all phones/visits.
2. Interpolated onto the 1m grid using linear interpolation (nearest-fill outside
   the convex hull).
3. The spatial gradient is computed via `np.gradient`.

### 4.2 Innovation Computation

At each timestep, the magnetic innovation is:

```
y_mag = obs_anomaly - A(x_pred)
```

where `A(x_pred)` is the bilinearly sampled map value at the predicted position.
If the prediction is correct, `y_mag ≈ 0` (within noise). If it's wrong, `y_mag`
is nonzero and the gradient `∇A(x_pred)` indicates which spatial direction reduces
the mismatch.

### 4.3 Why Two Matrix Gains (Not Scalars)

A scalar gain can only scale the correction uniformly in X and Y. But corridors are
often oriented along one axis, meaning the magnetic gradient may be strong in X but
weak in Y (or vice versa). A 2×2 matrix gain allows **cross-coupled corrections**:
a magnetic gradient primarily along X can still correct a Y-drift if the GRU has
learned the local corridor geometry from its hidden state history.

---

## 5. Training Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam  | Standard for RNNs |
| Learning rate | 2e-3 | Matches original KalmanNet |
| Weight decay | 1e-5 | Mild regularization |
| Batch size | 32 | Matches original KalmanNet |
| Epochs | 150 | Empirically sufficient for convergence |
| Hidden size | 64 | Same as original; 128 showed no improvement |
| Train walks | 250 | Synthetic, corridor-graph-faithful |
| Test walks | 60 | Independent, held-out seed |
| Loss | MSE on position | Same as original KalmanNet |

---

## 6. Evaluation Protocol

### 6.1 Regimes

| Regime | Wi-Fi Period | AP Dropout | Purpose |
|--------|-------------|------------|---------|
| Full   | 1.0s        | 0%         | Baseline conditions (magnetic is bonus) |
| Degraded | 5.0s     | 40%        | Stressed conditions (magnetic is critical) |

### 6.2 Metrics
- **Per-walk MAE** across 60 held-out synthetic walks
- **95% confidence interval** via `1.96 * std / sqrt(N)`
- **CDF plot** comparing WiFi-only vs Dual-Update under each regime
- **Relative improvement** `100 * (baseline - dual) / baseline`

---

## 7. Expected Outcomes

- **Full Wi-Fi:** Marginal improvement (1-5%). The 1 Hz Wi-Fi is already strong;
  magnetic adds slight inter-scan drift correction.
- **Degraded Wi-Fi:** Substantial improvement (10-25%). With 5-second Wi-Fi gaps,
  the filter drifts significantly without magnetic corrections. The dense magnetic
  channel prevents the drift from accumulating.

---

## 8. Dependency Chain

```
build_fingerprint_db.py
    └── creates: Datasets/fingerprint_db/it_engineering/{nodes.csv, bssid_vocab.json}

stage2_wifi_heatmap.py
    └── uses: fingerprint DB → trains Wi-Fi heatmap MLP

stage2_mag_sequence.py (NEW)
    └── uses: fingerprint DB + synthetic walks → trains magnetic CNN

stage3_synthetic_eval.py
    └── uses: fingerprint DB + Wi-Fi heatmap → generates synthetic trajectories

stage3_dual_kalmannet.py (NEW)
    └── uses: everything above → trains and evaluates DualKalmanNet
```
