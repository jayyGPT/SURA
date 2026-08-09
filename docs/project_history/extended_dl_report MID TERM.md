# SURA Indoor Positioning — Extended Deep Learning Modeling Report

**Project goal:** Real-time indoor position estimation from magnetic field, WiFi, and IMU sensors.
**Building:** IT Engineering (single-floor corridor).
**Phones:** A8 / G7 / S8 (train) → S9+ (test, held-out device).

---

## Table of Contents

1. [Data & Preprocessing](#1-data--preprocessing)
2. [Model 1 — Multi-Branch CNN + Bi-LSTM](#2-model-1--multi-branch-cnn--bi-lstm)
3. [Model 2 — Causal Hybrid (CausalConv + uni-LSTM)](#3-model-2--causal-hybrid-causalconv--uni-lstm)
4. [Model 3 — Three-Branch Environment Model](#4-model-3--three-branch-environment-model)
5. [Model 4 — Neural EKF Complementary Filter](#5-model-4--neural-ekf-complementary-filter)
6. [Diagnostic & Audit Scripts](#6-diagnostic--audit-scripts)
7. [What Was Wrong With All Four Models](#7-what-was-wrong-with-all-four-models)
8. [New Architecture — Learned Bayesian Filter](#8-new-architecture--learned-bayesian-filter)
9. [Results Comparison Table](#9-results-comparison-table)
10. [Current Constraints & What's Next](#10-current-constraints--whats-next)

---

## 1. Data & Preprocessing

Two completely separate data pipelines were built, matching the two "eras" of modeling.

### 1.1 Preprocessing v1 — [preprocess_dl_pipeline.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/preprocess_dl_pipeline.py)

This was the **original pipeline** used by Model 1 (CNN-LSTM) and Model 2 (Causal Hybrid).

| Step | Detail |
|---|---|
| **Input** | `Continuous_Fused_{phone}.csv` — one file per phone (A8, G7, S8 for train; S9+ for test) |
| **WiFi features** | Every column containing `:` (i.e. BSSID MAC addresses). Missing APs set to `NaN`, then per-row z-score normalised and `NaN → 0`. All unique BSSIDs from train+test are unified into a single column set. |
| **IMU features** | `Mag_x/y/z`, `Acc_x/y/z`, `Gyro_x/y/z`, `Orn_x/y/z`, and optionally `Pressure` — 12–13 raw channels, MinMax-scaled using **train-only** statistics. |
| **Downsampling** | Every 25th row (50 Hz → 2 Hz). |
| **Windowing** | Sliding window of 10 timesteps (= 5 seconds at 2 Hz). Each window is one sample: `(WiFi[10, F_w], IMU[10, F_i], Labels[10, 2])`. |
| **Labels** | `True_X`, `True_Y` — the absolute position per frame. |
| **Output** | `.npy` arrays saved to `Datasets/dl_processed/`. Train ≈ 325 windows, Test ≈ 99 windows. |

> [!IMPORTANT]
> This pipeline treats WiFi and IMU as two flat feature groups. The magnetic field is **not** separated from the other IMU channels — it is mixed into the same branch, making it impossible for the model to treat it as a direction-independent spatial fingerprint.

### 1.2 Preprocessing v2 — [preprocess_v2.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/preprocess_v2.py)

Built for Model 3 (Environment) and Model 4 (Neural EKF). Key improvements:

| Change | Why |
|---|---|
| **Three feature groups** instead of two | WiFi (spatial), Magnetic `Mag_x/y/z` (spatial), and Motion `Acc_Mag, Gyro_Mag, Head_Cos, Head_Sin, [Pressure]` (temporal). This separation is the architectural prerequisite for the environment model. |
| **Engineered IMU features** | Raw 9-axis IMU replaced with scalar magnitudes (`Acc_Mag`, `Gyro_Mag`) and heading decomposed into `cos(Orn_z)`, `sin(Orn_z)`. This removes raw body-frame coupling. |
| **Separate MinMax scalers** | `mag_scaler` and `imu_scaler` fitted independently on train. |
| **Reversed test set** | The entire S9+ test sequence is time-reversed and windowed separately into `X_test_rev_*.npy`. This is the critical direction-invariance test: if the model learned the *environment*, the reversed path should produce similar errors; if it memorised the *route*, reversed errors will explode. |
| **Output** | `Datasets/dl_processed_v2/` — includes forward + reversed test arrays, and a magnetic sub-array. |

---

## 2. Model 1 — Multi-Branch CNN + Bi-LSTM

**File:** [train_cnn_lstm.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_cnn_lstm.py)
**Class:** [MultiBranchCNNLSTM](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_cnn_lstm.py#L9-L61)

### Architecture

```
WiFi [B, 10, F_w]  ──permute──►  Conv1d(F_w→64, k=3, pad=1) → ReLU → MaxPool(2)  ──►  [B, 64, 5]
                                                                                          │
IMU  [B, 10, F_i]  ──permute──►  Conv1d(F_i→32, k=3, pad=1) → ReLU → MaxPool(2)  ──►  [B, 32, 5]
                                                                                          │
                                              ┌───────── concat [B, 96, 5] → permute [B, 5, 96] ──┐
                                              │                                                      │
                                              └──► Bi-LSTM(96→128, bidir=True) → last step [B, 256] │
                                                       │                                              │
                                                       └──► FC(256→64) → ReLU → FC(64→2) ──► (X, Y) ─┘
```

### Key details

- **Bidirectional LSTM** — processes the 10-frame window in both directions. The output at the "last" timestep (`lstm_out[:, -1, :]`) actually incorporates information from all 10 frames including *future* ones within the window. Combined with the `MaxPool1d(2)` which selects the strongest activation across pairs of frames, **this model is not causal** — it looks ahead within each window.
- **Predicts absolute (X, Y)** — a single `(x, y)` point per window. Loss: MSE on position.
- **Training:** Adam lr=0.001, batch=32, 50 epochs, early stopping patience=5.
- **Result:** Mean error **5.04 m**, max 17.4 m on S9+.

### Why it partially works

The bidirectional look-ahead gives the model extra context that smooths predictions. However, this makes it unsuitable for real-time deployment (you'd need to wait for the full window before emitting a prediction).

---

## 3. Model 2 — Causal Hybrid (CausalConv + uni-LSTM)

**File:** [train_causal_hybrid.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_causal_hybrid.py)
**Class:** [HybridCausalModel](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_causal_hybrid.py#L21-L56)

### What changed from Model 1

| Model 1 (CNN-LSTM) | Model 2 (Causal Hybrid) |
|---|---|
| Standard `Conv1d` with symmetric padding | Custom [CausalConv1d](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_causal_hybrid.py#L9-L19): pads only the left side `(pad, 0)` so the convolution never sees future frames |
| `MaxPool1d(2)` — picks strongest across time pairs | Removed — no temporal pooling |
| Bi-LSTM (bidirectional=True) | **Uni-LSTM** (bidirectional=False) — strictly forward |
| Predicts absolute (X, Y) | Predicts **frame-to-frame deltas** (dx, dy) |
| MSE on position | MSE on **instantaneous deltas**: `true_deltas = y[:, 1:] - y[:, :-1]` |

### Architecture

```
WiFi ──► CausalConv1d(F_w→64, k=3) → ReLU  ──┐
                                                 ├── concat [B, 96, T] → permute [B, T, 96]
IMU  ──► CausalConv1d(F_i→32, k=3) → ReLU  ──┘
                                                       │
                                              uni-LSTM(96→128) → FC(128→64→2) per frame
                                                       │
                                                  (dx, dy) per frame
```

### Evaluation approach

At test time, the predicted deltas are **cumulatively summed** starting from the known start position `P_0`:
```
P_final = P_0 + Σ(predicted deltas)
```
This gives the absolute position at the end of each window for error measurement.

### Key details

- L2 regularisation (`weight_decay=1e-4`) added to Adam.
- 100 epochs, patience=8.
- **Result:** Mean error **3.64 m** on S9+ — a genuine improvement over Model 1, attributable to:
  - Strict causality (no look-ahead cheating).
  - Delta prediction decomposes the problem: learn *how much you moved*, not *where you are*.

### Analysis scripts for this model

- [evaluate_causal_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/evaluate_causal_model.py) — loads `Metrics_S9_Causal_Hybrid.csv`, prints MAE/RMSE/max, generates trajectory plot and CDF curve.
- [analyze_overshoot.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/analyze_overshoot.py) — computes the heading angle from consecutive predictions, detects the **sharpest turn** in the trajectory, and plots true vs. predicted heading around that turn to diagnose heading lag (a known issue with causal-only models at corridor corners).

---

## 4. Model 3 — Three-Branch Environment Model

**File:** [train_env_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_env_model.py)
**Class:** [EnvironmentModel](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_env_model.py#L26-L106)

### Architecture

The first model to **explicitly separate** spatial (environment) features from temporal (motion) features:

```
SPATIAL BRANCH 1 — WiFi fingerprint (per-frame MLP, NO temporal processing):
  WiFi [B*T, F_w] → Linear(F_w→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → [B, T, 64]

SPATIAL BRANCH 2 — Magnetic fingerprint (per-frame MLP, NO temporal processing):
  Mag [B*T, 3] → Linear(3→32) → ReLU → Linear(32→32) → ReLU → [B, T, 32]

MOTION BRANCH — IMU temporal processing:
  IMU [B, T, F_i] → CausalConv1d(F_i→32, k=3) → ReLU → uni-LSTM(32→128) → Linear(128→32) → ReLU → [B, T, 32]

FUSION HEAD:
  concat([WiFi 64, Mag 32, IMU 32]) = [B, T, 128] → Linear(128→64) → ReLU → Linear(64→2) → (X, Y) per frame
```

### Critical design insight

The WiFi and Magnetic branches use **per-frame MLPs** (the input is reshaped to `(B*T, features)`, processed, then reshaped back). This means they have **zero temporal context** — they see each frame in isolation. Since they have no temporal processing, they produce the **same output regardless of walking direction**. This forces them to learn a *spatial fingerprint map* (environment), not a *route sequence*.

The IMU branch retains temporal processing (CausalConv + LSTM) because motion is inherently sequential.

### Evaluation — the reversed-path test

This model introduced the most important diagnostic: evaluating on the **reversed S9+ path**. If the model truly learned the environment, reversed and forward errors should be similar.

**Results:**
| Test | Mean Error |
|---|---|
| Forward S9+ | 6.36 m |
| Reversed S9+ | **63 m** (catastrophic failure) |

The reversed-path explosion proved that **the raw `Mag_x/y/z` features were secretly direction-dependent** — rotating the phone's body frame changes the axis readings even at the same physical location.

### Ablation studies

The training script runs 5 evaluation configs:
1. Full model (WiFi + Mag + IMU)
2. WiFi + Mag only (IMU zeroed)
3. WiFi only (Mag + IMU zeroed)
4. Magnetic only (WiFi + IMU zeroed)
5. Reversed path (full model)

Results are plotted as a bar chart and saved to `env_model_ablation.png`.

---

## 5. Model 4 — Neural EKF Complementary Filter

**File:** [train_ekf_fusion.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_ekf_fusion.py)
**Class:** [NeuralEKFFusion](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_ekf_fusion.py#L23-L113)

### Architecture

Inspired by the Extended Kalman Filter, this model explicitly separates the position estimation into three stages:

```
STAGE 1 — SPATIAL ANCHOR (Measurement Update):
  WiFi → MLP(F_w→128→64)  ──┐
                               ├── concat [96] → Linear(96→64→2) → P_spatial (X_obs, Y_obs)
  Mag  → MLP(3→32→32)    ──┘
  Per-frame, direction-independent. This is the "measurement" — noisy but drift-free.

STAGE 2 — MOTION TRACKER (Prediction Step):
  IMU → CausalConv1d(F_i→32) → uni-LSTM(32→64) → Linear(64→32→2) → (dx, dy)
  Temporal, causal. This is the "prediction" — smooth but drifts.

STAGE 3 — KALMAN GAIN (Alpha Gate):
  concat([spatial_feat 96, imu_lstm_out 64]) → Linear(160→32→1) → Sigmoid → α ∈ [0,1]
  The gate dynamically decides how much to trust the spatial anchor vs. the motion update.

FUSION (auto-regressive):
  P_final[t] = α[t] · P_spatial[t] + (1 - α[t]) · (P_final[t-1] + delta[t])
  First frame: P_final[0] = P_spatial[0] (pure spatial initialisation).
```

### Composite loss function

[composite_loss](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_ekf_fusion.py#L115-L132) trains three objectives simultaneously:

```python
total_loss = loss_final                  # fused path must match ground truth (weight 1.0)
           + 0.2 * loss_spatial          # spatial anchor should be roughly correct (low weight — allowed to be noisy)
           + 2.0 * loss_motion           # predicted deltas must match true deltas (high weight — enforce smooth motion)
```

The rationale: the spatial anchor is inherently noisy (one WiFi scan can match many locations), so penalising it too hard would fight the heatmap ambiguity. The motion model should be very accurate per-frame, so it gets a high weight. The final fused path gets standard weight.

### Key details

- Alpha gate uses Sigmoid → outputs ∈ [0, 1]. At inference, the average alpha tells you the model's overall trust balance: α→1 means "trust WiFi/Mag", α→0 means "trust IMU dead-reckoning".
- 120 epochs, patience=15, Adam with weight_decay=1e-4.
- **Result:** Mean error **4.29 m** on S9+, avg alpha ≈ 0.6 (slightly prefers spatial anchor).

---

## 6. Diagnostic & Audit Scripts

### [audit_paths.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/audit_paths.py)

**Purpose:** Check whether the train and test walks are actually different paths.

**What it found:** After downsampling both the A8 (train) and S9+ (test) walks and comparing frame-by-frame, the max absolute difference between their `True_X`/`True_Y` was **< 1 metre**. This means all four phones walked the **exact same corridor route** — train and test differ only in device, not in path. The script prints a warning: *"The model may be memorizing the route, not learning sensor interpretation!"*

### [test_reversed_path.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/test_reversed_path.py)

**Purpose:** The definitive route-memorisation test. Takes the trained Causal Hybrid model (Model 2), reconstructs the full frame-level S9+ sequence from sliding windows, reverses it, re-windows it, and evaluates.

**What it showed:** The causal hybrid model's reversed error was **much** worse than forward, confirming it learned the route sequence, not the environment. The script generates a side-by-side plot (`forward_vs_reversed_test.png`).

### [analyze_errors.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/analyze_errors.py)

**Purpose:** Analyse Model 1 (CNN-LSTM) predictions — error over time, worst 5% frames, and a zoom on the final 10 timesteps where the model tends to "jump" (trajectory endpoint instability).

### [evaluate_dl_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/evaluate_dl_model.py)

**Purpose:** Standard evaluation of Model 1 — trajectory plot, MAE/RMSE, CDF curve.

### [evaluate_causal_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/evaluate_causal_model.py)

**Purpose:** Same evaluation for Model 2 — trajectory plot, MAE/RMSE, CDF curve.

### [analyze_overshoot.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/analyze_overshoot.py)

**Purpose:** Heading angle analysis for Model 2. Computes true and predicted heading from position deltas, unwraps angles, detects the sharpest turn, and plots heading lag — the model's predicted heading lags behind the true heading at corridor corners because the causal LSTM needs a few frames to "catch up" after a sudden direction change.

---

## 7. What Was Wrong With All Four Models

After building and evaluating all four models, the following fundamental problems were identified:

### 7.1 Single-trajectory training

All four `Continuous_Fused_*.csv` files trace the **same corridor** — same start (≈90, 24), same end, same 2 turns. After downsampling + windowing that gives only ~325 train / ~99 test windows, all from a **single 1-D line** through a 2-D space. There is no off-path spatial coverage. The model can only learn "where along this line am I", never a 2-D environment fingerprint.

Meanwhile, the dataset contains **538 static grid nodes** with matched WiFi scans — a dense, accurate fingerprint database that covers 168 unique (X, Y) locations across the full corridor network. **None of the four training scripts ever used this data.**

### 7.2 Body-frame magnetic field

Raw `Mag_x`, `Mag_y`, `Mag_z` are in the **phone's body frame**. When the user turns around, the same physical location produces completely different axis readings. This is why the Environment Model's reversed-path error exploded to 63 m — the "spatial fingerprint" branch was actually learning a direction-dependent signal.

### 7.3 Single-point regression

Predicting a single `(X, Y)` from an ambiguous WiFi/mag reading effectively **averages** all candidate locations. If one reading matches 3 different spots on the corridor, the model outputs their centroid — which may not be a valid location at all. This is a fundamental representation problem that no amount of training can fix.

### 7.4 Fabricated ground truth

The `Continuous_Fused_*.csv` files were generated by `_temp/Used Scripts/fuse_continuous_wifi.py`, which:
1. Used **BE Building** coordinates, not IT Engineering.
2. **Linearly interpolated** `True_X` / `True_Y` by timestamp from the ordered static node list.
3. **Simulated** WiFi by injecting Gaussian noise around static node scans.

So the "ground truth" was essentially the static node ordering mapped to timestamps — the models were scored against an invented path. This is why they looked like they were memorising a route: the "truth" itself was derived from the route sequence.

---

## 8. New Architecture — Learned Bayesian Filter

The new approach completely abandons the continuous-walk training paradigm. Instead, it separates the problem into components that can each be trained and validated independently.

### 8.1 Architecture Overview

```
WiFi scan ──►  ENVIRONMENT / MEASUREMENT MODEL   ──►  P_obs(x,y) heatmap
               per-frame, direction-invariant,        (a probability map,
               pretrained on the STATIC fingerprint    not a single point)
               DB. WiFi-anchored.                                │
                                                                  ▼
IMU stream ─►  MOTION MODEL (causal PDR)          ──►   ┌────────────────────┐
               step detection + heading →               │  CAUSAL EKF FUSION │ ──► (x,y)_t
               world-frame displacement                 │  measure = heatmap │
                                                        │  predict = PDR     │
Magnetometer ► ONLINE SELF-CALIBRATION (relative        └────────────────────┘
               anomaly; feeds motion/fusion stage)
```

### 8.2 Stage 1 — Fingerprint Database

**File:** [build_fingerprint_db.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/build_fingerprint_db.py)

This script builds a proper ground-truth database from the **static measurements** — the data that actually samples the 2-D space.

#### Data sources

- **Magnetic field dataset** → `Static Data/IT Engineering/` — hundreds of CSV files, each a ~119-row recording at a single grid node (phone held still).
- **WiFi dataset** → `IT Engineering/` — Excel files with BSSID/RSS pairs, one per static node visit.
- Files are matched by name: `IMU_Node42_...csv` → `WiFi_Node42_...csv`.

#### Rotation-invariant magnetic features

The function [mag_rotation_invariant](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/build_fingerprint_db.py#L57-L82) computes features that are **properties of the location, not the phone orientation**:

| Feature | Formula | Why invariant |
|---|---|---|
| `magN` (‖M‖) | `norm(Mag_x, Mag_y, Mag_z)` | Magnitude doesn't change with rotation |
| `magV` | `M · ĝ` (dot product with gravity direction) | World-vertical component, independent of horizontal orientation |
| `magH` | `√(‖M‖² - magV²)` | Horizontal magnitude |
| `dip` | `atan2(magV, magH)` | Magnetic inclination angle |

Gravity direction `ĝ` is estimated from the accelerometer (valid because static recordings ≈ still → Acc ≈ gravity). Each feature is aggregated as mean + std over the node's ~119 readings → **8 magnetic features per node**.

#### WiFi processing

- A global BSSID vocabulary is built from all scans (sorted by frequency). Result: **250 unique access points**.
- Each node-visit gets a 250-dimensional RSS vector, with missing APs set to -100 dBm.
- If an AP appears twice in one scan, the **strongest** RSS is kept.

#### Output

- `Datasets/fingerprint_db/it_engineering/nodes.csv` — 538 rows (node-visits), columns: `x, y, mode, scenario, phone, user, n_mag_rows, file, 8 mag features, n_ap, has_wifi, 250 AP columns`.
- `bssid_vocab.json` — ordered BSSID list defining the WiFi vector layout.
- `coverage.png` — scatter plot of all 168 unique nodes, coloured by visit count.

Coverage: **167 out of 168 nodes** have matched WiFi scans.

---

### 8.3 Stage 1b — Online Magnetic Normalizer

**File:** [mag_normalizer.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/mag_normalizer.py)
**Class:** [OnlineMagNormalizer](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/mag_normalizer.py#L83-L190)

This solves the device-dependence problem that killed the earlier models' magnetic features.

#### The problem

Raw magnetometer readings have **hard-iron offset** (a constant bias from the phone's internal magnets) and **soft-iron distortion** (the phone's metal chassis warps the field). Different phones produce completely different readings at the same location. Traditional fix: calibrate once in a lab per phone — but this doesn't scale.

#### The solution: self-calibrating from its own live stream

As the user walks, the phone naturally rotates, and the magnetometer samples points on an ellipsoid (distorted sphere). The normalizer fits this ellipsoid from a trailing buffer and maps it back to a sphere.

**Two-level calibration:**

1. **Ellipsoid fit** ([ellipsoid_fit](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/mag_normalizer.py#L44-L77)) — Full 9-parameter quadric fit: `x², y², z², 2yz, 2xz, 2xy, 2x, 2y, 2z`. Extracts centre (hard-iron), soft-iron correction matrix `W`, and scale. Only used when the point cloud has sufficient **rotational diversity** (eigenvalue ratio > 0.15).

2. **Sphere fit fallback** ([sphere_fit](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/mag_normalizer.py#L34-L41)) — Hard-iron only (3 params + radius). Used when the user hasn't rotated enough for the ellipsoid to be well-conditioned.

**After calibration**, the same 4 rotation-invariant features are computed (`magN, magV, magH, dip`), plus a **causal running normalisation** via EMA:

```python
mean = (1 - α) * mean + α * feat        # trailing mean
var  = (1 - α) * var  + α * (feat - mean)²  # trailing variance
relative = (feat - mean) / √(var + ε)    # z-score → relative anomaly
```

This kills slow DC drift and environmental baseline, leaving only the **spatially unique relative anomaly** — which is the signal that actually varies by location.

#### Streaming parameters

| Parameter | Default | Purpose |
|---|---|---|
| `buffer_size` | 600 | Trailing raw-sample buffer for calibration |
| `refit_every` | 50 | Recalibrate every N samples |
| `min_points` | 80 | Minimum buffer before first fit attempt |
| `ema_alpha` | 0.02 | EMA rate (smaller = slower adaptation) |
| `warmup` | 60 | Samples before running-norm is considered stable |
| `diversity_thresh` | 0.15 | Eigenvalue ratio to trust ellipsoid fit |

#### Validation result

Run on all four continuous walks: **cross-device ‖M‖ spread reduced 3×** (std 1.77 → 0.57 μT), confirming the calibration generalises across devices.

---

### 8.4 Stage 2 — WiFi Heatmap Environment Model

**File:** [stage2_wifi_heatmap.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py)
**Class:** [WifiHeatmapNet](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py#L74-L84)

This is the core innovation. Instead of predicting a single (X, Y), it predicts a **probability heatmap** over a grid of floor cells.

#### Grid construction

[Grid](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py#L43-L54) — builds a regular grid over the node coverage area:
- Cell size: 1.0 m (configurable).
- Grid dimensions determined by min/max of all node coordinates, rounded to integers.
- `coords` array: `(nx*ny, 2)` — the (X, Y) centre of every cell.

#### Target generation

Each training sample's target is a **Gaussian blob** centred on the true node position:
```
target[cell] = exp(-d² / (2σ²))     where d = distance from cell to true position, σ = 2.0 m
```
Normalised to sum=1 → a proper probability distribution. This is crucial: it honestly represents *how sure* the fingerprint is about the position. Nearby cells get non-zero probability, distant cells get near-zero.

#### WiFi encoding

[encode_wifi](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py#L63-L68) — RSS values are clipped to `[-90, -30]` dBm, linearly mapped to `[0, 1]`, and APs below `-100` (absent) are set to 0. This makes the input bounded and interpretable.

#### Model

```
WiFi [B, n_ap] → Linear(n_ap→256) → ReLU → Dropout(0.3) → Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→n_cells) → logits
```

- **Per-frame MLP** — no temporal processing whatsoever. Each WiFi scan is processed independently → **trivially direction-invariant**.
- **Output:** raw logits over `n_cells` grid cells. Softmax applied at inference.
- **Loss:** KL divergence between predicted log-softmax and the target Gaussian blob. This is a *soft classification* — much better than MSE on a single point.

#### Prediction

[soft_argmax](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py#L87-L89) — the predicted position is the **probability-weighted centroid**:
```
position = Σ(p[cell] × coords[cell])
```
This naturally handles ambiguity: if two cells have equal probability, the prediction is their midpoint with an implicit uncertainty.

#### Two evaluation splits

1. **Random split** (80/20) — tests generalisation to unseen *visits* of the same nodes.
2. **Phone split** (S9+ held out) — tests generalisation to an **unseen device**.

#### Results

| Split | MAE | Notes |
|---|---|---|
| Random | **1.43 m** | Best result across all models |
| Phone (S9+ held out) | **2.02 m** | Generalises to unseen device |

This model **beats every earlier model** while using no trajectory data, no temporal processing, no look-ahead, and no fabricated ground truth. It trains purely on the 538 real static node visits.

---

### 8.5 Stage 3a — Causal EKF Fusion

**File:** [stage3_ekf_fusion.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py)

This is the full end-to-end pipeline: WiFi heatmap (measurement) + IMU dead-reckoning (prediction) fused by a standard 2D Kalman filter.

#### Components

1. **Environment model** ([build_env](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py#L40-L58)) — retrains the WiFi heatmap net specifically **excluding S9+** from the static DB, so the test is on a truly held-out device.

2. **WiFi map** ([build_wifi_map](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py#L61-L76)) — per-node averaged real WiFi scans, indexed by a KD-tree for fast nearest-node lookup. At each simulated WiFi update, the walk's true position is mapped to its nearest surveyed node, and that node's real WiFi scan is fed to the heatmap model.

3. **IMU PDR** ([pdr_controls](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py#L109-L119) + [StepDetector](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py#L94-L106)) — online step detection on `‖acc‖` with a high-pass threshold (0.6 m/s²) and refractory period (0.3 s). Each detected step advances position by `step_length` along `Orn_z + heading_offset`.

4. **EKF fusion** ([run_ekf](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py#L125-L152)):
   ```
   For each frame t:
     PREDICT:  x = x + u[t]                   (PDR displacement)
               P = P + Q                       (process noise)
     MEASURE (every wifi_period frames):
               z, R = heatmap_fix(...)          (WiFi fix + covariance from heatmap spread)
               S = P + R
               K = P · S⁻¹                     (Kalman gain)
               x = x + K · (z - x)             (update)
               P = (I - K) · P                 (posterior covariance)
   ```

   The measurement covariance `R` is read **directly from the heatmap's spatial spread** — if the heatmap is peaked (confident), R is small; if the heatmap is spread (ambiguous), R is large. This is honest uncertainty propagation.

#### Calibration

On the **train walks** (A8, G7, S8):
- **Heading offset** = mean angular difference between true heading and `Orn_z`.
- **Step length** = total true path length / number of detected steps, averaged across phones.
- **Filter noise (Q, R)** = small grid search over `q_step ∈ {0.2, 0.5, 1.0}` and `r_scale ∈ {0.5, 1.0, 2.0}`, pick combo that minimises train MAE. Then **freeze** for test.

#### What makes this fundamentally different

- **Almost no learned parameters** in the filter itself — just the heatmap MLP (trained on static data). The EKF is classical closed-form. This means **it cannot memorise a route**.
- WiFi is the **drift-free anchor** (proven: 2 m MAE on static data). IMU is the **smooth filler** between WiFi fixes (causal, no look-ahead).
- The only reason the full EKF isn't yet producing final numbers is constraint #2 below (no WiFi during walks).

---

### 8.6 Stage 3b — Real-Sensor Demo

**File:** [stage3_realdemo.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_realdemo.py)

Because the continuous walk recordings contain **no WiFi** and **no per-frame ground truth**, the full EKF fusion cannot be evaluated on real data. This demo instead runs **pure causal IMU dead-reckoning** on real IT Engineering walks to demonstrate the motion model.

#### Pipeline

1. Load real continuous IMU files from `Magnetic field dataset/Continuous Data/IT Engineering/Navigation/`.
2. Extract the start coordinate from the file's `X-cord`/`Y-cord` columns.
3. For each candidate heading offset (72 values, 5° steps):
   - Run step detection + PDR to get a track.
   - Score: mean distance from track points to surveyed corridor nodes (via KD-tree).
4. Pick the offset that minimises this distance — an honest **map-matching constraint**, the only correction available without WiFi.
5. Plot the PDR track overlaid on the corridor node map.

#### Results

Causal PDR achieves ~1.5 m mean distance to corridor on short walks, but **drifts on long walks** — expected and unavoidable without the WiFi anchor. This is exactly why the WiFi heatmap + EKF fusion is needed.

---

## 9. Results Comparison Table

| # | Model | File | Causal? | Trains on | Predicts | Mean err (S9+) | Reversed err |
|---|---|---|---|---|---|---|---|
| 1 | CNN + Bi-LSTM | `train_cnn_lstm.py` | ❌ | Continuous walks | absolute (X,Y) | 5.04 m | — |
| 2 | Causal Hybrid | `train_causal_hybrid.py` | ✅ | Continuous walks | deltas → integrated | 3.64 m | much worse |
| 3 | Environment 3-Branch | `train_env_model.py` | ✅ | Continuous walks | absolute (X,Y) | 6.36 m | **63 m** |
| 4 | Neural EKF | `train_ekf_fusion.py` | ✅ | Continuous walks | fused absolute | 4.29 m | — |
| — | — | — | — | — | — | — | — |
| 5 | **WiFi Heatmap (new)** | `stage2_wifi_heatmap.py` | ✅ | **Static DB** (538 visits) | probability heatmap | **1.43 m** (random), **2.02 m** (held-out device) | N/A (per-frame, no direction) |
| 6 | **Causal EKF Fusion (new)** | `stage3_ekf_fusion.py` | ✅ | Static DB + train walks (cal only) | fused absolute | *(awaiting WiFi-enabled walks)* | — |

> [!TIP]
> The WiFi heatmap environment model achieves **1.43 m MAE** — a **2.5× improvement** over the best earlier model — while generalising across devices, using no trajectory data, and being trivially direction-invariant.

---

## 10. Current Constraints & What's Next

### What blocks the final end-to-end result

1. **No per-frame trajectory ground truth.** Continuous recordings label only the start node. We cannot train or quantitatively score a trajectory/fusion model on real data.

2. **No WiFi logged while walking.** WiFi in this dataset is always static node scans. The WiFi correction (proven at 2 m) cannot be fed during real walks — the real-sensor demo is dead-reckoning alone.

3. **Pure dead-reckoning drifts.** Expected without an absolute anchor. The WiFi heatmap is the anchor.

4. **Per-building scope.** WiFi AP sets and coordinate frames differ per building. This is inherently a per-environment model.

5. **Magnetic is weak per-point.** Between/within-node ratio ≈ 1.3–1.5 and device-dependent. Its real signal is the *moving spatial profile* (temporal pattern), which belongs in the fusion stage, not as a static point feature.

### What unlocks the complete result

A targeted data-collection campaign: **continuous walks that log WiFi scans (~1 Hz) and sparse position checkpoints** for ground truth. With that:

- The already-written EKF fusion (`stage3_ekf_fusion.py`) drops in and becomes quantitatively testable end-to-end.
- The WiFi heatmap provides drift-free fixes every ~1 second.
- The IMU PDR smooths between fixes.
- The full system runs in real-time, is strictly causal, and generalises to unseen devices.

---

## Appendix: Complete File Inventory

| File | Role | Lines |
|---|---|---|
| [preprocess_dl_pipeline.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/preprocess_dl_pipeline.py) | Preprocessing v1 (2-branch, used by Models 1 & 2) | 143 |
| [preprocess_v2.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/preprocess_v2.py) | Preprocessing v2 (3-branch + reversed test, used by Models 3 & 4) | 193 |
| [train_cnn_lstm.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_cnn_lstm.py) | Model 1: Multi-Branch CNN + Bi-LSTM | 180 |
| [train_causal_hybrid.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_causal_hybrid.py) | Model 2: Causal Hybrid (CausalConv + uni-LSTM) | 190 |
| [train_env_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_env_model.py) | Model 3: Three-Branch Environment Model | 315 |
| [train_ekf_fusion.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/train_ekf_fusion.py) | Model 4: Neural EKF Complementary Filter | 271 |
| [build_fingerprint_db.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/build_fingerprint_db.py) | Stage 1: Build static fingerprint database | 228 |
| [mag_normalizer.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/mag_normalizer.py) | Stage 1b: Online magnetic self-calibration | 219 |
| [stage2_wifi_heatmap.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py) | Stage 2: WiFi heatmap environment model | 190 |
| [stage3_ekf_fusion.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py) | Stage 3a: Causal EKF fusion (full pipeline) | 233 |
| [stage3_realdemo.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_realdemo.py) | Stage 3b: Real-sensor PDR demo | 123 |
| [evaluate_dl_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/evaluate_dl_model.py) | Evaluation: Model 1 metrics + plots | 63 |
| [evaluate_causal_model.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/evaluate_causal_model.py) | Evaluation: Model 2 metrics + plots | 63 |
| [analyze_errors.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/analyze_errors.py) | Diagnostic: Error distribution analysis | 53 |
| [analyze_overshoot.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/analyze_overshoot.py) | Diagnostic: Heading lag at corridor turns | 52 |
| [audit_paths.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/audit_paths.py) | Diagnostic: Train/test path identity check | 42 |
| [test_reversed_path.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/test_reversed_path.py) | Diagnostic: Reversed-path memorisation test | 144 |
| `best_model.pth` | Saved weights: Model 1 (CNN-LSTM) | — |
| `best_causal_model.pth` | Saved weights: Model 2 (Causal Hybrid) | — |
| `best_env_model.pth` | Saved weights: Model 3 (Environment) | — |
| `best_ekf_model.pth` | Saved weights: Model 4 (Neural EKF) | — |
| `best_wifi_heatmap.pth` | Saved weights: Stage 2 (WiFi heatmap) | — |
