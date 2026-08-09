# Real-Time Indoor Positioning by Learning the Environment, not the Trajectory
### A Technical Report on the SURA Magnetic + WiFi + IMU Localization System

---

## Abstract

We present a real-time, strictly causal indoor-positioning system that estimates a
pedestrian's position from three on-device signals — ambient WiFi, the magnetic field,
and inertial (IMU) motion. The central design principle is that the model should learn the
**environment** (the spatially-unique, temporally-stable fingerprint of a place) rather than
the **trajectory** (a particular walked path), because the environment is what is reusable at
deployment time. We achieve this by (i) training a per-frame, direction-invariant *environment
model* on a dense static fingerprint database, producing a probability **heatmap** over floor
cells; (ii) processing the magnetometer through an *online self-calibrating* pipeline that
generalizes to unseen phones; (iii) deriving causal motion from the IMU via pedestrian
dead-reckoning (PDR); and (iv) fusing the drift-free environment observation with the smooth
motion estimate using **KalmanNet**, a recurrent neural filter that preserves the Kalman
predict–update recursion while replacing the analytic, linear-Gaussian Kalman gain with a
learned matrix gain. The environment model attains **1.43 m** mean error (2.02 m on a fully
held-out device) and generalizes across five buildings; the KalmanNet fusion attains **0.54 m**
on faithful synthetic trajectories, a **46%** improvement over a classical Extended Kalman
Filter (EKF) under identical inputs. We also document the data constraints that shaped these
choices — notably that the dataset contains **no measured per-frame trajectory ground truth** —
and the physically/statistically faithful trajectory-synthesis procedure used to evaluate the
temporal filter.

---

## 1. Data Pre-processing and Real-Trajectory Generation

### 1.1 The raw dataset

The corpus spans six campus buildings and contains three modalities collected very differently:

| Modality | Acquisition | Spatial coverage | Ground-truth quality |
|---|---|---|---|
| Magnetic — **Static** | Stand on a floor grid node, record ~119 samples (~10 s @ ~12 Hz) | Dense grid nodes (IT: 168; 538 globally) | Excellent (feet on a marked node) |
| Magnetic — **Continuous** | Walk a corridor without stopping | Along walked lines only | **Start node only** (see §1.3) |
| **WiFi** | Stand on a node, scan access points (APs) | Same grid nodes as static | Excellent, but format/coord traps (§1.2) |
| **IMU** | Embedded in every magnetic recording | — | Core channels clean; `Orn_z`/`Pressure` ~35% missing |

Per record we have `Time`, ground-truth `(X,Y)`, `Mag_{x,y,z}`, `Acc_{x,y,z}`, `Gyro_{x,y,z}`,
`Orn_{x,y,z}`, and `Pressure`. Two practical facts established during pre-processing:

- **`Orn_z` is the heading** (yaw, in radians); the report's axis labelling was misleading
  (`Orn_x`/`Orn_y` are roll/pitch ≈ 0 for a flat phone). Validated by fitting a single
  constant rotation offset between `Orn_z` and the path tangent: residual **8.8°** std,
  98% of frames within 30°.
- The continuous stream runs at **~16.7 Hz**, pedestrian speed ≈ 0.9–1.3 m/s.

### 1.2 Two data traps (and their fixes)

**Trap A — WiFi files are mislabeled binaries with fake coordinates.** WiFi "`.csv`" files are
actually Excel `.xls` (OLE2/BIFF8). Parsed correctly they are long-format `(BSSID, RSS)` scans.
Crucially, the embedded `X-pos/Y-pos` are **not positions** — they are a 1-based sequential
counter with a constant second axis (verified: 30/30 matched pairs disagree with the magnetic
file). *Fix:* take ground-truth `(X,Y)` from the **magnetic** file and pair WiFi to magnetic by
filename timestamp; ignore the WiFi coordinate columns.

**Trap B — magnetometer is in the phone body frame.** Raw `Mag_{x,y,z}` rotate with heading,
so the same location yields different vectors depending on walking direction. Using them as a
"spatial fingerprint" silently encodes direction. *Fix:* rotation-invariant features (§3.2).

The fingerprint database is built by `build_fingerprint_db.py`: for each static node we store
the ground-truth `(X,Y)` (from magnetic), rotation-invariant magnetic features, and the
matched real WiFi scan as an RSS vector indexed by a per-building AP vocabulary (absent AP =
−100 dBm sentinel). For IT Engineering: **168 unique nodes, 250 APs, WiFi covering 167/168 nodes.**

### 1.3 The ground-truth problem and faithful trajectory generation

The continuous recordings label **only the start node** (verified: a raw IT walk reads `(90,24)`
for all 2 893 rows). There is therefore **no measured per-frame trajectory ground truth anywhere
in the dataset**. (A prior pipeline had *fabricated* it — interpolating the static node list by
timestamp, in the wrong building — which is why earlier models appeared to "memorize a route";
see §2.) We cannot manufacture measured truth, but for *evaluating the temporal filter* we can
synthesize trajectories that are **physically and statistically faithful** to the real
environment, with known ground truth (`stage3_synthetic_eval.py`):

- **Physical fidelity (paths):** build an ε-graph over the real IT nodes (edges ≤ 1.6 m → one
  connected corridor network of 132 nodes), then take **shortest paths between random
  endpoints**. Trajectories thus follow real corridors with real turn geometry.
- **Statistical fidelity (sensors), calibrated to the data:**
  - speed ∈ [1.0, 1.35] m/s, step cadence ∈ [1.7, 2.0] Hz (per-walk);
  - heading = true path tangent + a slow gyro-drift random walk + **8.8° white noise**
    (the measured real `Orn_z` residual);
  - accelerometer magnitude = 9.81 + A·sin(2π f_step t) + noise → drives real step detection;
  - WiFi fixes at ~1 Hz are drawn from **real held-out static scans** at the nearest node, so
    measurement noise (~3 dBm per AP across repeat visits) and device variation are *real*.
- **No leakage:** the environment heatmap model is trained on a disjoint split of static visits
  from the scans used as fixes.

We generate 250 walks for training learned filters and 60 independent walks for evaluation,
yielding error *distributions* with 95% confidence intervals rather than a single number.

---

## 2. Failed Models and the Reasons Behind the Failures

Four earlier deep models were trained on the (fabricated) `Continuous_Fused_*.csv` data,
predicting position from WiFi + IMU windows.

| Model (file) | Architecture | Causal? | Output | Mean err (S9+) |
|---|---|---|---|---|
| `train_cnn_lstm.py` | Multi-branch CNN + **Bi-LSTM** + MaxPool | ✗ (look-ahead) | absolute (x,y) | 5.04 m (max 17.4) |
| `train_causal_hybrid.py` | CausalConv + uni-LSTM | ✓ | **Δ** integrated | 3.64 m |
| `train_env_model.py` | 3-branch, body-frame mag | ✓ | absolute (x,y) | 6.36 m (reversed **63 m**) |
| `train_ekf_fusion.py` | learned-α neural EKF | ✓ | fused absolute | 4.29 m |

**Why they failed (root causes):**

1. **Trained on a single trajectory; ignored the environment map.** All continuous walks trace
   the *same* corridor; after windowing this is ≈ 325 train / 99 test windows — a 1-D line. The
   model can only learn "where along this line am I," never a 2-D fingerprint field. Meanwhile
   the dense static database (the actual environment) was never used for training.
2. **Direction-contaminated magnetics.** Feeding raw body-frame `Mag_{x,y,z}` into a "spatial"
   branch makes the fingerprint depend on heading; this is why `train_env_model.py`'s
   reversed-path error exploded to **63 m**.
3. **Single-point regression vs. multimodal ambiguity.** A WiFi/magnetic reading matches many
   places; regressing one `(x,y)` averages the candidates, biasing toward the path centre.
4. **Look-ahead (non-causality).** The Bi-LSTM and temporal MaxPool in `train_cnn_lstm.py` read
   future frames — invalid for a real-time system.
5. **Fabricated ground truth.** The target path was synthesized from the static node ordering in
   a different building, so the reported errors measured fit to an artifact, not real positioning.

The corrective principles — train the environment on the static database, make the spatial
branch direction-invariant, output a multimodal heatmap, keep every component strictly causal,
and never trust the fabricated trajectory — define the architecture below.

---

## 3. Real-Time Processing of WiFi, Magnetic and IMU Data

All three pipelines are **strictly causal and streaming**: every transform uses only past and
present samples, and the exact same transform is applied when building training data and at
inference (*train == deploy*). This subsection covers per-modality processing and its effects.

### 3.1 WiFi — the absolute anchor

A scan is an RSS vector over the building's AP vocabulary, arriving sparsely (~1 Hz). We encode
it causally per frame:
```
x_ap = clip(RSS, -90, -30);  x_ap = (x_ap + 90)/60  in [0,1];  absent AP -> 0
```
WiFi is **direction-independent** by nature and was empirically the strongest spatial signal:
same-node cosine similarity **0.83** vs. different-node **0.45**. It is therefore used as the
drift-free *absolute* anchor (the measurement in the filter). Effect: provides global position
information that arrests inertial drift, at the cost of being noisy/intermittent.

### 3.2 Magnetic — online self-calibration to a device-invariant relative field

Raw magnetometer readings are device-dependent (hard-iron offset + soft-iron scale distortion):
measured per-device bias was LG G6/Q6 ≈ −3.8 µT and G7 noisy (σ = 8.2 µT) vs. ~0 for S8/A8/S9+.
A model normalized with *dataset-global per-device statistics* would fail on an unseen phone, so
each phone must calibrate **from its own live stream** (`mag_normalizer.py`, `OnlineMagNormalizer`):

1. **Rotation-invariant features** (gravity `ĝ` estimated from the accelerometer):
   ```
   magN = |M|,   magV = M·ĝ,   magH = sqrt(|M|² − magV²),   dip = atan2(magV, magH)
   ```
   These are orientation-independent → a property of the place, not the phone's pose.
2. **Online hard/soft-iron calibration:** a rolling ellipsoid fit on a trailing buffer maps the
   magnetometer's distorted sphere back to a sphere, with a numerically-stable **sphere-fit
   fallback** and a **diversity gate** (only trust the ellipsoid when the phone has actually
   rotated). Effect: cross-device mean-|M| spread dropped **3× (σ 1.77 → 0.57 µT)**.
3. **Causal running normalization** (trailing EMA mean/std) → the *relative magnetic anomaly*.

Magnetic is a **weak per-point** fingerprint (between-node/within-node spread ratio ≈ 1.3–1.5)
and its discriminative power lives in the **moving spatial profile**, not a single sample.
Consequence (design): WiFi anchors absolute position; the self-normalized magnetic profile is a
*relative* signal for the temporal/fusion stage. (The calibrator is built and validated; full
magnetic-as-measurement integration is identified as future work in §6.)

### 3.3 IMU — causal pedestrian dead-reckoning (PDR)

The motion estimate is built causally from accelerometer + heading:
- **Step detection:** an online peak detector on high-passed `|acc|` with a refractory period.
- **Heading:** `Orn_z + offset`, where the constant frame offset (compass ↔ map) is calibrated
  once (8.8° residual, §1.1).
- **Displacement:** each detected step advances position by a fixed step length `L` along the
  heading: `u_t = L·[cos θ_t, sin θ_t]`.

Effect: IMU gives a *smooth, high-rate* relative motion estimate that is locally accurate but
**drifts** without correction (demonstrated on real walks: ~1.5 m over short corridors, tens of
metres over long ones). This is the complementary opposite of WiFi (drift-free but jumpy),
motivating fusion.

---

## 4. The Environment Map: an MLP that learns the place

### 4.1 Why an MLP and a heatmap (the reasoning)

The environment model must satisfy three properties that the failed models violated:

- **Direction-invariance** → it must not see temporal order. A **per-frame** network (no
  recurrence, no temporal convolution) that consumes only the WiFi scan is *trivially*
  direction-invariant: the same scan yields the same output regardless of walking direction.
  This is what forces it to learn the *environment*, not the trajectory.
- **Multimodal output** → because fingerprints are ambiguous, the model outputs a probability
  **heatmap over floor cells** rather than a single coordinate, honestly representing "I could
  be in one of these places." The downstream filter resolves the ambiguity over time.
- **Trained on dense, real ground truth** → the static fingerprint database (167/168 nodes),
  not the single continuous walk.

### 4.2 Architecture and training (`stage2_wifi_heatmap.py`)

```
WiFi scan x ∈ ℝ^A  →  MLP[ A→256→256→ (n_cells) ]  →  softmax  →  belief map p ∈ Δ^{n_cells}
```
The floor is discretized into 1 m cells (IT: 90 × 14 = 1260 cells). The training target is a
**Gaussian blob** (σ = 2 m) centred on the node, normalized to a distribution; the loss is the
**KL divergence** between predicted and target distributions:
```
L = Σ_c  t_c · ( log t_c − log p_c )
```
At inference the position estimate is the **soft-argmax** (probability-weighted centroid), and
the belief's spatial spread yields a measurement covariance `R` for the filter:
```
ẑ = Σ_c p_c · coord_c ,    R = Σ_c p_c (coord_c − ẑ)(coord_c − ẑ)ᵀ
```
Single floor: Stairs/Room modes are excluded (they reuse corridor `(X,Y)` at a different height
and would alias the 2-D map); multi-floor is future work (a per-floor heatmap + pressure
altitude channel).

### 4.3 Impact (results)

| Evaluation | Mean error | Median |
|---|---|---|
| IT, random visit split | **1.43 m** | 1.12 m |
| IT, **held-out device** (S9+ never trained on) | **2.02 m** | 1.51 m |

Generalization study (`eval_environment.py`):

- **Cross-building** (per-building model): random-split MAE 1.42 m (IT) to 4.11 m (BE);
  **leave-one-phone-out** 2.2–5.1 m — device generalization is a property of the method.
- **AP-dropout robustness:** dropping 20% / 30% of APs at inference costs only ~0.9 m / 1.7 m.
- **Environment-not-trajectory proof (cross-scenario):** train on Scenario-1, test on a
  *different* walk direction (Scenario-2): **2.16 m on shared nodes**, i.e. essentially the
  within-scenario number. The model localizes a *new path* because it learned the place.

---

## 5. KalmanNet: a learned, non-linear Kalman filter

### 5.1 Why not the classical (linear-Gaussian) Kalman filter

A Kalman filter assumes a linear state transition, a linear measurement, and Gaussian noise. In
our system:

- The **state dynamics are exactly linear**: state `x = (px, py)`, transition `xₜ = xₜ₋₁ + uₜ`
  (position is additive in displacement) — no approximation needed.
- The hard **RSS → position non-linearity is already absorbed** by the heatmap MLP (§4): by the
  time the filter sees a measurement, it is a position `ẑ`, so `H = I` is linear *because* the
  network did the non-linear work.
- The assumption that **actually bites is Gaussian, unimodal measurement noise.** The heatmap is
  genuinely *multimodal*; collapsing it to one mean + a Gaussian `R` discards that structure, and
  a fixed Kalman gain cannot adapt to context (good vs. poor WiFi locality, time since last fix).

So the right object to relax is the **gain/update**, not the dynamics. KalmanNet does exactly this.

### 5.2 Architecture (`stage3_kalmannet.py`)

KalmanNet keeps the Kalman **predict → innovate → correct** recursion but produces the gain with
a recurrent network, so it needs no `F, H, Q, R` and makes no linear-Gaussian assumption:

```
predict:   x_pred  = x_{t-1} + u_t                      # known motion (PDR displacement)
innovate:  y_t     = z_t − x_pred        (only if a WiFi fix is present this step)
gain:      h_t     = GRUCell( features_t , h_{t-1} )    # tracks uncertainty implicitly
           K_t     = reshape( Linear(h_t) , 2×2 )       # LEARNED matrix Kalman gain
correct:   x_t     = x_pred + mask_t · ( K_t · y_t )
```
The GRU input features mirror KalmanNet's design — innovation `y_t`, observation difference
`Δz`, the motion control `u_t`, the previous state update `Δx̂`, and the observation mask
(7-D). A 64-unit GRU implicitly carries the role of the covariance `P`. The gain is a full **2×2
matrix** (vs. the scalar α of the earlier neural fusion), so it can apply different, cross-coupled
corrections in x and y. Computation is done in a start-centred frame for numerical stability and
is **strictly causal** (GRUCell stepping, autoregressive innovation; gain masked to zero when no
WiFi observation is present, so between fixes the filter coasts on motion).

### 5.3 Training process

Trajectories are binned to a fixed length (T = 160) so each walk is a sequence of `(u_t, z_t,
mask_t)`. The network is trained on 250 synthetic walks (Adam, lr 2e-3, 150 epochs) with a
mean-squared-error loss on the produced position sequence:
```
L = (1/T) Σ_t ‖ x_t − x_t^{true} ‖²
```
The gain layer is initialized to a near-diagonal 0.5 (sane KF-like behaviour at start). Because
the synthetic generator supplies effectively unlimited ground-truth trajectories, the learned
filter is trainable **without the route-memorization risk** that originally forced a classical
filter when only one fabricated path existed.

### 5.4 Why KalmanNet over the classical EKF (and the scalar-α neural fusion) — results

Four fusion mechanisms, identical inputs (heatmap fix + PDR motion), identical 60 held-out walks:

| Fusion mechanism | Assumption relaxed | Mean error | Median |
|---|---|---|---|
| WiFi-only | (no motion fusion) | 1.46 ± 0.05 m | 1.44 m |
| Classical EKF | linear + Gaussian + fixed gain | 0.99 ± 0.04 m | 0.98 m |
| Neural fusion (scalar α) | learned *scalar* gain | 0.62 ± 0.04 m | 0.62 m |
| **KalmanNet (matrix gain)** | **learned matrix gain, no linear-Gaussian** | **0.54 ± 0.05 m** | **0.49 m** |

KalmanNet improves on the classical EKF by **+46%** and on the scalar-α fusion by **+13%**,
empirically confirming that the linear-Gaussian assumption costs accuracy here and that the gain
is recovered by a richer, learned, context-dependent gain. It also retains the Kalman inductive
bias (innovation-driven correction), making it more data-efficient and robust than a generic
LSTM regressor. *Caveat:* it is trained on synthetic motion, so this is an in-distribution
result; robustness to real gait distribution-shift is untested (a non-learned alternative — a
histogram/grid Bayes filter that propagates the full multimodal belief — remains available if an
interpretable, training-free non-linear filter is preferred).

---

## 6. The Full Architecture: how the units work together

### 6.1 Data flow and intermediate outputs

```
                          ┌──────────────────────────────────────────────┐
  WiFi scan (≈1 Hz)       │ ENVIRONMENT MODEL (per-frame MLP, §4)         │
  RSS ∈ ℝ^A  ───encode──► │  → belief heatmap  p ∈ Δ^{n_cells}           │
                          │  → soft-argmax fix  ẑ ∈ ℝ²                    │
                          │  → covariance       R ∈ ℝ^{2×2} (map spread)  │
                          └───────────────┬──────────────────────────────┘
                                          │  z_t , (R_t)            measurement
  Magnetometer (16.7 Hz)  ┌───────────────┴───────────────┐               │
  Mag,Acc ∈ ℝ^3 ────────► │ ONLINE MAG NORMALIZER (§3.2)   │ rel. anomaly  │
                          │  → device-invariant features ∈ ℝ^8 │ (profile) │
                          └────────────────────────────────┘               ▼
  IMU (Acc, Orn_z)        ┌────────────────────────────────┐      ┌──────────────────┐
  ───────────────────────►│ PDR (§3.3): step + heading     │ uₜ   │  KALMANNET (§5)  │
                          │  → displacement u_t ∈ ℝ² /step  │─────►│ predict-innovate- │──► x̂_t ∈ ℝ²
                          └────────────────────────────────┘      │ correct, matrix K │   position
                                                                   └──────────────────┘
```

**Intermediate outputs, with shapes and how the next unit consumes them:**

1. **Environment model output.** Input: encoded WiFi scan `x ∈ [0,1]^A` (A = 250 for IT). Output:
   a probability map `p ∈ Δ^{1260}` (the *belief*), from which we compute the **position fix**
   `ẑ ∈ ℝ²` (soft-argmax) and **measurement covariance** `R ∈ ℝ^{2×2}` (belief spread). *Consumed
   by:* KalmanNet as the measurement `z_t`; the spread informs how much to trust it.
2. **Magnetic normalizer output.** Input: streaming `(Mag, Acc) ∈ ℝ^3 × ℝ^3` at 16.7 Hz. Output:
   8 device-invariant features (calibrated `magN, magV, magH, dip` + their running-normalized
   relative anomalies). *Role:* a relative spatial profile; in the current best-validated fusion
   the absolute anchor is WiFi, with magnetic integration as the designated next step (§6.3).
3. **PDR output.** Input: `|Acc|` and `Orn_z` per frame. Output: a per-step displacement vector
   `u_t ∈ ℝ²` in the world frame (zero on non-step frames). *Consumed by:* KalmanNet as the motion
   control in the predict step `x_pred = x_{t-1} + u_t`.
4. **KalmanNet state.** Maintains position `x ∈ ℝ²` and GRU hidden state `h ∈ ℝ^{64}` (implicit
   uncertainty). At each step it predicts with `u_t`, forms the innovation `z_t − x_pred` when a
   WiFi fix is available, computes a matrix gain `K_t ∈ ℝ^{2×2}` from `h_t`, and corrects. *Output:*
   the fused position estimate `x̂_t`, emitted every frame in real time.

### 6.2 End-to-end behaviour

At runtime the IMU (high rate) continuously advances the KalmanNet state via PDR — smooth but
drift-prone. Roughly once per second a WiFi scan passes through the environment MLP, yielding a
drift-free (but noisy, possibly multimodal) position fix; the learned gain decides — *per step,
from context* — how strongly to pull the estimate toward that fix. The result is a trajectory
that is simultaneously **smooth** (from inertial motion) and **drift-free** (from the WiFi
environment anchor), produced causally for real-time use. Quantitatively the components compose
as: WiFi-only 1.46 m → +IMU via classical EKF 0.99 m → +learned matrix-gain fusion **0.54 m**.

### 6.3 Status, constraints, and future work

- **Validated and real:** the environment model (real static ground truth; 1.43 m / 2.02 m
  held-out device; campus-wide and cross-scenario generalization) and the online magnetic
  self-calibration (3× cross-device improvement).
- **Validated on faithful synthetic trajectories:** the EKF and KalmanNet fusion (0.99 m vs.
  0.54 m), because **no measured trajectory ground truth exists** in the dataset.
- **Designed, integration pending:** magnetic as a *second measurement* (sequence/relative-profile
  matching) inside the filter; the normalizer is built but magnetic currently contributes as a
  relative profile rather than an absolute anchor.
- **Unblocking step:** a small data-collection campaign of continuous walks that **log WiFi
  (~1 Hz) and sparse position checkpoints** would convert the synthetic-GT fusion numbers into
  end-to-end real-world accuracy, and would test the learned filter under real gait distribution.

### 6.4 Component & code map

| Stage | Script | Output artifact |
|---|---|---|
| Fingerprint DB build | `dl_models/build_fingerprint_db.py` | `Datasets/fingerprint_db/<bldg>/nodes.csv`, `bssid_vocab.json` |
| Online magnetic calibration | `dl_models/mag_normalizer.py` | `OnlineMagNormalizer` (streaming module) |
| Environment model (MLP heatmap) | `dl_models/stage2_wifi_heatmap.py` | `best_wifi_heatmap.pth`, heatmap figures |
| Generalization / robustness eval | `dl_models/eval_environment.py` | `env_eval_*.png/csv` |
| Trajectory synthesis + EKF | `dl_models/stage3_synthetic_eval.py` | `stage3_synth_eval.png/csv` |
| Scalar-α neural fusion | `dl_models/stage3_neural_fusion.py` | `stage3_neural_vs_ekf.png` |
| KalmanNet (learned matrix gain) | `dl_models/stage3_kalmannet.py` | `stage3_kalmannet.png` |

---

*All errors are Euclidean position errors in metres. Synthetic-trajectory results are reported as
mean ± 95% CI over 60 independent held-out walks. Every runtime component is strictly causal; the
same transforms are applied at training and inference (train == deploy).*
