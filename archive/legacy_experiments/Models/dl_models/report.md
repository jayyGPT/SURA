# SURA Indoor Positioning — Modeling Report

**Goal:** real-time indoor position from magnetic + WiFi + IMU. The model must learn
the **environment** (the spatially-unique, temporally-stable mag/WiFi fingerprint),
**not the trajectory** (the specific walked path), and must be **strictly causal**
(no look-ahead) so it can run in real time.

---

## 1. Earlier deep-learning models (and their results)

All earlier models trained on `Continuous_Fused_*.csv` (A8/G7/S8 train, S9+ test) and
predicted position from WiFi + IMU windows.

| Model | File | Causal? | Predicts | Mean err (S9+) |
|---|---|---|---|---|
| Multi-branch CNN + **Bi**-LSTM | `train_cnn_lstm.py` | ❌ Bi-LSTM + MaxPool look ahead | absolute (X,Y) | 5.04 m (max 17.4) |
| Causal hybrid (CausalConv + uni-LSTM) | `train_causal_hybrid.py` | ✅ | **deltas** → integrated | 3.64 m |
| Three-branch environment model | `train_env_model.py` | ✅ | absolute (X,Y) | 6.36 m (reversed max **63 m**) |
| Neural EKF (learned α-gate fusion) | `train_ekf_fusion.py` | ✅ | fused absolute | 4.29 m |

The progression already moved in the right direction — causal convolutions, separating
direction-invariant spatial branches from temporal IMU, and a learned complementary
filter — but the headline errors (3.6–6.4 m) were misleading (see §2).

---

## 2. What was wrong with them

1. **They trained on a single trajectory, and ignored the real environment map.**
   All four continuous walks trace the *same corridor* (same start/end, 2 turns). After
   downsampling + windowing that is ~325 train / 99 test windows — a single 1-D line.
   There is no off-path spatial coverage, so the network can only memorize *where along
   this line am I*, not a 2-D environment fingerprint. Meanwhile the **538 static grid
   nodes + matched WiFi scans** — the dense, accurate fingerprint database — were never
   used by any training script.

2. **The magnetic "spatial fingerprint" was secretly direction-dependent.**
   Raw `Mag_x/Mag_y/Mag_z` are in the *phone body frame* and rotate with heading, so the
   same spot produced different values per walking direction. This is why the env model's
   reversed-path error exploded to **63 m**.

3. **Single-(X,Y) regression fights fingerprint ambiguity.** One mag/WiFi reading matches
   many locations; regressing a single point *averages* the candidates, biasing toward the
   path center.

4. **The "ground truth" was fabricated.** `Continuous_Fused_*.csv` was built by
   `_temp/Used Scripts/fuse_continuous_wifi.py`, which (a) used **BE Building, not IT**,
   (b) **synthesized** `True_X/True_Y` by linearly interpolating the static node list by
   timestamp, and (c) **simulated** WiFi with injected noise. So the models were scored
   against an invented path whose "truth" was essentially the static node ordering — which
   is *why they looked like they were memorizing a route*.

---

## 3. New architecture — a learned Bayesian filter

A clean split between a learned **environment/measurement model** and a classical
**causal motion filter**:

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

**Design principles**

- **Environment is learned on the dense static DB, not the walk.** This is the only data
  that actually samples the 2-D space (167/168 nodes covered).
- **Environment branch is direction-invariant** (per-frame, WiFi-only) → it learns the
  *environment*, never the direction or path.
- **Heatmap output** honestly represents fingerprint ambiguity; the filter resolves it
  over time and supplies the EKF measurement covariance directly from the heatmap spread.
- **Online magnetic self-calibration** (rolling hard/soft-iron + causal running-norm),
  computed from each phone's own live stream → generalizes to **unseen devices** with no
  dataset-specific calibration.
- **Classical EKF fusion** (almost no learned params) → cannot memorize a route; WiFi is
  the drift-free anchor, IMU PDR smooths between fixes. Fully causal / real-time.
- **Floor-aware, single-floor for now**: position = (floor index → 2-D heatmap) +
  session-relative pressure altitude channel. Full 3-D supervision is Phase 2.

**Results so far**

| Component | File | Result |
|---|---|---|
| Fingerprint DB | `build_fingerprint_db.py` | 168 IT-Eng nodes, real mag + 250-AP WiFi; WiFi covers 167/168 |
| Online mag normalizer | `mag_normalizer.py` | cross-device mean \|M\| spread 3× lower (std 1.77 → 0.57) |
| **WiFi heatmap env model** | `stage2_wifi_heatmap.py` | **MAE 1.43 m** (random) / **2.02 m on a held-out device (S9+)** |
| Real-sensor motion demo | `stage3_realdemo.py` | causal PDR ~1.5 m on-corridor (short walks); drifts on long walks |

The environment model **beats every prior model while generalizing across devices** — and
it does so without any trajectory, look-ahead, or fabricated ground truth.

---

## 4. Generalization & robustness results (paper evidence)

Evaluated with `eval_environment.py`. Each building uses its own AP vocabulary and grid
(per-environment model); single floor (Stairs/Room excluded).

**Pillar 1+2 — cross-building accuracy & device generalization (leave-one-phone-out):**

| Building | Nodes | Visits | APs | Random-split MAE | Leave-one-phone-out MAE |
|---|---|---|---|---|---|
| IT Engineering | 132 | 816 | 250 | **1.42 m** (med 1.03) | **2.21 m** |
| CS Engineering | 75 | 173 | 154 | 3.30 m (med 2.84) | 3.56 m |
| Electrical Eng. | 64 | 139 | 146 | 2.48 m (med 2.24) | 3.00 m |
| IACT | 64 | 162 | 121 | 2.65 m (med 2.35) | 2.42 m |
| BE Building | 105 | 235 | 174 | 4.11 m (med 2.42) | 5.06 m |

The model works **campus-wide**, and **leave-one-phone-out** (train on N−1 devices, test on a
fully held-out device) stays at 2–5 m — device generalization is a property of the method,
not a single lucky holdout. Accuracy tracks data density (IT best with 816 visits; BE worst,
where the G7/S9+ folds had the fewest visits). Figure: `Datasets/env_eval_buildings.png`.

**Pillar 3 — WiFi AP-dropout robustness (IT):** randomly removing APs at inference
(APs get added/removed over a building's life) degrades gracefully:

| APs dropped | 0% | 10% | 20% | 30% | 50% | 70% |
|---|---|---|---|---|---|---|
| MAE | 1.49 m | 1.80 m | 2.31 m | 3.23 m | 6.67 m | 11.48 m |

Up to ~30% AP loss costs only ~1.7 m. Figure: `Datasets/env_eval_ap_dropout.png`.

**Pillar 4 — environment, not trajectory (cross-scenario, IT):** train on **Scenario-1**
walks, test on **Scenario-2** (a *different* path/direction). On the spatial nodes shared by
both scenarios: **MAE 2.16 m (median 1.25 m)** — essentially the within-scenario number. The
model trained on one direction localizes a *different* walk because it learned the **place**,
not the route. (All-S2-nodes MAE 3.12 m includes nodes never surveyed in S1, i.e. genuine
spatial extrapolation.)

**Pillar 5 — causal EKF fusion gain (quantitative, on faithful synthetic trajectories).**
Because no *measured* trajectory ground truth exists (§5), the fusion is scored on
trajectories that are **physically and statistically faithful** to the real environment,
with known GT (`stage3_synthetic_eval.py`):

- *Physical:* paths walk the **real surveyed corridor graph** (ε-graph over IT nodes,
  shortest paths between random endpoints, full 132-node connected topology).
- *Statistical:* gait speed/cadence in measured ranges; heading = path tangent + gyro
  drift + the **8.8° noise measured from real `Orn_z`**; WiFi fixes (~1 Hz) drawn from
  **real held-out static scans** at the nearest node (real ~3 dBm measurement noise +
  device variation); the heatmap model is trained on a disjoint visit split (no leakage).
- 60 independent walks, 95% CI, with single-modality baselines:

| Method | Mean error | Median |
|---|---|---|
| PDR-only (IMU dead-reckoning) | 2.16 ± 0.40 m | 1.84 m |
| WiFi-only (heatmap fixes) | 1.38 ± 0.05 m | 1.37 m |
| **Causal EKF fusion** | **0.71 ± 0.06 m** | **0.66 m** |

The fusion roughly **halves** the error of the best single modality (**+49%** vs WiFi-only,
**+67%** vs PDR-only): the drift-free WiFi anchor removes PDR drift, while IMU motion
smooths the jumpy per-scan WiFi fixes. Figures: `Datasets/stage3_synth_eval.png`,
data `Datasets/stage3_synth_eval.csv`. (Caveat: synthetic motion; real-walk validation
still needs WiFi-logged collection — see §5.)

**Pillar 6 — neural Kalman fusion vs classical EKF (head-to-head).** With the synthetic
generator supplying unlimited GT trajectories, the learned fusion from `train_ekf_fusion.py`
(CausalConv+LSTM motion encoder + a **learned α-gate** Kalman gain, autoregressive update)
becomes trainable without the route-memorisation risk that originally forced a classical
filter. We swap *only* the fixed Kalman update for this learned gate; both filters consume
**identical inputs** (Stage-2 WiFi heatmap fix + per-bin IMU dead-reckoning) on identical
held-out walks (`stage3_neural_fusion.py`, 250 train / 60 test walks, strictly causal):

| Method (identical binned inputs) | Mean error | Median |
|---|---|---|
| WiFi-only | 1.41 ± 0.06 m | 1.39 m |
| Classical EKF | 0.93 ± 0.04 m | 0.93 m |
| **Neural Kalman fusion** | **0.60 ± 0.06 m** | **0.56 m** |

The learned fusion beats the classical EKF by **+35%**. It learns a *context-dependent* gain
(mean gated α≈0.17 — trusts the smoothed motion ~83%, using WiFi to arrest drift) and the
LSTM denoises per-step displacement — neither possible for a fixed-gain KF. Figure:
`Datasets/stage3_neural_vs_ekf.png`. *Caveat:* the neural fusion is trained on synthetic
walks, so this is an in-distribution result; robustness to real gait distribution shift is
untested (and is a further reason the WiFi-logged real-walk collection in §5 matters).

**Pillar 7 — beyond linear-Gaussian: KalmanNet (learned matrix gain).** The classical EKF
assumes a linear-Gaussian model. In our system the *state dynamics* are exactly linear
(`xₜ = xₜ₋₁ + uₜ`) and the RSS→position non-linearity is already absorbed by the heatmap
net; the assumption that actually bites is **Gaussian, unimodal measurement noise** (a WiFi
fingerprint is genuinely multimodal). We test relaxing the fixed linear-Gaussian update with
a **KalmanNet** (`stage3_kalmannet.py`): the KF predict→innovate→correct recursion is kept,
but the Kalman gain becomes a full 2×2 matrix produced by a GRU that implicitly tracks
uncertainty — no F/H/Q/R, no linear-Gaussian assumption. Four fusion mechanisms, identical
inputs / identical 60 held-out walks:

| Fusion mechanism | Assumption relaxed | Mean error | Median |
|---|---|---|---|
| WiFi-only | (no motion fusion) | 1.46 ± 0.05 m | 1.44 m |
| Classical EKF | linear + Gaussian + fixed gain | 0.99 ± 0.04 m | 0.98 m |
| Neural fusion (scalar α) | learned *scalar* gain | 0.62 ± 0.04 m | 0.62 m |
| **KalmanNet (matrix gain)** | **learned matrix gain, no linear-Gaussian** | **0.54 ± 0.05 m** | **0.49 m** |

KalmanNet beats the classical EKF by **+46%** and the scalar-α neural fusion by **+13%**,
confirming the linear-Gaussian assumption costs accuracy here and that the gain is recovered
by a richer, learned, cross-coupled gain. Figure: `Datasets/stage3_kalmannet.png`. *Same
in-distribution caveat as Pillar 6.* (A fully non-parametric alternative — a histogram/grid
Bayes filter that propagates the entire multimodal belief instead of collapsing it — remains
an option if a non-learned, interpretable nonlinear filter is preferred.)

**Pillar 8 — phone holding-mode (posture) robustness.** Whether a unit is posture-robust
depends on whether it consumes an orientation at all (`eval_holding_modes.py`, evaluated at
*shared nodes* so it is pure localization, not extrapolation):

| Test (IT, shared nodes) | MAE | Median |
|---|---|---|
| Navigation random split (reference) | 1.62 m | 1.11 m |
| Trajectory held-out (train Scenario-1 → test Scenario-2) | 2.16 m | 1.09 m |
| Posture held-out — Call listening (near ear) | 1.99 m | 1.74 m |
| Posture held-out — Swinging | 1.39 m | 1.01 m |
| Posture + device held-out — Call (S8, S8 absent from training) | 1.91 m | 1.43 m |
| Posture + device held-out — Swinging (S8, S8 absent from training) | 2.16 m | 1.83 m |

The **WiFi environment model is posture-robust by construction** (per-frame, no orientation
input): swinging held-out matches the within-posture split, and even posture+device held-out
stays ≈ 2 m. Call-near-ear is marginally worse, consistent with body attenuation of APs. The
**magnetic** features are made orientation-invariant by projecting onto the accelerometer
gravity vector, but that gravity estimate degrades under dynamic swing (a quasi-static/low-pass
gravity estimate would harden it). The **PDR heading** (`Orn_z + offset`) is robust for stable
holding (Navigation/Call) but not for swinging, where instantaneous heading oscillates at step
frequency; the remedy is stride-averaged heading / forward-direction PCA plus gain down-weighting.

*Data constraint:* Swinging/Call exist only on the **S8** phone, only in **Scenario-1**, and only
as **static dwells** (no swing/call walks). A fully combined posture+device+trajectory held-out
test, and any test of **PDR-under-swing on a real trajectory**, therefore require synthesis
(swing-corrupted heading in the generator) or new data collection.

## 5. Current constraints

1. **No measured per-frame trajectory ground truth exists.** Continuous recordings label
   only the **start node** (verified: a raw IT walk is `(90,24)` for all 2,893 rows). So we
   cannot train or *quantitatively* score a trajectory/fusion model on real data — only the
   per-node environment model has real labels.

2. **No WiFi was logged while walking.** WiFi in this dataset is always *static node scans*;
   continuous walks contain IMU + magnetometer only. So the WiFi correction (proven in
   Stage 2) cannot be fed on real walks — the real-sensor demo is dead-reckoning alone,
   which drifts without it.

3. **Pure dead-reckoning drifts on long walks.** Expected and unavoidable without an
   absolute anchor — exactly the role the WiFi heatmap fills when WiFi is available.

4. **Per-building scope.** WiFi AP sets and coordinate frames differ per building, so this
   is inherently a per-environment model (consistent with "learn *the* environment"). Work
   so far is IT Engineering only.

5. **Magnetic is a weak per-point fingerprint** (between/within-node ratio ~1.3–1.5) and
   device-dependent; its real signal is the *moving spatial profile*, which belongs in the
   temporal/fusion stage, not as a static point feature.

### What unlocks the complete real-world result
A small, targeted data-collection campaign: continuous walks that **log WiFi scans
(~1 Hz)** and **sparse position checkpoints** for ground truth. With that, the already-written
EKF fusion (`stage3_ekf_fusion.py`) drops in and becomes quantitatively testable end-to-end.
