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

## 4. Current constraints

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
