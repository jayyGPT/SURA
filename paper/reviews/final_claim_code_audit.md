# Final claim/code audit

Scope: active manuscript `paper/main.tex`, active code, corrected processed database, selected checkpoints, and canonical seed-4 result artifacts. Historical material is not treated as evidence.

## Data and preprocessing

| Manuscript item | Active evidence | Status |
|---|---|---|
| Wi-Fi exact registration to magnetic survey frame | `tools/fingerprint_builder.py:27,159-279`; `data/processed/fingerprint_db/it_engineering/pairing_audit.json` | Verified |
| Wi-Fi RSSI normalization, Eq. `wifi_norm` | `models/wifi_heatmap.py:64-78` | Verified |
| Magnetic gravity proxy/features, Eqs. `mag_gravity_proxy`-`mag_feature_vector` | `tools/fingerprint_builder.py:79-110` | Verified |
| Per-phone centering/node interpolation | `train/kalmannet_wifiheatmap_magneticCNN_pdr.py:280-330` and `train/train_magnetic_sequence.py` magnetic-map builder | Verified |
| 1 m heatmap/environment grid | `models/wifi_heatmap.py:23-61`; training config `GRID_CELL_M=1.0` | Verified |

The database contains 2,245 parsed magnetic static visits, 969 unique exact Wi-Fi attachments, 250 AP identities, and 168 unique magnetic survey nodes. Of the exact attachments, 759 fall in the three modes used by the paper. Raw Wi-Fi coordinates are retained separately and are not treated as the canonical localization coordinate.

## Measurement models and equations

| Equation/claim | Active evidence | Status |
|---|---|---|
| Wi-Fi softmax and centroid, Eqs. `softmax_def`, `heatmap_centroid` | `models/wifi_heatmap.py:111-117` | Verified |
| Gaussian heatmap target, Eq. `target_q` | `models/wifi_heatmap.py:52-61` | Verified |
| KL objective, Eq. `kl_loss` | `models/wifi_heatmap.py:129-134` | Verified |
| Magnetic 84x4 causal window | `train/kalmannet_wifiheatmap_magneticCNN_pdr.py:532-563`; selected checkpoint metadata | Verified |
| Magnetic CNN topology | `models/magnetic_sequence_cnn.py:11-66` | Verified |
| Magnetic uncertainty-weighted objective, Eq. `nll_loss` | `models/magnetic_sequence_cnn.py:69-92` | Verified, with manuscript caveat that it is not exact 2-D Gaussian NLL |
| EMA/step detector/PDR displacement, Eqs. `lpf`-`pdr` | `models/pdr.py` and `build_sequence()` at `train/...py:565+` | Verified |
| Synthetic heading, Eq. `sim_heading` | `causal_path_heading()` at `train/...py:457+` and `synthesize_walk()` | Verified causal: current/previous positions only, then drift + white noise |

## DualKalmanNet equations

| Equation/claim | Active evidence | Status |
|---|---|---|
| Prior `x^- = x_{t-1}+u_t` | `train/kalmannet_wifiheatmap_magneticCNN_pdr.py:173` | Verified |
| Two Cartesian innovations | lines 175-177 | Verified |
| 13-D GRU feature vector | lines 182-203 | Verified |
| GRU + 8-value gain head | class definition and lines 204-207 | Verified |
| Two separate 2x2 gains | lines 206-207 | Verified; manuscript says separate, not statistically independent |
| Training-reference magnetic score | final fusion code lines 921-925 | Verified training-only |
| Relative magnetic confidence | lines 186-190 | Verified, including clipping |
| Posterior correction | lines 209-215 | Verified |
| Availability-mask behavior | lines 175-216 | Verified |

The analytical EKF gain equation in the manuscript is background/contrast and is not claimed as an implemented operation.

## Evaluation protocol

- Survey-node graph uses Euclidean epsilon=1.6 m; no wall or obstacle geometry is used.
- PDR heading is causal at the simulator-to-estimator boundary.
- Fusion training: seed 1, 250 trajectories.
- Development protocol retained before final freeze: seed 2.
- Seed 3 is retired because it was inspected before correction of the Wi-Fi/magnetic registration bug.
- Final test: seed 4, 60 trajectories x 160 bins, first inspected after the corrected database/checkpoints/metrics were frozen.
- Exact binned-target trajectory overlap between train and final: zero in both regimes. This check does not claim zero shared graph segments.
- KNN fills missing magnetic log-uncertainty with a training-derived value and selects K by five-fold grouped CV on training trajectories only (`benchmarks/knn/wifi_mag_knn.py:317-401`).
- Mean/CI use 60 per-trajectory means; median/P90/max/CDF use 9,600 pointwise errors (`train/...py:816-841`).
- Reported CI excludes retraining variability, as stated in the manuscript.

## Numerical claims

All paper-facing temporal numbers are copied from `benchmarks/final_protocol/current_results/metrics.json`. No old seed-2/seed-3 headline value remains in the active manuscript.

## Residual scope limitations (not blockers)

- Synthetic temporal ground truth, not real continuously labeled trajectories.
- Known initial 2-D position.
- Fixed surveyed environment; no building-independent claim.
- Magnetic fusion assumes the per-phone-centered survey feature domain; causal alignment of an uncalibrated unseen phone is not evaluated.
- Graph is proximity-based and not obstacle-aware.
- CI measures trajectory variability for one trained model, not retraining variability.
