# Final protocol and canonical results

This directory contains the only paper-facing result set. Earlier development experiments are not canonical.

## Data/model freeze

- Wi-Fi/magnetic database registration: exact mode + scenario + phone + user + timestamped filename; magnetic static survey coordinate is the common map frame.
- Selected Wi-Fi checkpoint: `../../checkpoints/wifi_heatmap.pt`.
- Selected magnetic checkpoint: `../../checkpoints/magnetic_sequence.pt`, 84-frame window.
- Fusion training seed: 1.
- Development seed used before final freeze: 2.
- Seed 3 is retired because it had been inspected before the database-registration correction.
- Final-test seed: 4, first inspected after the corrected database, models, causal heading, KNN preprocessing, and metric definitions were frozen.

The trajectory overlap check rejects exact duplicate binned target trajectories; it does not claim zero shared path segments.

## Metrics

Mean error and the reported 95% half-width use the 60 per-trajectory mean errors. The half-width is `1.96 * SD / sqrt(60)` for one fixed trained model and does not include retraining variability. Median, P90, maximum, and CDF use all `60 x 160 = 9600` pointwise errors.

| Regime | Model | Mean ± half-width (m) | Pointwise median (m) | Pointwise P90 (m) |
|---|---|---:|---:|---:|
| Full Wi-Fi | Wi-Fi-only KalmanNet | 0.523 ± 0.052 | 0.423 | 1.028 |
| Full Wi-Fi | Wi-Fi + magnetic KNN | 0.745 ± 0.034 | 0.510 | 1.613 |
| Full Wi-Fi | CNN Dual + relative confidence | **0.490 ± 0.046** | **0.403** | **0.976** |
| Degraded Wi-Fi | Wi-Fi-only KalmanNet | 1.734 ± 0.243 | 1.243 | 3.767 |
| Degraded Wi-Fi | Wi-Fi + magnetic KNN | 2.019 ± 0.187 | 1.282 | 4.412 |
| Degraded Wi-Fi | CNN Dual + relative confidence | **0.900 ± 0.101** | **0.710** | **1.845** |

Relative mean reduction of the weighted Dual model versus Wi-Fi-only KalmanNet is 6.38% in full Wi-Fi and 48.07% in degraded Wi-Fi. KNN `K` is selected by five-fold grouped CV on training trajectories only: `K=5` (full) and `K=20` (degraded).

## Canonical artifacts

- `current_results/metrics.json`: combined machine-readable result/protocol record.
- `current_results/final_cdf.png`: paper CDF figure.
- `current_results/fusion/<regime>/`: trained final filter checkpoints and predictions/errors.
- `current_results/knn/<regime>/`: KNN predictions/errors and grouped-CV record.
- `wifi_heatmap/corrected_paired/`: selected Wi-Fi development run and checkpoint provenance.
- `magnetic_sequence/magnetic_frozen/`: selected magnetic development run and checkpoint provenance.

The generated train/final array caches used to work around runtime limits are not part of the cleaned repository because they are deterministic from the checked-in code, checkpoints, seeds, and processed database.
