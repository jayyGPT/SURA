# Benchmarks

This folder is the single place for benchmark numbers and comparison baselines.

## Current manuscript / legacy results

These are the numbers currently carried by the manuscript or older experiment records. They remain useful for comparison, but they are not all reproduced under the new CNN-output fusion implementation.

| Model / condition | Mean error | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Wi-Fi heatmap, random split | 1.43 m | 1.12 m | 2.65 m | 7.84 m |
| Wi-Fi heatmap, S9+ held out | 2.02 m | 1.68 m | 3.74 m | 9.12 m |
| Magnetic sequence CNN, 84 frames | 3.58 m | — | — | — |
| Wi-Fi-only KalmanNet, full Wi-Fi | 0.55 m | — | — | — |
| Legacy anomaly DualKalmanNet, full Wi-Fi | 0.47 m | — | — | — |
| Wi-Fi-only KalmanNet, degraded Wi-Fi | 1.44 m | — | — | — |
| Legacy anomaly DualKalmanNet, degraded Wi-Fi | 1.07 m | — | — | — |

Machine-readable legacy values are in `results.yaml`.

## CNN-output DualKalmanNet experiment

The first full run that actually feeds the magnetic CNN's 2-D output into KalmanNet is recorded in:

- `cnn_dual_kalmannet_full_run.md` — setup, final 60-walk results, interpretation
- `cnn_dual_kalmannet_training_history.csv` — all 150 training epochs for both regimes and both models
- `magnetic_variance_calibration/` — comparison of CNN-predicted uncertainty with actual magnetic position error
- `analyze_magnetic_variance.py` — reproducible variance/error analysis

Headline result from the 250-train / 60-test / 150-epoch run:

| Wi-Fi regime | Wi-Fi-only KalmanNet | CNN-output DualKalmanNet | Relative change |
|---|---:|---:|---:|
| Full Wi-Fi (1 Hz) | **0.473 m** | 0.506 m | -7.0% |
| Degraded Wi-Fi (5 s, 40% AP dropout) | 1.533 m | **1.171 m** | **+23.6%** |

The CNN variance head is useful for ranking magnetic fixes (least-confident quartile has about 2.4x the mean error of the most-confident quartile), but the raw variance is overestimated by roughly 4.4x relative to empirical per-axis variance. Therefore the next fusion experiment should use **relative/calibrated confidence**, not blindly treat the raw CNN variance as an absolute covariance.

## Generated runs

Training scripts can write checkpoints, metrics, prediction arrays, error CDFs, and training curves under `benchmarks/runs/`. That directory is ignored because it can become large. Results worth preserving should be summarized here or copied into a labelled benchmark file.

## Older baseline

The older KNN proof-of-concept code and figures are kept under `knn/`.
