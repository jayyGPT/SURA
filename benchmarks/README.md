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

## CNN-output DualKalmanNet experiments

The experiments that actually feed the magnetic CNN's 2-D output into KalmanNet are recorded in:

- `cnn_dual_kalmannet_full_run.md` — first unweighted CNN-output run
- `cnn_dual_kalmannet_training_history.csv` — all 150 epochs from that unweighted run
- `cnn_dual_kalmannet_relative_variance.md` — current relative-variance weighted result and comparison
- `cnn_dual_kalmannet_relative_variance_metrics.json` — exact machine-readable weighted metrics
- `cnn_dual_kalmannet_relative_variance_training_history.csv` — all 150 epochs from the weighted run
- `magnetic_variance_calibration/` — comparison of CNN-predicted uncertainty with actual magnetic error
- `analyze_magnetic_variance.py` — reproducible variance/error analysis

### Current best CNN-output variant

The relative-variance model computes a reference uncertainty from the fusion **training set only** and weights the magnetic correction by

\[
w_{mag}=\frac{1}{1+\sigma_{mag}^2/\sigma_{ref}^2}.
\]

Full 250-train / 60-test / 150-epoch comparison:

| Wi-Fi regime | Wi-Fi-only | Unweighted CNN Dual | Relative-variance CNN Dual |
|---|---:|---:|---:|
| Full Wi-Fi (1 Hz) | **0.473 m** | 0.506 m | 0.494 m |
| Degraded Wi-Fi (5 s, 40% AP dropout) | 1.533 m | 1.171 m | **1.154 m** |

The relative weight improves the CNN-output model in both regimes. Under degraded Wi-Fi it gives a **24.7% mean-error reduction versus Wi-Fi-only**, and lowers P90 from 2.064 m in the unweighted CNN Dual to **1.612 m**. Under full Wi-Fi it reduces the CNN-Dual penalty from 7.0% to 4.5%, although the Wi-Fi-only mean remains slightly better.

The variance calibration experiment explains why relative weighting is preferable to treating the CNN variance as an absolute covariance: uncertainty ranks weak magnetic fixes usefully, but its raw scale is substantially over-conservative.

## Generated runs

Training scripts can write checkpoints, metrics, prediction arrays, error CDFs, and training curves under `benchmarks/runs/`. That directory is ignored because it can become large. Results worth preserving should be summarized here or copied into a labelled benchmark file.

## Older baseline

The older KNN proof-of-concept code and figures are kept under `knn/`.
