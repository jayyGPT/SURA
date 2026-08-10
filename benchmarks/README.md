# Benchmarks

This folder is the single place for benchmark numbers and comparison baselines.

## Current recorded results

The values below come from the current manuscript / previous experiment records and are marked
for reproduction before final publication.

| Model / condition | Mean error | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Wi-Fi heatmap, random split | 1.43 m | 1.12 m | 2.65 m | 7.84 m |
| Wi-Fi heatmap, S9+ held out | 2.02 m | 1.68 m | 3.74 m | 9.12 m |
| Magnetic sequence CNN, 84 frames | 3.58 m | — | — | — |
| Wi-Fi-only KalmanNet, full Wi-Fi | 0.55 m | — | — | — |
| Legacy anomaly DualKalmanNet, full Wi-Fi | 0.47 m | — | — | — |
| Wi-Fi-only KalmanNet, degraded Wi-Fi | 1.44 m | — | — | — |
| Legacy anomaly DualKalmanNet, degraded Wi-Fi | 1.07 m | — | — | — |

Machine-readable values are in `results.yaml`.

## Status

The anomaly-based DualKalmanNet is a **legacy comparison**, not the architecture we are taking
forward. The next implementation will feed the magnetic CNN's 2-D position estimate (and its
predicted uncertainty) into DualKalmanNet. Once that experiment is complete, the benchmark table
and paper will be updated with newly reproduced numbers.

## Generated runs

Training scripts write checkpoints, metrics, prediction arrays, error CDFs, and training curves
under:

```text
benchmarks/runs/
```

That directory is ignored because it can become large. Results we want to preserve should be
copied into `results.yaml` after they are checked.

## Older baseline

The older KNN proof-of-concept code and figures are kept under:

```text
benchmarks/knn/
```
