# CNN-output DualKalmanNet with relative magnetic-variance weighting

This experiment adds a relative confidence weight to the magnetic-CNN correction while keeping the same Wi-Fi heatmap, magnetic CNN, PDR, GRU architecture, synthetic corridor protocol, training/test seeds, and run sizes as the unweighted CNN-output experiment.

## Weighting rule

The magnetic CNN predicts `log(sigma^2)`. For each Wi-Fi regime, a reference uncertainty is computed from the **fusion training set only**:

```text
reference_log_variance = median(training magnetic log-variance)
```

The relative weight is

\[
w_{mag} = \frac{1}{1 + \sigma_{mag}^2 / \sigma_{ref}^2}
\]

and the update is

\[
x_t = x_{pred} + K_{wifi} y_{wifi} + w_{mag} K_{mag} y_{mag}.
\]

The CNN log-variance remains an input feature to the GRU. No test-set uncertainty statistic is used to choose the reference.

## Run setup

- 250 synthetic corridor trajectories for fusion training
- 60 independent synthetic corridor trajectories for testing
- 160 temporal bins
- 150 KalmanNet training epochs
- same seeds/protocol as `cnn_dual_kalmannet_full_run.md`

Training-only reference uncertainties:

| Regime | Median training log-variance | Reference sigma |
|---|---:|---:|
| Full Wi-Fi | 3.6264 | 6.130 m |
| Degraded Wi-Fi | 3.6304 | 6.142 m |

## Final 60-walk test results

| Wi-Fi regime | Model | Mean error | Median | P90 | 95% CI half-width |
|---|---|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | Wi-Fi-only KalmanNet | 0.473 m | 0.449 m | 0.697 m | 0.035 m |
| Full Wi-Fi (1 Hz) | Unweighted CNN Dual | 0.506 m | 0.440 m | 0.769 m | 0.056 m |
| Full Wi-Fi (1 Hz) | **Relative-variance CNN Dual** | **0.494 m** | **0.437 m** | **0.764 m** | **0.046 m** |
| Degraded Wi-Fi (5 s, 40% AP dropout) | Wi-Fi-only KalmanNet | 1.533 m | 1.392 m | 2.643 m | 0.193 m |
| Degraded Wi-Fi (5 s, 40% AP dropout) | Unweighted CNN Dual | 1.171 m | 1.042 m | 2.064 m | 0.139 m |
| Degraded Wi-Fi (5 s, 40% AP dropout) | **Relative-variance CNN Dual** | **1.154 m** | **1.113 m** | **1.612 m** | **0.129 m** |

## Comparison

Relative weighting versus the unweighted CNN-output model:

- Full Wi-Fi mean error: **0.506 -> 0.494 m** (2.34% lower).
- Degraded Wi-Fi mean error: **1.171 -> 1.154 m** (1.41% lower).
- Degraded Wi-Fi P90: **2.064 -> 1.612 m** (21.9% lower).
- Improvement over Wi-Fi-only in degraded Wi-Fi increases from **23.6% to 24.7%**.
- Under full Wi-Fi the mean is still about **4.5% worse than Wi-Fi-only**, although the weighted model's median (0.437 m) is lower than Wi-Fi-only (0.449 m).

## Training behavior

Relative weighting greatly stabilizes the full-Wi-Fi run:

- unweighted CNN Dual epoch-1 training MSE: about `3.22e6`
- weighted CNN Dual epoch-1 training MSE: `0.4376`

The degraded run is also much better initially:

- unweighted epoch-1 MSE: about `6.73e3`
- weighted epoch-1 MSE: `4.9449`

However, degraded training still has isolated transient spikes (notably around epochs 8 and 20-21), so relative variance weighting improves but does not completely solve gain instability.

The exact 150-epoch histories are stored in `cnn_dual_kalmannet_relative_variance_training_history.csv`.

## Status

This is the best CNN-output fusion variant tested so far. It preserves the strong degraded-Wi-Fi benefit, improves tail error substantially, reduces the full-Wi-Fi penalty, and uses only training-set uncertainty statistics for the relative reference. The full-Wi-Fi mean is still slightly worse than the Wi-Fi-only baseline, so this result should be reported accurately rather than presented as uniformly superior.
