# CNN-output DualKalmanNet — full run

This file records the first full experiment in which the magnetic 1D-CNN output is used directly by DualKalmanNet.

## Architecture tested

The legacy scalar magnetic-anomaly/gradient correction was removed. The magnetic branch is now:

```text
84-frame magnetic window
        -> magnetic 1D-CNN
        -> 2-D position fix z_mag + predicted log-variance
        -> magnetic innovation (z_mag - x_pred)
        -> learned 2x2 magnetic gain
```

The GRU also receives the CNN log-variance as a confidence feature.

## Run setup

- 250 synthetic corridor trajectories for fusion training
- 60 independent synthetic corridor trajectories for testing
- 160 temporal bins per trajectory
- 150 KalmanNet training epochs
- Wi-Fi-only KalmanNet trained under the same protocol as the comparison baseline
- Magnetic CNN checkpoint: 84-frame model

## Final test results

| Wi-Fi regime | Model | Mean error | Median | P90 | 95% CI half-width |
|---|---|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | Wi-Fi-only KalmanNet | **0.473 m** | 0.449 m | 0.697 m | 0.035 m |
| Full Wi-Fi (1 Hz) | CNN-output DualKalmanNet | 0.506 m | **0.440 m** | 0.769 m | 0.056 m |
| Degraded Wi-Fi (5 s, 40% AP dropout) | Wi-Fi-only KalmanNet | 1.533 m | 1.392 m | 2.643 m | 0.193 m |
| Degraded Wi-Fi (5 s, 40% AP dropout) | **CNN-output DualKalmanNet** | **1.171 m** | **1.042 m** | **2.064 m** | **0.139 m** |

Relative change from adding the magnetic CNN:

- Full Wi-Fi: **-7.0%** mean-error improvement (slightly worse mean; median improves slightly).
- Degraded Wi-Fi: **+23.6%** mean-error improvement.

## Magnetic measurement quality inside the fusion test walks

| Regime | Availability | Mean standalone magnetic-fix error | Median magnetic-fix error |
|---|---:|---:|---:|
| Full Wi-Fi | 88.1% | 3.36 m | 2.47 m |
| Degraded Wi-Fi | 88.1% | 3.43 m | 2.52 m |

The first part of each trajectory has no magnetic fix until an 84-frame CNN window is available.

## Training behavior

The CNN-output model converged to useful test performance, but the unrestricted learned 2x2 gain matrices caused very large transient MSE spikes during early training, especially in the degraded-Wi-Fi regime. The exact 150-epoch training history is stored in `cnn_dual_kalmannet_training_history.csv`.

These values are an **experiment record**, not yet the final paper numbers. Before publication we still need to evaluate whether the CNN-predicted variance can reliably identify weak magnetic fixes and use that information to stabilize the magnetic correction.
