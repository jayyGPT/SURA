# PDR in the current fusion experiment

**Active fusion source:** `train/kalmannet_wifiheatmap_magneticCNN_pdr.py`  
**Reusable detector/helper source:** `models/pdr.py`  
**Paper reference:** Section II-D, Causal Pedestrian Dead Reckoning (PDR)

## What the estimator actually uses

The PDR prediction in the current fusion experiment is intentionally simple and causal. Acceleration magnitude is passed through an EMA-based step detector:

```text
a_bar[t] = 0.98 a_bar[t-1] + 0.02 a[t]
a_hp[t]  = a[t] - a_bar[t]
```

A step is declared when the high-pass residual exceeds `0.6 m/s^2` and at least `0.3 s` has elapsed since the previous detected step.

For each detected step, the displacement control is

```text
u_t = L_s [cos(theta_hat_t), sin(theta_hat_t)]
```

with the fixed nominal stride length

```text
L_s = 0.65 m
```

No ground-truth training trajectory is used to adapt this stride length in the active benchmark.

## Heading used in the current benchmark

The final fusion benchmark does **not** use the MagWi `Orn_z` column and does **not** call `fit_heading_offset()`.

The synthetic trajectory generator knows the latent path and first computes its geometric tangent `theta_true`. It then generates the heading observation supplied to PDR as

```text
theta_hat = theta_true + random_walk_drift + white_noise
```

where the white-noise standard deviation is `8.8 deg` and the random-walk increment standard deviation is `0.5 deg / sqrt(16.7)` per frame.

This is a simulator/estimator distinction:

- the **simulator** uses the latent ground truth to create a noisy sensor observation;
- the **PDR/KalmanNet estimator** receives only that noisy heading observation and never receives `theta_true`.

A real deployment would require a causal device-heading estimator to supply `theta_hat`. Raw-sensor heading estimation is outside the present fusion evaluation.

## About `models/pdr.py`

`models/pdr.py` still contains generic helpers `fit_heading_offset()` and `calibrate_step_length()`. They are reusable utilities and historical calibration options, but they are **not called by the current CNN-output DualKalmanNet experiment**. Do not cite those helpers as the evaluation protocol.

Using calibration parameters estimated only from a training set would not itself constitute test leakage. The reason the paper no longer describes those calibrations is simply that the reported fusion experiment did not use them.

## Evaluation scope

The PDR is evaluated inside the survey-derived synthetic trajectory protocol:

- 250 generated fusion-training trajectories, seed `1`;
- 60 independently generated fusion-test trajectories, seed `2`;
- exact binned target trajectories are hashed and the run aborts if a train/test duplicate exists;
- the latest verified run found zero exact train/test trajectory overlaps.

The fusion benchmark therefore evaluates unseen **trajectories within a fixed surveyed environment**. It is not a held-out-smartphone experiment. Device-held-out S9+ results are reported separately for experiments that actually use that split, such as the standalone Wi-Fi heatmap phone split and the static KNN baseline.
