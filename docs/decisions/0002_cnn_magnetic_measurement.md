# ADR 0002: Replace anomaly-gradient fusion with CNN magnetic fixes

- **Status:** Accepted; implementation pending
- **Date:** 2026-08-09

## Context

The manuscript describes a magnetic sequence CNN that outputs a two-dimensional position estimate and heteroscedastic uncertainty. The evaluated DualKalmanNet instead uses a scalar magnetic anomaly residual multiplied by a pre-surveyed field gradient. This creates a material mismatch between the proposed architecture, equations, and code.

## Decision

The final DualKalmanNet will consume the magnetic CNN's causal position estimate directly. The anomaly map, scalar anomaly innovation, and gradient projection will be removed from the active architecture after a controlled baseline comparison.

The intended innovations are:

```text
y_wifi = z_wifi - x_pred
y_mag  = z_mag  - x_pred
```

The recurrent fusion model will receive both measurement innovations, PDR motion, temporal measurement changes, uncertainty features, previous update information, and explicit availability masks. It will output independent 2x2 gains for Wi-Fi and magnetic corrections.

## Validation requirements

- Train and evaluate on the same synthetic trajectories and sensor regimes as the Wi-Fi-only baseline.
- Preserve strict causality of the magnetic window.
- Evaluate Wi-Fi only, magnetic only, Wi-Fi + IMU, magnetic + IMU, and full fusion where possible.
- Include full and degraded Wi-Fi regimes.
- Record seeds, checkpoints, configurations, confidence intervals, and result files.
- Update the paper only after the new implementation and figures are verified.

## Transitional state

`AnomalyDualKalmanNet` remains in `src/sura/fusion/dual_kalmannet_anomaly.py` solely as a legacy reproducibility baseline.
