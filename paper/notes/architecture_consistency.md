# Architecture consistency checkpoint

## Current mismatch

The magnetic sequence model outputs:

```text
z_mag in R^2
log variance_mag in R
```

The legacy DualKalmanNet evaluated in the present manuscript instead consumes:

```text
scalar anomaly residual = A_obs - A(x_pred)
spatial anomaly gradient = grad A(x_pred)
```

Therefore the current figure and magnetic-CNN description do not accurately describe the evaluated fusion pathway.

## Accepted resolution

Implement and evaluate a CNN-output DualKalmanNet with:

```text
y_wifi = z_wifi - x_pred
y_mag  = z_mag  - x_pred
```

The fusion feature vector should include measurement uncertainty or confidence from both spatial models, sensor availability masks, motion, measurement deltas, and previous state-update context. Independent learned 2x2 gains will produce Wi-Fi and magnetic corrections.

## Paper editing rule

Do not update the abstract, architecture equations, result table, or conclusion to the new design until the code has been trained and evaluated. Once verified, remove the anomaly map and gradient equations entirely rather than retaining both incompatible descriptions.
