# Magnetic CNN uncertainty calibration

This benchmark predates the P6/P7 terminology audit and stores the CNN scalar under historical names such as `log_variance` and `sigma`. The active training objective uses one scalar scale on the **summed 2-D radial squared error**, so this value is not interpreted as a calibrated Cartesian variance or covariance. In this document, `sqrt(q)` denotes the positive scale derived from the stored score, where `q = exp(ell_mag)`.

The benchmark compares that learned scale with the actual 2-D position error on the same synthetic 60-walk test protocol used by the CNN-output DualKalmanNet experiment.

## Correlation summary

| Regime | Samples | Mean error | Mean derived sqrt(q) | Spearman uncertainty/error | Pearson score/error | Q4/Q1 mean-error ratio |
|---|---:|---:|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | 8458 | 3.363 m | 6.896 m | 0.310 | 0.359 | 2.41x |
| Degraded Wi-Fi (5 s, 40% AP drop) | 8458 | 3.429 m | 6.917 m | 0.312 | 0.335 | 2.40x |

## Uncertainty quartiles

### Full Wi-Fi (1 Hz)

| Quartile | Samples | Derived sqrt(q) | Mean actual error | Median actual error | Radial RMSE |
|---|---:|---:|---:|---:|---:|
| Q1 most confident | 2115 | 5.118 m | 2.258 m | 1.860 m | 2.795 m |
| Q2 | 2114 | 5.978 m | 2.744 m | 2.440 m | 3.310 m |
| Q3 | 2114 | 6.880 m | 2.996 m | 2.037 m | 4.340 m |
| Q4 least confident | 2115 | 9.609 m | 5.452 m | 4.154 m | 7.361 m |

### Degraded Wi-Fi (5 s, 40% AP drop)

| Quartile | Samples | Derived sqrt(q) | Mean actual error | Median actual error | Radial RMSE |
|---|---:|---:|---:|---:|---:|
| Q1 most confident | 2115 | 5.123 m | 2.286 m | 1.954 m | 2.815 m |
| Q2 | 2114 | 5.981 m | 2.855 m | 2.461 m | 3.473 m |
| Q3 | 2114 | 6.875 m | 3.082 m | 2.101 m | 4.776 m |
| Q4 least confident | 2115 | 9.689 m | 5.494 m | 4.306 m | 7.255 m |

## Interpretation

The uncertainty head is **useful for ranking**, but its raw scale should **not be treated as an absolute covariance**.

- Spearman correlation between the learned uncertainty scale and actual radial error is about **0.31** in both regimes: moderate, but clearly positive.
- The least-confident quartile has roughly **2.4x** the mean error of the most-confident quartile.
- The stored positive scale is conservative relative to the observed radial errors.
- These results support confidence **ordering** far more strongly than a literal probabilistic calibration claim.

This is why the final fusion method does not insert the CNN scalar as a Kalman measurement covariance. It uses the training-normalized relative score

```text
ell_ref = median training ell_mag
w_mag   = 1 / (1 + exp(ell_mag - ell_ref))
```

to suppress comparatively uncertain magnetic corrections.

The numerical values above are unchanged from the original calibration run; only their mathematical interpretation has been tightened. The reproducible analysis is `benchmarks/analyze_magnetic_variance.py`.