# Magnetic CNN variance calibration

This benchmark compares the magnetic CNN's predicted `log(sigma^2)` with its actual 2-D position error on the same synthetic 60-walk test protocol used by the CNN-output DualKalmanNet experiment.

## Correlation summary

| Regime | Samples | Mean error | Mean predicted sigma | Spearman uncertainty/error | Pearson logvar/error | Q4/Q1 mean-error ratio |
|---|---:|---:|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | 8458 | 3.363 m | 6.896 m | 0.310 | 0.359 | 2.41x |
| Degraded Wi-Fi (5 s, 40% AP drop) | 8458 | 3.429 m | 6.917 m | 0.312 | 0.335 | 2.40x |

## Uncertainty quartiles

### Full Wi-Fi (1 Hz)

| Quartile | Samples | Predicted sigma | Mean actual error | Median actual error | Radial RMSE |
|---|---:|---:|---:|---:|---:|
| Q1 most confident | 2115 | 5.118 m | 2.258 m | 1.860 m | 2.795 m |
| Q2 | 2114 | 5.978 m | 2.744 m | 2.440 m | 3.310 m |
| Q3 | 2114 | 6.880 m | 2.996 m | 2.037 m | 4.340 m |
| Q4 least confident | 2115 | 9.609 m | 5.452 m | 4.154 m | 7.361 m |

### Degraded Wi-Fi (5 s, 40% AP drop)

| Quartile | Samples | Predicted sigma | Mean actual error | Median actual error | Radial RMSE |
|---|---:|---:|---:|---:|---:|
| Q1 most confident | 2115 | 5.123 m | 2.286 m | 1.954 m | 2.815 m |
| Q2 | 2114 | 5.981 m | 2.855 m | 2.461 m | 3.473 m |
| Q3 | 2114 | 6.875 m | 3.082 m | 2.101 m | 4.776 m |
| Q4 least confident | 2115 | 9.689 m | 5.494 m | 4.306 m | 7.255 m |

## Interpretation

The uncertainty head is **useful for ranking**, but it is **not absolutely calibrated**.

- Spearman correlation between predicted uncertainty and actual error is about **0.31** in both regimes: moderate, but clearly positive.
- The least-confident quartile has roughly **2.4x** the mean error of the most-confident quartile.
- Mean predicted sigma is about **6.9 m**, while the actual mean magnetic position error is about **3.4 m**.
- Mean predicted variance is about **4.4x** the empirical per-axis variance, so the raw variance is strongly conservative/overestimated.
- About **92%** of magnetic errors fall inside one predicted sigma; for a perfectly calibrated isotropic 2-D Gaussian this would be much lower, confirming the overestimation.

This means we should not directly use `1/sigma^2` as an absolute Kalman covariance without calibration. A better first experiment is to use the variance as a **relative confidence signal** (for example, normalized against the median training uncertainty) so low-confidence magnetic fixes are down-weighted while high-confidence fixes remain useful.

The reproducible analysis is `benchmarks/analyze_magnetic_variance.py`. Running it also generates a decile calibration plot and per-sample CSV locally.
