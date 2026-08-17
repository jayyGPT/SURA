# Wi-Fi temporal-delta ablation

Only `delta_z_wifi = z_wifi,t - z_wifi,previous` is removed. Negative paired differences favor the no-delta model.

| Regime | Model | With delta mean | No delta mean | Paired mean diff (no-with) | 95% CI |
|---|---|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | Wi-Fi-only | 0.4728 m | 0.4678 m | -0.0051 m | [-0.0366, +0.0265] m |
| Full Wi-Fi (1 Hz) | CNN Dual + rel. variance | 0.4664 m | 0.4945 m | +0.0280 m | [+0.0016, +0.0545] m |
| Degraded Wi-Fi (5 s, 40% AP drop) | Wi-Fi-only | 1.5708 m | 1.5480 m | -0.0228 m | [-0.2622, +0.2166] m |
| Degraded Wi-Fi (5 s, 40% AP drop) | CNN Dual + rel. variance | 1.1107 m | 1.2025 m | +0.0918 m | [-0.0224, +0.2060] m |

## Decision for the current paper

Retain `delta_z_wifi` in the final CNN DualKalmanNet. In the paired DualKalmanNet comparison, removing the feature increased mean error by `+0.0280 m` in full Wi-Fi, with a paired 95% CI of `[+0.0016, +0.0545] m`. In degraded Wi-Fi it increased mean error by `+0.0918 m`, although that interval `[-0.0224, +0.2060] m` includes zero. The Wi-Fi-only ablations were inconclusive.

The absolute with-delta values in this ablation are not replacements for the headline paper metrics. For this experiment the model-initialization and minibatch-shuffle seeds were reset before every paired training so the architectural comparison differed only in the two `delta_z_wifi` inputs. Interpret the paired differences, not cross-experiment changes in absolute error.
