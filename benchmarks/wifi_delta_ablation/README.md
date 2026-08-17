# Wi-Fi temporal-delta ablation

Only `delta_z_wifi = z_wifi,t - z_wifi,previous` is removed. Negative paired differences favor the no-delta model.

| Regime | Model | With delta mean | No delta mean | Paired mean diff (no-with) | 95% CI |
|---|---|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | Wi-Fi-only | 0.4728 m | 0.4678 m | -0.0051 m | [-0.0366, +0.0265] m |
| Full Wi-Fi (1 Hz) | CNN Dual + rel. variance | 0.4664 m | 0.4945 m | +0.0280 m | [+0.0016, +0.0545] m |
| Degraded Wi-Fi (5 s, 40% AP drop) | Wi-Fi-only | 1.5708 m | 1.5480 m | -0.0228 m | [-0.2622, +0.2166] m |
| Degraded Wi-Fi (5 s, 40% AP drop) | CNN Dual + rel. variance | 1.1107 m | 1.2025 m | +0.0918 m | [-0.0224, +0.2060] m |

Interpretation should be based on the paired differences above; this file does not automatically choose the final architecture.
