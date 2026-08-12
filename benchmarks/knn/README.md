# KNN baselines

The current reproducible Wi-Fi + magnetic KNN baseline is `wifi_mag_knn.py`. Older proof-of-concept scripts remain under `src/` and their old plots remain under `legacy_results/`; those historical numbers are not used in the paper.

## Current protocols

### 1. Real static held-out-device fingerprinting

Uses `data/processed/fingerprint_db/it_engineering/` directly.

- train phones: A8, G7, S8
- held-out final test phone: S9+
- train/test visits: 726 / 90
- Wi-Fi: canonical 250-AP RSSI encoding
- magnetic: rotation-invariant `magN`, `magV`, `magH`, and `dip` node statistics
- K and Wi-Fi/magnetic block weight: selected using leave-one-phone-out grouped cross-validation on the training phones only

Final S9+ mean errors:

| Variant | K | Mean | Median | P90 |
|---|---:|---:|---:|---:|
| Wi-Fi KNN | 7 | 3.310 m | 1.571 m | 6.352 m |
| Magnetic KNN | 20 | 17.538 m | 10.775 m | 42.698 m |
| Wi-Fi + magnetic KNN | 3 | 7.459 m | 4.359 m | 14.400 m |

The hybrid selects a 0.75 Wi-Fi / 0.25 magnetic distance weighting. Its degradation relative to Wi-Fi-only KNN is evidence that simple static magnetic concatenation does not solve cross-device magnetic heterogeneity.

### 2. Matched synthetic trajectory protocol

Uses the exact 250-training/60-test, 160-bin trajectory generation used by `train/kalmannet_wifiheatmap_magneticCNN_pdr.py`. This KNN is deliberately non-temporal: its features are the current Wi-Fi heatmap fix, magnetic-CNN fix, magnetic log-variance, and availability masks. It receives no PDR motion and no recurrent history. K is selected using GroupKFold with whole trajectories kept together.

| Regime | Selected K | Wi-Fi+Mag KNN | Wi-Fi-only KalmanNet | CNN Dual + relative variance |
|---|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | 5 | 0.802 m | 0.473 m | 0.494 m |
| Degraded Wi-Fi (5 s, 40% AP drop) | 20 | 2.606 m | 1.533 m | 1.154 m |

## Outputs

Reviewed outputs are under `current_results/`:

- `summary.json` and protocol-specific `metrics.json`
- real static predictions CSV
- trajectory prediction/error NPZ files
- large-font CDF plots used by the manuscript

Run both protocols with:

```bash
python benchmarks/knn/wifi_mag_knn.py --protocol both
```

### Attribution note

The MagWi paper is a benchmark-dataset/characterization paper. We did not locate an explicit KNN localization method or KNN result in that paper, so the values above are **our reproducible classical baselines on MagWi data**, not results copied from or attributed to the MagWi authors.
