# Repository renovation map

This document records how the pre-renovation repository was reorganized. The preserved state is available at `archive/pre-renovation-2026-08-09`.

## Top-level migration

| Previous path | Canonical destination | Status |
|---|---|---|
| `Models (Read Only)/` | `archive/model_snapshots/` plus selected active code in `src/sura/` | Historical snapshots retained |
| `Testing_and_Benchmarking_Sandbox/` | `archive/legacy_experiments/` plus maintained workflows in `scripts/` | Historical experiments retained |
| `Publication/` | `archive/legacy_publication/` plus current manuscript in `paper/` | Historical drafts retained |
| `Presentations and Progress/` | `docs/project_history/` | Project record retained |
| `Research Papers (Previous)/` | `references/literature/` | Literature retained |
| `_temp/` | `archive/scratch/` | Non-canonical scratch retained |
| legacy KNN directory | `baselines/knn/` | Maintained as a comparison baseline |

## Canonical model mapping

| Model | Canonical implementation | Supporting documentation |
|---|---|---|
| Wi-Fi probability heatmap | `src/sura/models/wifi_heatmap.py` | `docs/architecture/wifi_heatmap.md` |
| Magnetic sequence CNN | `src/sura/models/magnetic_sequence_cnn.py` | `docs/architecture/magnetic_sequence_cnn.md` |
| PDR | `src/sura/motion/pdr.py` | `docs/architecture/pdr.md` |
| Wi-Fi-only KalmanNet | `src/sura/fusion/wifi_kalmannet.py` | `docs/architecture/kalman_fusion.md` |
| Legacy anomaly DualKalmanNet | `src/sura/fusion/dual_kalmannet_anomaly.py` | `docs/decisions/0002_cnn_magnetic_measurement.md` |

## Important interpretation

A path under `archive/` may be useful for provenance, but it is not an active source of truth. New development should not import code from archived paths.
