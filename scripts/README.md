# Executable workflows

Scripts are organized by purpose and should import reusable components from `sura` rather than defining duplicate model classes.

- `data/`: dataset conversion and fingerprint-database construction.
- `train/`: model training entry points.
- `evaluate/`: held-out evaluation and ablation studies.
- `figures/`: publication-figure generation from saved result files.

The legacy one-file reproduction scripts remain under `archive/model_snapshots/` and `archive/legacy_experiments/`. They are preserved for provenance but are not the active development interface.

During the next milestone, the CNN-output DualKalmanNet training and evaluation entry points will be added here alongside an explicitly named anomaly baseline.
