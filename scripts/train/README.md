# Training workflows

Training entry points must:

1. load a checked-in configuration from `configs/`;
2. resolve data through `SURA_DATA_ROOT` or `data/`;
3. save checkpoints and raw logs below ignored `experiments/runs/`;
4. save a compact metrics/configuration summary for validated runs; and
5. record the Git commit and random seed.

The next implementation task will add separate commands for the Wi-Fi model, magnetic CNN, Wi-Fi-only KalmanNet, anomaly baseline, and CNN-output DualKalmanNet.
