# Training workflows

Available canonical commands:

```bash
python -m sura train wifi
python -m sura train magnetic
python -m sura train magnetic --sweep
python -m sura train all
```

Each command:

1. loads checked-in YAML from `configs/`;
2. resolves data through `--data-root`, `SURA_DATA_ROOT`, or `data/`;
3. validates the fingerprint database before training;
4. saves checkpoints, predictions, histories, metrics, seeds, and the Git commit below the
   ignored `experiments/runs/` directory; and
5. supports `--dry-run` for path/schema/model preflight.

The CNN-output DualKalmanNet training command will be added with that architecture. The legacy
anomaly-gradient implementation is not exposed as the default fusion workflow.
