# Runnable scripts

There is no project-specific command-line application. Use the ordinary Python scripts in the
subdirectories:

```text
scripts/data/     inspect raw data and build processed fingerprints
scripts/train/    train the standalone models
```

Examples:

```bash
python scripts/data/count_dataset.py
python scripts/data/build_fingerprint_db.py
python scripts/train/train_wifi_heatmap.py
python scripts/train/train_magnetic_sequence.py
```

Each script supports `--help`. They can be run from the repository root or from their own
folder and automatically locate the repository's `src/`, `data/`, `configs/`, and
`experiments/` directories.
