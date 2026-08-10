# Data scripts

From the repository root:

```bash
python scripts/data/count_dataset.py
python scripts/data/check_fingerprint_db.py
python scripts/data/analyze_dataset.py
```

To rebuild processed fingerprints:

```bash
python scripts/data/build_fingerprint_db.py --dry-run
python scripts/data/build_fingerprint_db.py
```

Or enter this folder and omit `scripts/data/` from the commands.

- `count_dataset.py` reports exact uploaded source counts and per-building totals.
- `check_fingerprint_db.py` validates the included prebuilt table and AP vocabulary.
- `analyze_dataset.py` writes compact JSON and Markdown reports under
  `experiments/runs/dataset_analysis/`.
- `build_fingerprint_db.py` rebuilds IT Engineering fingerprints from the uploaded raw subset.

The earlier audit found more Wi-Fi files than the GitHub upload. Preserve the prebuilt database
or pass `--output` when testing a rebuild.
