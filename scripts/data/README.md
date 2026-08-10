# Data scripts

Run from the repository root:

```bash
python scripts/data/count_dataset.py
python scripts/data/build_fingerprint_db.py --dry-run
python scripts/data/build_fingerprint_db.py
python scripts/data/check_fingerprint_db.py
python scripts/data/analyze_dataset.py
```

Or enter this folder and omit `scripts/data/` from the command.

- `count_dataset.py` counts the uploaded raw files and reports per-building totals.
- `build_fingerprint_db.py` builds the IT Engineering static Wi-Fi/magnetic database.
- `check_fingerprint_db.py` validates the generated table and AP vocabulary.
- `analyze_dataset.py` writes a compact JSON and Markdown inventory under
  `experiments/runs/dataset_analysis/`.

All scripts default to the repository's `data/` directory and accept path overrides through
command-line arguments when needed.
