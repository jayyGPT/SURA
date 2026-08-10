# Data

The dataset is tracked directly in this repository.

```text
data/
├── raw/
│   └── magwi/
│       ├── Magnetic field dataset/
│       │   ├── Static Data/
│       │   └── Continuous Data/
│       └── WiFi dataset/
└── processed/
    └── fingerprint_db/
        └── it_engineering/
            ├── nodes.csv
            ├── bssid_vocab.json
            └── coverage.png
```

Verified raw-source counts:

| Source | Files |
|---|---:|
| Magnetic static | 4,135 |
| Magnetic continuous | 127 |
| Wi-Fi | 2,831 |
| **Total** | **7,093** |

These counts match the original uploaded directory manifest.

Do not rename the folders inside `data/raw/magwi/`; the preprocessing scripts use the original
MagWi directory names.

Useful commands from the repository root:

```bash
python tools/count_dataset.py
python tools/check_fingerprint_db.py
python tools/analyze_dataset.py
python tools/build_fingerprint_db.py --dry-run
python tools/build_fingerprint_db.py
```

The processed fingerprint database is also tracked, so normal model training does **not** require
rebuilding it first.
