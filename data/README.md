# Dataset layout

The complete MagWi dataset is tracked in this repository at:

```text
data/raw/magwi/
├── Magnetic field dataset/
│   ├── Static Data/
│   └── Continuous Data/
└── WiFi dataset/
```

Do not rename folders inside the original dataset. Active code refers only to the stable
`data/raw/magwi` root and then uses the dataset's original internal names.

## Verify the upload

```bash
python scripts/data/count_dataset.py
```

This reports the repository file count, raw dataset file count and size, magnetic static and
continuous counts, Wi-Fi count, extensions, and per-building counts.

## Generated data

The original raw files are tracked. Derived data is not tracked:

```text
data/interim/      temporary conversions
data/processed/    fingerprint databases and model-ready data
data/local/        machine-specific scratch data
```

Build the IT Engineering fingerprint database with:

```bash
python scripts/data/build_fingerprint_db.py
```

Output:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
├── bssid_vocab.json
└── coverage.png
```

Validate or summarize it with:

```bash
python scripts/data/check_fingerprint_db.py
python scripts/data/analyze_dataset.py
```

The Wi-Fi source files use a `.csv` filename but many contain BIFF8 Excel data. The builder
handles this through the `xlrd` dependency listed in `requirements.txt`.
