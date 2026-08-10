# Dataset layout

## Raw uploaded source files

```text
data/raw/magwi/
├── Magnetic field dataset/
│   ├── Static Data/
│   └── Continuous Data/
└── WiFi dataset/
```

Verified tracked counts:

| Source | Files |
|---|---:|
| Magnetic static | 4,135 |
| Magnetic continuous | 127 |
| Magnetic total | 4,262 |
| Wi-Fi | 2,831 |
| Raw source total | 7,093 |

These counts exactly match the source portion of the original uploaded directory manifest. In addition, the Git tree IDs for both `Magnetic field dataset/` and `WiFi dataset/` are unchanged from the pre-reorganization commit, which verifies the complete directory names and file contents—not only the counts.

The older project audit mentioning 4,399 Wi-Fi files belongs to a different historical audit/snapshot and should not be interpreted as missing files from this upload.

Do not rename folders inside the source trees. The stable root is `data/raw/magwi/`.

## Prebuilt processed database

The uploaded processed IT Engineering database was separated from the raw data and is tracked at:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
├── bssid_vocab.json
└── coverage.png
```

Validate it:

```bash
python scripts/data/check_fingerprint_db.py
```

This prebuilt database is the default input to both current training scripts.

## Rebuilding

```bash
python scripts/data/build_fingerprint_db.py --dry-run
python scripts/data/build_fingerprint_db.py
```

Use `--output` to preserve the included database while testing a rebuilt version.

## Historical generated files

The original uploaded folder also contained 62 derived/generated artifacts beside the two source trees. During cleanup these were moved to `archive/dataset_generated/`. They include fused CSVs, simulated scans, merged data, metrics, figures, and the old processed fingerprint database. They remain available for provenance but are not raw model inputs.

`data/interim/` and `data/local/` remain ignored scratch areas.
