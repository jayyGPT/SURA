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

The prior exhaustive local audit recorded 4,399 Wi-Fi files. Therefore the current GitHub
snapshot appears to be missing 1,568 Wi-Fi files, although the magnetic side is essentially
complete. Use `python scripts/data/count_dataset.py` for the full per-building inventory.

Do not rename folders inside the source trees. The stable root is `data/raw/magwi/`.

## Prebuilt processed database

The uploaded processed IT Engineering database was separated from the raw data and is tracked
at:

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

A rebuild uses the currently uploaded raw subset and may differ from the prebuilt database due
to missing Wi-Fi source files. Use `--output` to preserve the included database while testing a
rebuild.

## Historical generated files

Files that were uploaded beside the source data but were produced by older experiments were
moved to `archive/dataset_generated/`. These include fused CSVs, simulated scans, merged data,
metrics, and figures. They are not raw model inputs.

`data/interim/` and `data/local/` remain ignored scratch areas.
