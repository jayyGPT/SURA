# Local data layout

The full MagWi dataset and generated model-ready data are intentionally excluded from Git.

Create the standard directories with:

```bash
python -m sura data init
# or
make data-init
```

```text
data/
├── raw/          untouched downloaded datasets
├── interim/      decoded, converted, or temporary intermediate files
├── processed/    fingerprint databases and model-ready arrays
├── local/        personal symlinks, notes, or machine-specific helpers
└── sample/       tiny tracked fixture used only by tests and CI
```

The directory markers are tracked, but everything users place inside `raw`, `interim`,
`processed`, and `local` is ignored.

## Existing legacy `Datasets/` directory

Preview migration from the former layout:

```bash
python -m sura data migrate-legacy
```

Apply it:

```bash
python -m sura data migrate-legacy --apply
```

Targets that already exist are skipped rather than overwritten.

## Raw MagWi placement

Keep the original directory names:

```text
data/raw/magwi/
├── Magnetic field dataset/
│   ├── Static Data/
│   │   └── IT Engineering/
│   └── Continuous Data/
└── WiFi dataset/
    └── IT Engineering/
```

The fingerprint builder currently uses static magnetic data and matched Wi-Fi scans. The
continuous directory may still be stored here for later fusion and real-walk work.

## Build the processed database

Inspect paths and file counts:

```bash
python -m sura data build-fingerprint --dry-run
```

Build:

```bash
python -m sura data build-fingerprint
```

Expected output:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
├── bssid_vocab.json
└── coverage.png
```

Validate it:

```bash
python -m sura data check
```

Generate a compact report:

```bash
python -m sura data analyze
```

## External data root

To avoid storing the dataset in the Git checkout:

```bash
export SURA_DATA_ROOT=/absolute/path/to/sura-data
```

PowerShell:

```powershell
$env:SURA_DATA_ROOT="D:\research\sura-data"
```

The external directory must use the same `raw/interim/processed/local` structure. A per-command
override is also available:

```bash
python -m sura data check --data-root /absolute/path/to/sura-data
```

Never commit participant data, raw sensor recordings, generated arrays, or checkpoints.
