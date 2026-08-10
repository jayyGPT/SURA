# Running the project

The project uses ordinary Python scripts rather than a custom command-line application. Run these commands from the repository root unless stated otherwise.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Inspect the tracked dataset

```bash
python scripts/data/count_dataset.py
```

Verified source snapshot:

```text
4,262 magnetic files
2,831 Wi-Fi files
7,093 source files total
```

These counts match the original uploaded directory manifest. The magnetic and Wi-Fi Git subtree hashes also match the pre-reorganization upload, confirming that the source filenames and bytes were preserved exactly.

## 3. Check the prebuilt fingerprint database

```bash
python scripts/data/check_fingerprint_db.py
python scripts/data/analyze_dataset.py
```

The included database is:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
├── bssid_vocab.json
└── coverage.png
```

It lets the standalone models train without rebuilding raw data first.

## 4. Train the Wi-Fi heatmap model

```bash
python scripts/train/train_wifi_heatmap.py
```

Common options:

```bash
python scripts/train/train_wifi_heatmap.py --dry-run
python scripts/train/train_wifi_heatmap.py --split random
python scripts/train/train_wifi_heatmap.py --split phone
python scripts/train/train_wifi_heatmap.py --epochs 5
python scripts/train/train_wifi_heatmap.py --device cuda
```

## 5. Train the magnetic sequence CNN

```bash
python scripts/train/train_magnetic_sequence.py
```

Common options:

```bash
python scripts/train/train_magnetic_sequence.py --dry-run
python scripts/train/train_magnetic_sequence.py --epochs 5
python scripts/train/train_magnetic_sequence.py --sweep
python scripts/train/train_magnetic_sequence.py --device cuda
```

`--sweep` trains the configured 50, 84, 134, and 167-frame windows.

## 6. Rebuild processed fingerprints when needed

```bash
python scripts/data/build_fingerprint_db.py --dry-run
python scripts/data/build_fingerprint_db.py
```

Use `--output` when you want to preserve the included prebuilt database while testing a rebuilt version.

## 7. Outputs

```text
experiments/runs/wifi_heatmap/
experiments/runs/magnetic_sequence/
```

Training outputs are ignored by Git.

## 8. Paper

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

The generated PDF is `paper/build/main.pdf`.
