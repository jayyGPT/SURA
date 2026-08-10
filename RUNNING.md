# Running the project

This project intentionally uses ordinary Python scripts rather than a custom `sura` command.
Run the commands below from the repository root unless a section says otherwise.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

For tests and linting, also install:

```bash
python -m pip install pytest ruff
```

## 2. Check the uploaded dataset

```bash
python scripts/data/count_dataset.py
```

The expected raw location is `data/raw/magwi/`. The script prints total file counts, total
size, static/continuous magnetic counts, Wi-Fi counts, and counts per building.

## 3. Build the fingerprint database

Preview the raw inputs without processing:

```bash
python scripts/data/build_fingerprint_db.py --dry-run
```

Build the database:

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

Validate it:

```bash
python scripts/data/check_fingerprint_db.py
```

Create a compact report:

```bash
python scripts/data/analyze_dataset.py
```

## 4. Train the Wi-Fi heatmap model

```bash
python scripts/train/train_wifi_heatmap.py
```

The default run trains both the random-visit split and the S9+ held-out-device split.
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

## 6. Outputs

Training outputs are placed below:

```text
experiments/runs/wifi_heatmap/
experiments/runs/magnetic_sequence/
```

These generated checkpoints and logs are ignored by Git.

## 7. Paper

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

The generated PDF is `paper/build/main.pdf`.
