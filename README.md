# SURA Indoor Localization

Research code, dataset, experiments, and the IEEE manuscript for indoor localization using
smartphone Wi-Fi, magnetic, and inertial measurements.

The project was developed under IIT Delhi's Summer Undergraduate Research Award by Jayendra
Vijay Birhade and Utkarsh Agrawal under the supervision of Dr. Neel Kanth Kundu.

## Repository map

```text
SURA/
├── data/raw/magwi/          uploaded original magnetic and Wi-Fi source trees
├── data/processed/          prebuilt and regenerated fingerprint databases
├── scripts/data/            dataset inspection and preparation scripts
├── scripts/train/           direct model training scripts
├── src/sura/                reusable model and algorithm implementations
├── experiments/runs/        generated checkpoints and metrics (ignored)
├── paper/                   canonical LaTeX manuscript
├── docs/                    architecture and project documentation
├── references/              literature library
└── archive/                 historical code and generated dataset artifacts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The runnable scripts add `src/` to Python's import path themselves. There is no custom `sura`
command and no editable package installation is required.

## Verified uploaded data

The cleaned raw source folder contains:

- **4,262 magnetic files**: 4,135 static and 127 continuous;
- **2,831 Wi-Fi files**;
- **7,093 raw source files in total**.

An earlier exhaustive local audit recorded 4,399 Wi-Fi files, so the GitHub upload appears to
be missing **1,568 Wi-Fi files**. The magnetic upload is effectively complete, with one extra
file relative to the older 4,261-file count. Run the exact inventory at any time:

```bash
python scripts/data/count_dataset.py
```

The uploaded prebuilt IT Engineering fingerprint database has been placed at
`data/processed/fingerprint_db/it_engineering/`, so the standalone training scripts can run
immediately. Older fused CSVs, metrics, merged datasets, and plots were moved to
`archive/dataset_generated/` and are not treated as raw inputs.

## Training

From the repository root:

```bash
python scripts/train/train_wifi_heatmap.py
python scripts/train/train_magnetic_sequence.py
```

Or:

```bash
cd scripts/train
python train_wifi_heatmap.py
python train_magnetic_sequence.py
```

Useful checks and short runs:

```bash
python scripts/train/train_wifi_heatmap.py --dry-run
python scripts/train/train_wifi_heatmap.py --epochs 5 --split random
python scripts/train/train_magnetic_sequence.py --dry-run
python scripts/train/train_magnetic_sequence.py --epochs 5
python scripts/train/train_magnetic_sequence.py --sweep
```

Outputs are written to `experiments/runs/`. The current scripts train the standalone Wi-Fi
heatmap model and magnetic sequence CNN. CNN-output DualKalmanNet fusion is the next milestone.

## Data scripts

```bash
python scripts/data/count_dataset.py
python scripts/data/check_fingerprint_db.py
python scripts/data/analyze_dataset.py
```

Regenerate the fingerprint database from the uploaded raw subset only when needed:

```bash
python scripts/data/build_fingerprint_db.py --dry-run
python scripts/data/build_fingerprint_db.py
```

Because the uploaded Wi-Fi source tree is incomplete relative to the earlier audit, a rebuilt
database may differ from the included prebuilt database.

See [`RUNNING.md`](RUNNING.md), [`scripts/data/README.md`](scripts/data/README.md), and
[`scripts/train/README.md`](scripts/train/README.md) for the complete instructions.

## Tests and paper

```bash
python -m pip install pytest ruff
python -m pytest
python -m ruff check src scripts tests
```

Compile the manuscript:

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

The paper source of truth is `paper/main.tex`. Files under `archive/` are provenance only.
