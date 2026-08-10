# SURA Indoor Localization

Research code, dataset, experiments, and the IEEE manuscript for indoor localization using
smartphone Wi-Fi, magnetic, and inertial measurements.

The project was developed under IIT Delhi's Summer Undergraduate Research Award by Jayendra
Vijay Birhade and Utkarsh Agrawal under the supervision of Dr. Neel Kanth Kundu.

## Repository map

```text
SURA/
├── data/raw/magwi/          complete tracked MagWi dataset
├── data/processed/          generated fingerprint databases (ignored)
├── scripts/data/            dataset preparation and inspection scripts
├── scripts/train/           model training scripts
├── src/sura/                reusable model and algorithm implementations
├── experiments/runs/        generated checkpoints and metrics (ignored)
├── paper/                   canonical LaTeX manuscript
├── docs/                    architecture and project documentation
├── references/              literature library
└── archive/                 historical code and outputs
```

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The runnable scripts add `src/` to Python's import path themselves, so an editable package
installation and a custom command-line tool are not required.

## Dataset

The uploaded dataset is stored at:

```text
data/raw/magwi/
├── Magnetic field dataset/
└── WiFi dataset/
```

Count and inspect the uploaded files:

```bash
python scripts/data/count_dataset.py
```

Build the processed IT Engineering fingerprint database:

```bash
python scripts/data/build_fingerprint_db.py
```

Validate the processed database:

```bash
python scripts/data/check_fingerprint_db.py
```

The generated files appear under:

```text
data/processed/fingerprint_db/it_engineering/
```

## Training

Run either script from the repository root:

```bash
python scripts/train/train_wifi_heatmap.py
python scripts/train/train_magnetic_sequence.py
```

Or enter the training directory first:

```bash
cd scripts/train
python train_wifi_heatmap.py
python train_magnetic_sequence.py
```

Useful examples:

```bash
python scripts/train/train_wifi_heatmap.py --dry-run
python scripts/train/train_wifi_heatmap.py --epochs 5 --split random
python scripts/train/train_magnetic_sequence.py --dry-run
python scripts/train/train_magnetic_sequence.py --epochs 5
python scripts/train/train_magnetic_sequence.py --sweep
```

Checkpoints, predictions, metrics, and histories are written to `experiments/runs/`.
The current scripts train the standalone Wi-Fi heatmap model and magnetic sequence CNN.
CNN-output DualKalmanNet fusion is the next implementation milestone.

See [`RUNNING.md`](RUNNING.md), [`scripts/data/README.md`](scripts/data/README.md), and
[`scripts/train/README.md`](scripts/train/README.md) for the complete but intentionally small
set of instructions.

## Tests and paper

```bash
python -m pytest
python -m ruff check src scripts tests
```

Compile the manuscript:

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

The paper source of truth is `paper/main.tex`. Historical files under `archive/` are retained
for provenance and must not be treated as the active implementation.
