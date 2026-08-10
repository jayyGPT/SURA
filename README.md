# SURA Indoor Localization

Research code, dataset, benchmarks, and the IEEE paper for smartphone indoor localization using
Wi-Fi, magnetic, and inertial measurements.

The project was developed under IIT Delhi's Summer Undergraduate Research Award by Jayendra
Vijay Birhade and Utkarsh Agrawal under the supervision of Dr. Neel Kanth Kundu.

## Repository

```text
SURA/
├── data/           tracked MagWi dataset + processed fingerprint database
├── models/         model definitions only
├── train/          scripts you actually run to train models
├── tools/          small dataset/preprocessing utilities
├── benchmarks/     benchmark summary, baselines, and generated runs
├── paper/          current IEEE LaTeX paper
├── references/     research papers
├── docs/           useful architecture/data notes and project history
└── archive/        old code and historical artifacts
```

There is no custom Python package, CLI framework, config directory, CI workflow, or test
framework. Hyperparameters are written clearly near the top of each training script.

## Setup

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The tracked raw dataset is already included at:

```text
data/raw/magwi/
├── Magnetic field dataset/
└── WiFi dataset/
```

Verified source counts:

- 4,135 static magnetic files
- 127 continuous magnetic files
- 2,831 Wi-Fi files
- 7,093 source files total

These counts match the original directory manifest that was uploaded.

The processed IT Engineering fingerprint database is included at:

```text
data/processed/fingerprint_db/it_engineering/
```

Quick checks:

```bash
python tools/count_dataset.py
python tools/check_fingerprint_db.py
```

Rebuild the processed fingerprint database only when needed:

```bash
python tools/build_fingerprint_db.py --dry-run
python tools/build_fingerprint_db.py
```

## Training

The normal workflow is simply:

```bash
cd train
python train_wifi_heatmap.py
python train_magnetic_sequence.py
```

Or from the repository root:

```bash
python train/train_wifi_heatmap.py
python train/train_magnetic_sequence.py
```

Before a long run, use the built-in dry run:

```bash
python train/train_wifi_heatmap.py --dry-run
python train/train_magnetic_sequence.py --dry-run
```

Useful options:

```bash
python train/train_wifi_heatmap.py --epochs 5 --split random
python train/train_wifi_heatmap.py --device cuda

python train/train_magnetic_sequence.py --epochs 5
python train/train_magnetic_sequence.py --sweep
python train/train_magnetic_sequence.py --device cuda
```

To change model/training settings, edit the `Configuration` block near the top of the relevant
training script. No separate YAML config is required.

Each training run writes its checkpoint, metrics, predictions, CDF, and training curve under:

```text
benchmarks/runs/
```

The generated run directory is ignored by Git; benchmark numbers that we decide to keep are
recorded in `benchmarks/results.yaml`.

## Dataset tools

```bash
python tools/count_dataset.py
python tools/check_fingerprint_db.py
python tools/analyze_dataset.py
python tools/build_fingerprint_db.py
```

## Benchmarks

See [`benchmarks/README.md`](benchmarks/README.md) for the current benchmark table and the
status of each result. Older KNN baselines are kept under `benchmarks/knn/`.

## Paper

The current paper source is:

```text
paper/main.tex
```

Compile locally with:

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

The next research step is to replace the legacy magnetic-anomaly DualKalmanNet with fusion that
uses the magnetic CNN's 2-D position estimate and uncertainty directly.
