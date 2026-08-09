# Command reference

The installed console command is `sura`. Every command also works as
`python -m sura`, which is useful before the console script is on `PATH`.

On systems with `make`, the root `Makefile` provides shorter aliases.

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Equivalent:

```bash
make setup
```

Display available commands:

```bash
sura commands
make help
```

## 2. Local dataset directories

Create the standard ignored directory tree:

```bash
sura data init
# or
make data-init
```

This creates:

```text
data/
├── raw/          untouched downloaded data
├── interim/      decoded or temporary conversions
├── processed/    model-ready fingerprint databases and arrays
├── local/        personal notes, symlinks, or machine-specific helpers
└── sample/       tiny tracked CI/test fixture
```

Everything inside `raw`, `interim`, `processed`, and `local` is ignored by Git.

To keep the dataset outside this repository, either pass `--data-root`:

```bash
sura data init --data-root /mnt/research/sura-data
```

or set the environment once:

```bash
export SURA_DATA_ROOT=/mnt/research/sura-data       # bash/zsh
$env:SURA_DATA_ROOT="D:\research\sura-data"         # PowerShell
```

The command-line flag takes precedence over the environment variable.

## 3. Migrate an existing legacy `Datasets/` folder

Older checkouts may already contain an ignored root-level `Datasets/` directory. Preview the
safe mapping first:

```bash
sura data migrate-legacy
make migrate-legacy
```

Apply it by moving known raw and processed directories:

```bash
sura data migrate-legacy --apply --mode move
```

Copy instead of moving:

```bash
sura data migrate-legacy --apply --mode copy
```

The command never overwrites an existing canonical target. It migrates the magnetic dataset,
Wi-Fi dataset, fingerprint database, and any `Continuous_Fused_*.csv` files it recognizes.

## 4. Place the MagWi dataset

The default raw layout is:

```text
data/raw/magwi/
├── Magnetic field dataset/
│   └── Static Data/
│       └── IT Engineering/
└── WiFi dataset/
    └── IT Engineering/
```

Keep the original MagWi directory names. The Wi-Fi files may have a `.csv` suffix while
containing BIFF8 Excel data; the builder handles this through `xlrd`.

## 5. Build the processed fingerprint database

First inspect the resolved paths without parsing files:

```bash
sura data build-fingerprint --dry-run
```

Build `nodes.csv`, `bssid_vocab.json`, and a coverage plot:

```bash
sura data build-fingerprint
# or
make build-fingerprint
```

Output:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
├── bssid_vocab.json
└── coverage.png
```

Use alternate building folder names when necessary:

```bash
sura data build-fingerprint \
  --building "BE Building" \
  --wifi-building "BE Engineering"
```

## 6. Validate and analyze data

Validate the processed database and AP vocabulary:

```bash
sura data check
make data-check
```

Generate a compact raw-file inventory plus fingerprint summary:

```bash
sura data analyze
make data-analyze
```

The generated JSON and Markdown reports are written below the ignored directory:

```text
experiments/runs/dataset_analysis/
```

Run the complete preparation sequence after placing raw data:

```bash
make prepare-data
```

## 7. Train the Wi-Fi heatmap model

Run both the random-visit split and the held-out-device split:

```bash
sura train wifi
make train-wifi
```

Run only one split:

```bash
sura train wifi --split random
sura train wifi --split phone
```

Validate all paths, columns, and tensor dimensions without training:

```bash
sura train wifi --dry-run
```

Override the number of epochs or compute device:

```bash
sura train wifi --epochs 5 --device cpu
sura train wifi --device cuda
```

Use a named run:

```bash
sura train wifi --run-name wifi-paper-rerun-01
```

Outputs are stored under:

```text
experiments/runs/wifi_heatmap/<run-name>/
├── run.json
├── random/
│   ├── model.pt
│   ├── predictions.npz
│   └── history.json
└── phone/
    ├── model.pt
    ├── predictions.npz
    └── history.json
```

The default hyperparameters are in `configs/wifi_heatmap.yaml`.

## 8. Train the standalone magnetic sequence CNN

Train the configured 84-frame model:

```bash
sura train magnetic
make train-magnetic
```

Run the configured window-size sweep:

```bash
sura train magnetic --sweep
make train-magnetic-sweep
```

Preflight the magnetic map, corridor graph, features, and paths without training:

```bash
sura train magnetic --dry-run
```

Outputs are stored under:

```text
experiments/runs/magnetic_sequence/<run-name>/
├── run.json
└── window_<frames>/
    ├── model.pt
    ├── predictions.npz
    └── history.json
```

The model uses the processed static fingerprint database to build gravity-referenced
magnetic maps and generates causal training windows along the surveyed corridor graph.
The default settings are in `configs/magnetic_sequence.yaml`.

## 9. Train all currently canonical standalone models

This is computationally expensive:

```bash
sura train all
make train-all
```

Preflight both stages without training:

```bash
sura train all --dry-run
```

Fusion training is intentionally not exposed yet. The next feature branch will implement the
approved CNN-output DualKalmanNet and add one canonical fusion-training command. The old
anomaly-gradient system remains only as a reproducibility baseline.

## 10. Environment checks

Display package, dataset, and LaTeX readiness:

```bash
sura doctor
make doctor
```

Return a non-zero status unless required packages and the processed fingerprint database are
ready:

```bash
sura doctor --strict
```

## 11. Tests and linting

```bash
pytest
ruff check src tests
```

Aliases:

```bash
make test
make lint
```

## 12. Compile the paper

```bash
sura paper build
make paper
```

Clean first:

```bash
sura paper build --clean
make clean-paper
```

Output:

```text
paper/build/main.pdf
```

## 13. Full repository check

```bash
make check
```

This runs Ruff, tests, and LaTeX compilation. GitHub Actions runs the equivalent checks on
every pull request.

## 14. Python API examples

```python
from sura.data import fingerprint_database, load_fingerprint_database
from sura.models import MagSequenceMatcher, WifiHeatmapNet
from sura.motion import StepDetector

database = load_fingerprint_database(fingerprint_database("it_engineering"))
print(database.summary())

wifi_model = WifiHeatmapNet(
    n_access_points=len(database.access_point_columns),
    n_cells=100,
)

magnetic_model = MagSequenceMatcher(in_channels=4)
step_detector = StepDetector()
```
