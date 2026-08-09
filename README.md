# SURA Indoor Localization

Research code, experiments, documentation, and the IEEE manuscript for a strictly causal
indoor-localization system using smartphone Wi-Fi, magnetic, and inertial measurements.

The project was developed under the Summer Undergraduate Research Award at IIT Delhi by
Jayendra Vijay Birhade and Utkarsh Agrawal under the supervision of Dr. Neel Kanth Kundu.

## Research objective

The system separates localization into three interpretable components:

1. **Spatial measurement models** learn the environment from surveyed Wi-Fi and magnetic
   fingerprints.
2. **Pedestrian dead reckoning (PDR)** converts IMU streams into causal relative motion.
3. **Neural Kalman fusion** learns context-dependent gains that combine spatial fixes with
   motion while handling unavailable sensors.

The current canonical code contains a Wi-Fi heatmap MLP, a magnetic sequence CNN, PDR, and
Wi-Fi-only KalmanNet. The old anomaly-gradient DualKalmanNet is retained only as a named
reproduction baseline. The next research milestone will fuse the magnetic CNN's 2D position
and uncertainty outputs directly into DualKalmanNet.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

python -m sura data init
python -m sura doctor
```

If this checkout already has the old ignored `Datasets/` folder, migrate it safely:

```bash
python -m sura data migrate-legacy          # preview only
python -m sura data migrate-legacy --apply  # perform move
```

After placing the MagWi dataset under `data/raw/magwi/`:

```bash
python -m sura data build-fingerprint
python -m sura data check
python -m sura data analyze
python -m sura train wifi
python -m sura train magnetic
# or run both sequentially:
python -m sura train all
```

On systems with `make`, the same workflow is:

```bash
make setup
make data-init
make prepare-data
make train-wifi
make train-magnetic
# or run both sequentially:
make train-all
```

See **[`COMMANDS.md`](COMMANDS.md)** for all commands, options, outputs, Windows examples,
and Python API usage.

## Canonical repository layout

```text
SURA/
├── src/sura/                 reusable active Python implementation
├── scripts/                  notes for command and workflow entry points
├── configs/                  checked-in reproducible experiment configuration
├── data/                     standard local dataset layout
├── experiments/              validated metrics and ignored run artifacts
├── baselines/                maintained comparison methods
├── paper/                    canonical IEEE LaTeX workspace
├── docs/                     architecture, dataset, decisions, and project history
├── references/               literature library and index
├── tests/                    unit and command-workflow tests
└── archive/                  historical, generated, or non-canonical material
```

A detailed old-to-new path map is maintained in [`MIGRATION.md`](MIGRATION.md).

## Standard dataset layout

Local data is intentionally untracked:

```text
data/
├── raw/          original MagWi download
├── interim/      decoded or temporary conversions
├── processed/    fingerprint databases and model-ready data
├── local/        machine-specific helpers or symlinks
└── sample/       tiny tracked fixture used by tests and CI
```

The default raw location is:

```text
data/raw/magwi/
├── Magnetic field dataset/
└── WiFi dataset/
```

The processed fingerprint database is generated at:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
├── bssid_vocab.json
└── coverage.png
```

Keep data elsewhere by setting:

```bash
export SURA_DATA_ROOT=/absolute/path/to/sura-data
```

PowerShell:

```powershell
$env:SURA_DATA_ROOT="D:\research\sura-data"
```

See [`data/README.md`](data/README.md) for exact placement rules.

## One-command data preparation

After the raw MagWi files have been placed correctly:

```bash
make prepare-data
```

This performs:

```text
raw MagWi files
    -> processed fingerprint database
    -> schema and AP-vocabulary validation
    -> local JSON/Markdown dataset report
```

The direct cross-platform equivalents are:

```bash
python -m sura data build-fingerprint
python -m sura data check
python -m sura data analyze
```

## Training

### Wi-Fi heatmap

```bash
python -m sura train wifi
```

This trains both:

- a reproducible random-visit split;
- an S9+ held-out-device split.

Check paths and tensors without training:

```bash
python -m sura train wifi --dry-run
```

### Magnetic sequence CNN

```bash
python -m sura train magnetic
```

Run the configured window sweep:

```bash
python -m sura train magnetic --sweep
```

Check the magnetic maps and corridor graph without training:

```bash
python -m sura train magnetic --dry-run
```

Checkpoints, predictions, histories, configuration snapshots, seeds, metrics, and the Git
commit are written below the ignored directory `experiments/runs/`.

Fusion training will be added only with the approved CNN-output DualKalmanNet so there is no
ambiguous or scientifically obsolete "main fusion" command.

## Configuration

Default files:

```text
configs/datasets/magwi_it_engineering.yaml
configs/wifi_heatmap.yaml
configs/magnetic_sequence.yaml
```

Pass another configuration with `--config` or `--dataset-config`. Do not hard-code dataset
paths inside model or training files.

## Using the code as a Python package

```python
from sura.data import fingerprint_database, load_fingerprint_database
from sura.models import MagSequenceMatcher, WifiHeatmapNet
from sura.motion import StepDetector

database = load_fingerprint_database(fingerprint_database("it_engineering"))
print(database.summary())
```

Reusable models and algorithms live only under `src/sura/`; commands import those modules
rather than duplicating their implementation.

## Validation

```bash
python -m sura doctor
pytest
ruff check src tests
```

Compile the manuscript:

```bash
python -m sura paper build
```

Or run everything:

```bash
make check
```

GitHub Actions validates Python syntax, unit tests, command smoke workflows, and
`paper/main.tex` compilation.

## Source-of-truth rules

- Active reusable code belongs only under `src/sura/`.
- Executable workflows use `sura` / `python -m sura`.
- Paper source belongs only under `paper/`.
- Headline paper values must be recorded under `experiments/results/` before entering the
  manuscript.
- Files under `archive/` must not be imported or cited as current results.
- Raw datasets, checkpoints, caches, reports generated during local analysis, and LaTeX build
  products are not committed.

## Current development sequence

1. Establish reproducible standalone Wi-Fi and magnetic runs.
2. Implement CNN-output magnetic measurement fusion in DualKalmanNet.
3. Compare Wi-Fi-only, legacy anomaly, and CNN-based dual fusion on identical splits.
4. Update equations, figures, tables, abstract, and claims from verified outputs.

## Preservation

The exact repository state before renovation is preserved on:

```text
archive/pre-renovation-2026-08-09
```
