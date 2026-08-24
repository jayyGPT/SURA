# SURA Indoor Localization

Research code, data, canonical benchmarks, and the IEEE manuscript for smartphone indoor localization using Wi-Fi, magnetic, and inertial measurements.

The project was developed under IIT Delhi's Summer Undergraduate Research Award by Jayendra Vijay Birhade and Utkarsh Agrawal under the supervision of Dr. Neel Kanth Kundu.

## Active repository layout

```text
SURA/
├── data/           raw MagWi source data + corrected processed fingerprint database
├── models/         active model definitions
├── train/          active training and fusion scripts
├── tools/          dataset/preprocessing utilities
├── checkpoints/   selected Wi-Fi and magnetic checkpoints with SHA-256 provenance
├── benchmarks/     frozen final protocol, KNN baseline, canonical metrics/predictions/CDF
└── paper/          current IEEE LaTeX manuscript, bibliography, and final audit records
```

Historical experiments, temporary audit helpers, local literature copies, and stale project reports are intentionally excluded from the active tree. They remain recoverable from Git history.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\Activate.ps1`.

## Dataset

Raw MagWi data are retained under `data/raw/magwi/`. The paper uses the corrected IT Engineering processed database at:

```text
data/processed/fingerprint_db/it_engineering/
```

The corrected builder keeps each static magnetic visit as one database row and attaches a Wi-Fi scan only when mode, scenario, phone, user, and timestamped filename all match exactly. The magnetic static survey coordinate is the common map coordinate; raw Wi-Fi coordinates are retained separately for audit.

Rebuild/check:

```bash
python tools/count_dataset.py
python tools/build_fingerprint_db.py
python tools/check_fingerprint_db.py
```

Canonical database facts are recorded in `data/processed/fingerprint_db/it_engineering/pairing_audit.json`.

## Selected checkpoints

```text
checkpoints/wifi_heatmap.pt
checkpoints/magnetic_sequence.pt
```

Their SHA-256 hashes are stored in `checkpoints/SHA256SUMS.txt` and the final result manifest.

## Training / development

Standalone model entry points:

```bash
python train/train_wifi_heatmap.py
python train/train_magnetic_sequence.py
```

The magnetic sequence window was selected on the development protocol; no standalone magnetic final-test headline number is claimed in the paper.

The active temporal fusion implementation is:

```bash
python train/kalmannet_wifiheatmap_magneticCNN_pdr.py
```

The final paper-facing trajectory protocol is frozen under `benchmarks/final_protocol/`. Earlier development results are not canonical.

## Frozen final protocol

- Fusion training seed: **1** (250 trajectories)
- Development seed: **2**
- Seed **3**: retired after the Wi-Fi/magnetic registration bug was discovered
- Final test seed: **4** (60 trajectories × 160 bins)
- Heading: strictly causal backward displacement + simulated drift/white noise
- Path graph: Euclidean survey-node proximity graph, epsilon = 1.6 m; no wall/obstacle geometry
- Mean/CI: across 60 per-trajectory mean errors
- Median/P90/max/CDF: across all 9,600 pointwise localization errors

Canonical results and provenance:

```text
benchmarks/final_protocol/README.md
benchmarks/final_protocol/current_results/metrics.json
benchmarks/final_protocol/current_results/manifest.json
```

## Paper

Active manuscript:

```text
paper/main.tex
```

Build with:

```bash
cd paper
latexmk -pdf main.tex
```

Final audit records are under `paper/reviews/`, especially:

- `pre_final_reaudit_todo.md` — R1–R13 final disposition
- `final_claim_code_audit.md` — claim/equation-to-code audit
- `numerical_traceability.json` — manuscript-number traceability
- `repository_cleanup_record.md` — retained/removed inventory

## Scope

The reported temporal results are controlled synthetic trajectory experiments within one surveyed environment with a known initial 2-D position. They are not claimed as real continuously labeled trajectory validation, building-independent deployment, obstacle-aware path simulation, or causal magnetic-domain alignment for an uncalibrated unseen handset.
