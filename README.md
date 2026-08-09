# SURA Indoor Localization

Research code, experiments, documentation, and the IEEE manuscript for a strictly causal indoor-localization system using smartphone Wi-Fi, magnetic, and inertial measurements.

The project was developed under the Summer Undergraduate Research Award at IIT Delhi by Jayendra Vijay Birhade and Utkarsh Agrawal under the supervision of Dr. Neel Kanth Kundu.

## Research objective

The system separates the localization problem into three interpretable components:

1. **Spatial measurement models** learn the environment from surveyed Wi-Fi and magnetic fingerprints.
2. **Pedestrian dead reckoning (PDR)** converts IMU streams into causal relative motion.
3. **Neural Kalman fusion** learns context-dependent gains that combine spatial fixes with motion while handling unavailable sensors.

The current validated repository snapshot contains a Wi-Fi heatmap MLP, a magnetic sequence CNN, PDR, and an anomaly-based DualKalmanNet. The next research milestone replaces the anomaly-map correction with the magnetic CNN's position and uncertainty outputs. This architecture change will be implemented and benchmarked only after the repository renovation is complete.

## Canonical repository layout

```text
SURA/
├── src/sura/                 Reusable active Python implementation
├── scripts/                  Training, evaluation, data, and figure entry points
├── configs/                  Reproducible experiment configuration
├── data/                     Local dataset layout; raw and processed data are ignored
├── experiments/              Validated metrics and ignored run artifacts
├── baselines/                Maintained comparison methods
├── paper/                    Canonical IEEE LaTeX workspace
├── docs/                     Architecture, dataset, decisions, and project history
├── references/               Literature library and index
├── tests/                    Fast structural and model smoke tests
└── archive/                  Historical, generated, or non-canonical material
```

A detailed old-to-new path map is maintained in [`MIGRATION.md`](MIGRATION.md).

## Source-of-truth rules

- Active reusable code belongs only under `src/sura/`.
- Executable workflows belong under `scripts/` and import the package.
- Paper source belongs only under `paper/`.
- Headline paper values must be recorded under `experiments/results/` before they are written into the manuscript.
- Files under `archive/` are retained for provenance and must not be treated as current implementations or current results.
- Raw datasets, generated checkpoints, caches, and LaTeX build products are not committed.

## Local setup

Create a Python environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The full MagWi dataset is not stored in Git. Place it under `data/raw/` or set:

```bash
export SURA_DATA_ROOT=/absolute/path/to/data
```

See [`data/README.md`](data/README.md) for the expected layout.

## Validation

Run fast repository checks with:

```bash
pytest
```

Compile the paper with:

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

## Current development sequence

1. Complete and review repository renovation.
2. Establish reproducible baseline runs from the canonical code.
3. Implement CNN-output magnetic measurement fusion in DualKalmanNet.
4. Retrain and compare Wi-Fi-only, anomaly-based legacy, and CNN-based dual fusion.
5. Update equations, figures, results, and claims in the manuscript from verified outputs.

## Preservation

The repository state before this renovation is preserved on the branch:

```text
archive/pre-renovation-2026-08-09
```
