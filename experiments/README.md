# Experiments

`runs/` is ignored and stores checkpoints, raw logs, predictions, generated dataset reports,
and exploratory artifacts. `results/` stores compact, reviewed summaries that support paper
tables and figures.

Canonical commands create self-describing runs:

```text
experiments/runs/<model>/<run-name>/
├── run.json
├── <split-or-window>/
│   ├── model.pt
│   ├── predictions.npz
│   └── history.json
└── ...
```

`run.json` records the effective configuration, dataset summary, random seed, Git commit,
device, output paths, and final metrics.

A value may move into `experiments/results/` only after its split, seed, configuration,
implementation, and generated artifacts have been reviewed. The manuscript must cite values
from `results/`, not directly from an arbitrary local run.
