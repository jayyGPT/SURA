# Experiments

`runs/` is ignored and stores checkpoints, raw logs, and exploratory artifacts. `results/` stores compact reviewed summaries that support paper tables and figures.

A reproducible run directory should contain:

```text
config.yaml
metrics.json
seed.txt
git_commit.txt
notes.md
```

A metric should move into `results/` only after its split, seed, configuration, and implementation have been reviewed.
