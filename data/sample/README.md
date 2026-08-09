# Tracked sample data

This directory contains a tiny synthetic fingerprint database used by unit tests and GitHub
Actions. It is not a scientific dataset and its outputs must never be reported in the paper.

Run both training smoke workflows manually with:

```bash
python -m sura train wifi \
  --data-root data/sample \
  --config configs/testing/wifi_heatmap_smoke.yaml \
  --split random \
  --epochs 1 \
  --output-dir /tmp/sura-smoke

python -m sura train magnetic \
  --data-root data/sample \
  --config configs/testing/magnetic_sequence_smoke.yaml \
  --epochs 1 \
  --output-dir /tmp/sura-smoke
```
