# Training scripts

Install dependencies once from the repository root:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
cd scripts/train
python train_wifi_heatmap.py
python train_magnetic_sequence.py
```

Both scripts also work from the repository root:

```bash
python scripts/train/train_wifi_heatmap.py
python scripts/train/train_magnetic_sequence.py
```

Use `--dry-run` to verify data loading and model construction without training. Use `--epochs`
for a short run, `--device cuda` for a GPU, and `--help` for all options. The magnetic script
also supports `--sweep` for the configured window-size sweep.

Outputs are written to `experiments/runs/`.
