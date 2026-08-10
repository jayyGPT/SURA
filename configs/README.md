# Experiment configuration

The training scripts use these checked-in YAML defaults:

```text
configs/datasets/magwi_it_engineering.yaml
configs/wifi_heatmap.yaml
configs/magnetic_sequence.yaml
```

Normally no configuration argument is needed:

```bash
python scripts/train/train_wifi_heatmap.py
python scripts/train/train_magnetic_sequence.py
```

To use another file:

```bash
python scripts/train/train_wifi_heatmap.py --config path/to/wifi.yaml
python scripts/train/train_magnetic_sequence.py --config path/to/magnetic.yaml
```

The `configs/testing/` files are tiny settings used by automated smoke tests.
