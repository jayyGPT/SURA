# Experiment configuration

Checked-in YAML files define reproducible model and dataset defaults.

```text
configs/
├── datasets/magwi_it_engineering.yaml
├── wifi_heatmap.yaml
├── magnetic_sequence.yaml
├── wifi_kalmannet.yaml
├── dual_kalmannet_anomaly.yaml
└── testing/                         tiny CI-only settings
```

Use another file without editing code:

```bash
python -m sura train wifi --config path/to/wifi.yaml
python -m sura train magnetic --config path/to/magnetic.yaml
```

Dataset locations are resolved separately through `--data-root`, `SURA_DATA_ROOT`, or the
repository's `data/` directory. Do not put machine-specific absolute paths in checked-in YAML.
