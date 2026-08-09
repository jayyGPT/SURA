# Data workflows

Use the package commands rather than archived one-off scripts:

```bash
python -m sura data init
python -m sura data migrate-legacy
python -m sura data build-fingerprint
python -m sura data check
python -m sura data analyze
```

The complete placement and output structure is documented in [`../../data/README.md`](../../data/README.md).
