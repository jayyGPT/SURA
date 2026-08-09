# Local data layout

The MagWi dataset and generated model-ready data are intentionally excluded from Git.

```text
data/
├── raw/          untouched downloaded dataset
├── interim/      decoded or normalized intermediate files
├── processed/    fingerprint databases and model-ready arrays
└── sample/       tiny tracked fixtures suitable for unit tests
```

The expected fingerprint database path is:

```text
data/processed/fingerprint_db/it_engineering/
├── nodes.csv
└── bssid_vocab.json
```

Set `SURA_DATA_ROOT` to use a data directory outside the repository:

```bash
export SURA_DATA_ROOT=/absolute/path/to/sura-data
```

Do not commit participant data, raw sensor recordings, or generated training arrays.
