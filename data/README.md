# Data

Raw MagWi source data are retained under `data/raw/magwi/`. The paper uses the IT Engineering processed database under `data/processed/fingerprint_db/it_engineering/`.

The processed database is anchored to the static magnetic survey coordinate frame. A Wi-Fi scan is attached only by exact normalized mode/scenario/phone/user plus timestamped filename. `nodes.csv` retains the raw Wi-Fi coordinate as `wifi_x_raw,wifi_y_raw` for audit, while `pairing_audit.json` records attachment counts and coordinate-discrepancy statistics.

Rebuild and check:

```bash
python tools/build_fingerprint_db.py
python tools/check_fingerprint_db.py
```

Do not rename the original MagWi directories; the builder follows their source layout.
