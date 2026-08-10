from __future__ import annotations

from pathlib import Path

import pandas as pd

from sura.data.fingerprint import load_fingerprint_database


def _write_nodes(directory: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "x": 1.0,
                "y": 2.0,
                "mode": "Navigation",
                "phone": "S8",
                "has_wifi": True,
                "AP_0": -55.0,
            }
        ]
    )
    frame.to_csv(directory / "nodes.csv", index=False)


def test_loads_valid_vocabulary_json(tmp_path: Path) -> None:
    _write_nodes(tmp_path)
    (tmp_path / "bssid_vocab.json").write_text(
        '{"ap_columns": ["AP_0"], "wifi_floor": -100.0}\n',
        encoding="utf-8",
    )

    database = load_fingerprint_database(tmp_path)

    assert database.access_point_columns == ("AP_0",)
    assert len(database.frame) == 1


def test_recovers_uploaded_legacy_vocabulary_missing_opening_brace(tmp_path: Path) -> None:
    _write_nodes(tmp_path)
    (tmp_path / "bssid_vocab.json").write_text(
        '"bssid_vocab": ["example"], '
        '"wifi_floor": -100.0, "ap_columns": ["AP_0"]}\n',
        encoding="utf-8",
    )

    database = load_fingerprint_database(tmp_path)

    assert database.access_point_columns == ("AP_0",)
    assert len(database.frame) == 1
