#!/usr/bin/env python3
"""Validate a processed fingerprint database and print its summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sura.data.fingerprint import load_fingerprint_database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "data"
            / "processed"
            / "fingerprint_db"
            / "it_engineering"
        ),
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        help="optional mode filter; repeat for multiple modes",
    )
    parser.add_argument(
        "--allow-missing-wifi",
        action="store_true",
        help="do not filter visits without a Wi-Fi scan",
    )
    args = parser.parse_args()

    database = load_fingerprint_database(
        args.database,
        included_modes=args.modes,
        require_wifi=not args.allow_missing_wifi,
    )
    print(json.dumps(database.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
