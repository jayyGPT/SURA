#!/usr/bin/env python3
"""Validate the processed fingerprint database and print a short summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fingerprint import load_fingerprint_database

DEFAULT_DATABASE = REPO_ROOT / "data" / "processed" / "fingerprint_db" / "it_engineering"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--allow-missing-wifi", action="store_true")
    args = parser.parse_args()

    database = load_fingerprint_database(
        args.database,
        require_wifi=not args.allow_missing_wifi,
    )
    print(json.dumps(database.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
