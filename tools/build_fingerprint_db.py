#!/usr/bin/env python3
"""Build the processed fingerprint database from the tracked MagWi source data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fingerprint_builder import build_fingerprint_database, slugify

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "magwi"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--building", default="IT Engineering")
    parser.add_argument("--wifi-building")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output or (
        REPO_ROOT / "data" / "processed" / "fingerprint_db" / slugify(args.building)
    )
    summary = build_fingerprint_database(
        raw_dataset_directory=args.raw_dataset,
        output_directory=output,
        building=args.building,
        wifi_building=args.wifi_building,
        make_coverage_plot=not args.no_plot,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
