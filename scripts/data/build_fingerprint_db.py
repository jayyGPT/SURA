#!/usr/bin/env python3
"""Build the processed IT Engineering Wi-Fi and magnetic fingerprint database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sura.data.fingerprint_builder import build_fingerprint_database, slugify  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw" / "magwi",
        help="directory containing 'Magnetic field dataset' and 'WiFi dataset'",
    )
    parser.add_argument("--building", default="IT Engineering")
    parser.add_argument(
        "--wifi-building",
        help="Wi-Fi building folder when it differs from the magnetic building name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: data/processed/fingerprint_db/<building>)",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip coverage.png")
    parser.add_argument("--dry-run", action="store_true", help="only count matching inputs")
    args = parser.parse_args()

    output = args.output or (
        REPOSITORY_ROOT
        / "data"
        / "processed"
        / "fingerprint_db"
        / slugify(args.building)
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
