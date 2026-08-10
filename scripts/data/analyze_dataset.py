#!/usr/bin/env python3
"""Create a compact raw-data and fingerprint-database report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sura.data.analysis import analyze_dataset, report_markdown, write_dataset_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="processed fingerprint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "runs" / "dataset_analysis",
    )
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args()

    database = args.database or (
        args.data_root / "processed" / "fingerprint_db" / "it_engineering"
    )
    report = analyze_dataset(
        data_directory=args.data_root,
        fingerprint_directory=database,
    )
    print(report_markdown(report))
    if not args.no_write:
        json_path, markdown_path = write_dataset_report(report, args.output)
        print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
