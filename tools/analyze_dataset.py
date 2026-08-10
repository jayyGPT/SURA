#!/usr/bin/env python3
"""Create a compact dataset and fingerprint-database report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analysis import analyze_dataset, report_markdown, write_dataset_report

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_DATABASE = DEFAULT_DATA / "processed" / "fingerprint_db" / "it_engineering"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "runs" / "dataset_analysis"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = analyze_dataset(
        data_directory=args.data_root,
        fingerprint_directory=args.database,
    )
    print(report_markdown(report))
    if not args.no_write:
        json_path, markdown_path = write_dataset_report(report, args.output)
        print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
