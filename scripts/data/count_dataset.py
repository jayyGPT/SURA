#!/usr/bin/env python3
"""Count the tracked MagWi upload and compare it with the earlier dataset audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "data" / "raw" / "magwi"
IGNORED_NAMES = {".DS_Store", ".gitkeep", "Thumbs.db"}

# Exact source-file counts in the GitHub upload that this repository now tracks.
TRACKED_SNAPSHOT_COUNTS = {
    "dataset_files": 7093,
    "magnetic_files": 4262,
    "wifi_files": 2831,
}

# Counts from the earlier exhaustive local audit. These are shown as a completeness reference,
# not enforced by default, because the current GitHub upload contains fewer Wi-Fi files.
REFERENCE_AUDIT_COUNTS = {
    "dataset_files": 8660,
    "magnetic_files": 4261,
    "wifi_files": 4399,
}


def _files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    )


def _group_by_building(files: list[Path], root: Path, prefix: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    prefix_length = len(prefix)
    for path in files:
        parts = path.relative_to(root).parts
        if parts[:prefix_length] == prefix and len(parts) > prefix_length:
            counts[parts[prefix_length]] += 1
    return dict(sorted(counts.items()))


def _comparison(payload: dict[str, object], expected: dict[str, int]) -> dict[str, object]:
    return {
        name: {
            "expected": count,
            "actual": int(payload[name]),
            "difference": int(payload[name]) - count,
            "matches": int(payload[name]) == count,
        }
        for name, count in expected.items()
    }


def build_inventory(dataset_root: Path, repository_root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_root}")

    dataset_files = _files(dataset_root)
    magnetic_root = dataset_root / "Magnetic field dataset"
    wifi_root = dataset_root / "WiFi dataset"
    magnetic_files = _files(magnetic_root) if magnetic_root.is_dir() else []
    wifi_files = _files(wifi_root) if wifi_root.is_dir() else []
    static_root = magnetic_root / "Static Data"
    continuous_root = magnetic_root / "Continuous Data"
    static_files = _files(static_root) if static_root.is_dir() else []
    continuous_files = _files(continuous_root) if continuous_root.is_dir() else []

    repository_files = [
        path
        for path in repository_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    known_top_level = {"Magnetic field dataset", "WiFi dataset"}
    extra_files = [
        str(path.relative_to(dataset_root))
        for path in dataset_files
        if path.relative_to(dataset_root).parts[0] not in known_top_level
    ]

    payload: dict[str, object] = {
        "repository_root": str(repository_root.resolve()),
        "repository_files": len(repository_files),
        "dataset_root": str(dataset_root),
        "dataset_files": len(dataset_files),
        "dataset_bytes": sum(path.stat().st_size for path in dataset_files),
        "magnetic_files": len(magnetic_files),
        "magnetic_static_files": len(static_files),
        "magnetic_continuous_files": len(continuous_files),
        "wifi_files": len(wifi_files),
        "extra_files": extra_files,
        "extensions": dict(
            sorted(Counter(path.suffix.lower() or "<none>" for path in dataset_files).items())
        ),
        "magnetic_static_by_building": _group_by_building(
            dataset_files,
            dataset_root,
            ("Magnetic field dataset", "Static Data"),
        ),
        "magnetic_continuous_by_building": _group_by_building(
            dataset_files,
            dataset_root,
            ("Magnetic field dataset", "Continuous Data"),
        ),
        "wifi_by_building": _group_by_building(
            dataset_files,
            dataset_root,
            ("WiFi dataset",),
        ),
    }
    payload["tracked_snapshot_comparison"] = _comparison(payload, TRACKED_SNAPSHOT_COUNTS)
    payload["reference_audit_comparison"] = _comparison(payload, REFERENCE_AUDIT_COUNTS)
    return payload


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{value} B"


def _print_comparison(title: str, comparison: dict[str, object]) -> None:
    print(f"\n{title}:")
    for name, result in comparison.items():
        marker = "OK" if result["matches"] else "DIFF"
        print(
            f"  {name:20s} expected={result['expected']:,} "
            f"actual={result['actual']:,} difference={result['difference']:+,} [{marker}]"
        )


def print_inventory(payload: dict[str, object]) -> None:
    print(f"Repository files:          {payload['repository_files']:,}")
    print(f"Raw source files:          {payload['dataset_files']:,}")
    print(f"Raw source size:           {_human_bytes(int(payload['dataset_bytes']))}")
    print(f"Magnetic files:            {payload['magnetic_files']:,}")
    print(f"  Static:                  {payload['magnetic_static_files']:,}")
    print(f"  Continuous:              {payload['magnetic_continuous_files']:,}")
    print(f"Wi-Fi files:               {payload['wifi_files']:,}")
    print(f"Unexpected raw files:      {len(payload['extra_files']):,}")
    _print_comparison("Tracked GitHub snapshot", payload["tracked_snapshot_comparison"])
    _print_comparison("Earlier full-dataset audit reference", payload["reference_audit_comparison"])

    print("\nFiles by building:")
    for label, key in (
        ("Magnetic static", "magnetic_static_by_building"),
        ("Magnetic continuous", "magnetic_continuous_by_building"),
        ("Wi-Fi", "wifi_by_building"),
    ):
        print(f"  {label}:")
        for building, count in payload[key].items():
            print(f"    {building}: {count:,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="raw MagWi directory (default: data/raw/magwi)",
    )
    parser.add_argument("--json", type=Path, help="write the inventory to this JSON file")
    parser.add_argument(
        "--verify-snapshot",
        action="store_true",
        help="return an error if tracked source-file counts change unexpectedly",
    )
    args = parser.parse_args()

    payload = build_inventory(args.dataset_root)
    print_inventory(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote: {args.json}")

    if args.verify_snapshot:
        mismatches = [
            name
            for name, result in payload["tracked_snapshot_comparison"].items()
            if not result["matches"]
        ]
        if mismatches or payload["extra_files"]:
            print(f"\nSnapshot verification failed: {', '.join(mismatches) or 'extra files'}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
