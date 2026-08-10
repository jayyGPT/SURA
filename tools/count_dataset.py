#!/usr/bin/env python3
"""Count the tracked MagWi source snapshot."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "magwi"

EXPECTED = {
    "magnetic_static": 4135,
    "magnetic_continuous": 127,
    "wifi": 2831,
}


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def by_building(root: Path) -> dict[str, int]:
    if not root.is_dir():
        return {}
    counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts:
            counts[relative.parts[0]] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    magnetic = DATASET_ROOT / "Magnetic field dataset"
    static = magnetic / "Static Data"
    continuous = magnetic / "Continuous Data"
    wifi = DATASET_ROOT / "WiFi dataset"

    actual = {
        "magnetic_static": count_files(static),
        "magnetic_continuous": count_files(continuous),
        "wifi": count_files(wifi),
    }
    total = sum(actual.values())

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Magnetic static:     {actual['magnetic_static']:,}")
    print(f"Magnetic continuous: {actual['magnetic_continuous']:,}")
    print(f"Wi-Fi:               {actual['wifi']:,}")
    print(f"Total source files:  {total:,}")
    print()
    print("Verified original-upload counts:")
    for key, expected in EXPECTED.items():
        marker = "OK" if actual[key] == expected else "DIFF"
        print(f"  {key:22s} {actual[key]:,} / {expected:,} [{marker}]")

    print("\nWi-Fi files by building:")
    for building, value in by_building(wifi).items():
        print(f"  {building}: {value:,}")

    return 0 if actual == EXPECTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
