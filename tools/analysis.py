"""Compact raw-data inventory and processed fingerprint analysis."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fingerprint import load_fingerprint_database


def _file_inventory(root: Path) -> dict[str, Any]:
    suffixes: Counter[str] = Counter()
    top_level: Counter[str] = Counter()
    total_bytes = 0
    file_count = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            total_bytes += path.stat().st_size
            suffixes[path.suffix.lower() or "<none>"] += 1
            relative = path.relative_to(root)
            top_level[relative.parts[0] if relative.parts else "."] += 1
    return {
        "directory": str(root),
        "exists": root.is_dir(),
        "files": file_count,
        "bytes": total_bytes,
        "extensions": dict(sorted(suffixes.items())),
        "top_level_file_counts": dict(sorted(top_level.items())),
    }


def analyze_dataset(
    *,
    data_directory: str | Path,
    fingerprint_directory: str | Path,
    included_modes: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable summary of raw and processed data."""
    data_root = Path(data_directory).expanduser().resolve()
    fingerprint_root = Path(fingerprint_directory).expanduser().resolve()
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "raw": _file_inventory(data_root / "raw"),
        "interim": _file_inventory(data_root / "interim"),
        "processed": _file_inventory(data_root / "processed"),
    }
    if fingerprint_root.is_dir():
        database = load_fingerprint_database(
            fingerprint_root,
            included_modes=included_modes,
            require_wifi=False,
        )
        report["fingerprint_database"] = database.summary()
    else:
        report["fingerprint_database"] = {
            "directory": str(fingerprint_root),
            "exists": False,
        }
    return report


def report_markdown(report: dict[str, Any]) -> str:
    """Render the compact dataset report as Markdown."""
    lines = [
        "# Dataset analysis",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Local directories",
        "",
        "| Area | Exists | Files | Size (bytes) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("raw", "interim", "processed"):
        section = report[name]
        lines.append(
            f"| {name} | {section['exists']} | {section['files']} | {section['bytes']} |"
        )

    fingerprint = report.get("fingerprint_database", {})
    lines.extend(["", "## Fingerprint database", ""])
    if fingerprint.get("exists") is False:
        lines.append(f"Not found at `{fingerprint['directory']}`.")
    else:
        lines.extend(
            [
                f"- Directory: `{fingerprint['directory']}`",
                f"- Node visits: **{fingerprint['visits']}**",
                f"- Unique nodes: **{fingerprint['unique_nodes']}**",
                f"- Access points: **{fingerprint['access_points']}**",
                f"- Wi-Fi coverage: **{fingerprint['wifi_coverage_fraction']:.1%}**",
                f"- Phones: {', '.join(fingerprint['phones'])}",
                f"- Modes: {', '.join(fingerprint['modes'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_dataset_report(
    report: dict[str, Any],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown report files."""
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "dataset_report.json"
    markdown_path = output / "dataset_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
    return json_path, markdown_path
