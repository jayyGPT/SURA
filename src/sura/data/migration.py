"""Safe migration of the repository's former ``Datasets/`` layout."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .paths import data_root, repository_root


def legacy_migration_plan(
    *,
    legacy_directory: str | Path | None = None,
    target_data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Describe known legacy-to-canonical moves without modifying files."""
    source = (
        Path(legacy_directory).expanduser().resolve()
        if legacy_directory is not None
        else repository_root() / "Datasets"
    )
    target = data_root(target_data_root)
    mappings = [
        (
            source / "Magnetic field dataset",
            target / "raw" / "magwi" / "Magnetic field dataset",
        ),
        (
            source / "WiFi dataset",
            target / "raw" / "magwi" / "WiFi dataset",
        ),
        (
            source / "fingerprint_db",
            target / "processed" / "fingerprint_db",
        ),
    ]
    plan = [
        {
            "source": str(old),
            "target": str(new),
            "source_exists": old.exists(),
            "target_exists": new.exists(),
            "kind": "directory",
        }
        for old, new in mappings
    ]
    for fused_file in sorted(source.glob("Continuous_Fused_*.csv")):
        target_file = target / "interim" / "legacy_fused" / fused_file.name
        plan.append(
            {
                "source": str(fused_file),
                "target": str(target_file),
                "source_exists": True,
                "target_exists": target_file.exists(),
                "kind": "file",
            }
        )
    return plan


def migrate_legacy_data(
    *,
    legacy_directory: str | Path | None = None,
    target_data_root: str | Path | None = None,
    mode: str = "move",
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Move or copy known legacy data paths; dry-run unless ``apply`` is true."""
    if mode not in {"move", "copy"}:
        raise ValueError("mode must be 'move' or 'copy'")
    plan = legacy_migration_plan(
        legacy_directory=legacy_directory,
        target_data_root=target_data_root,
    )
    if not apply:
        return plan

    completed: list[dict[str, Any]] = []
    for item in plan:
        source = Path(item["source"])
        target = Path(item["target"])
        result = dict(item)
        if not source.exists():
            result["status"] = "missing"
        elif target.exists():
            result["status"] = "skipped_target_exists"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "move":
                shutil.move(str(source), str(target))
            elif source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            result["status"] = mode
        completed.append(result)
    return completed
