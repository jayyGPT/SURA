"""Create and inspect the standard local data directory layout."""

from __future__ import annotations

from pathlib import Path

from .paths import data_root

DATA_DIRECTORIES = ("raw", "interim", "processed", "local")


def initialize_data_layout(root: str | Path | None = None) -> dict[str, Path]:
    """Create the canonical ignored data directories and return their paths."""
    base = data_root(root)
    base.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}
    for name in DATA_DIRECTORIES:
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        created[name] = path
    return created
