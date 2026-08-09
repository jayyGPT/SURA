"""Portable paths for local datasets and generated artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[3]


def data_root(explicit: str | Path | None = None) -> Path:
    """Resolve the data root from an argument, environment, or repository default."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configured = os.getenv("SURA_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return repository_root() / "data"


def fingerprint_database(
    building: str = "it_engineering",
    root: str | Path | None = None,
) -> Path:
    """Return the canonical processed fingerprint-database directory."""
    return data_root(root) / "processed" / "fingerprint_db" / building
