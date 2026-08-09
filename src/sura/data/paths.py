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


def configured_data_path(
    value: str | Path,
    root: str | Path | None = None,
) -> Path:
    """Resolve an absolute or data-root-relative path from configuration."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (data_root(root) / path).resolve()


def raw_dataset_root(
    dataset: str = "magwi",
    root: str | Path | None = None,
) -> Path:
    """Return the canonical directory containing one raw dataset."""
    return data_root(root) / "raw" / dataset


def fingerprint_database(
    building: str = "it_engineering",
    root: str | Path | None = None,
) -> Path:
    """Return the canonical processed fingerprint-database directory."""
    return data_root(root) / "processed" / "fingerprint_db" / building


def experiment_runs_root(explicit: str | Path | None = None) -> Path:
    """Return the ignored directory used for checkpoints and raw run outputs."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return repository_root() / "experiments" / "runs"


def paper_root() -> Path:
    """Return the canonical LaTeX workspace."""
    return repository_root() / "paper"
