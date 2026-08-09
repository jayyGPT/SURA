"""Configuration loading for reproducible SURA workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sura.data.paths import repository_root


def default_config_path(name: str) -> Path:
    """Return a checked-in configuration path from the repository."""
    path = repository_root() / "configs" / name
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject malformed or empty configuration files."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a YAML mapping: {resolved}")
    return payload
