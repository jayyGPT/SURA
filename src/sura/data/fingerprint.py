"""Loading and validation for processed fingerprint databases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_METADATA_COLUMNS = {"x", "y", "mode", "phone", "has_wifi"}


@dataclass(frozen=True)
class FingerprintDatabase:
    """A validated table of node visits and its ordered AP vocabulary."""

    directory: Path
    frame: pd.DataFrame
    access_point_columns: tuple[str, ...]

    @property
    def positions(self) -> np.ndarray:
        return self.frame[["x", "y"]].to_numpy(dtype=np.float32)

    def summary(self) -> dict[str, object]:
        rounded = self.frame[["x", "y"]].round(1)
        return {
            "directory": str(self.directory),
            "visits": int(len(self.frame)),
            "unique_nodes": int(len(rounded.drop_duplicates())),
            "access_points": int(len(self.access_point_columns)),
            "phones": sorted(str(value) for value in self.frame["phone"].dropna().unique()),
            "modes": sorted(str(value) for value in self.frame["mode"].dropna().unique()),
            "wifi_coverage_fraction": float(_boolean_series(self.frame["has_wifi"]).mean()),
            "x_range_m": [
                float(self.frame["x"].min()),
                float(self.frame["x"].max()),
            ],
            "y_range_m": [
                float(self.frame["y"].min()),
                float(self.frame["y"].max()),
            ],
        }


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _load_access_point_columns(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("ap_columns") or payload.get("access_points")
    else:
        values = None
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must define a non-empty 'ap_columns' list")
    columns = tuple(str(value) for value in values)
    if len(columns) != len(set(columns)):
        raise ValueError(f"duplicate AP columns found in {path}")
    return columns


def load_fingerprint_database(
    directory: str | Path,
    *,
    included_modes: Iterable[str] | None = None,
    require_wifi: bool = True,
) -> FingerprintDatabase:
    """Load and validate ``nodes.csv`` and ``bssid_vocab.json``."""
    root = Path(directory).expanduser().resolve()
    nodes_path = root / "nodes.csv"
    vocabulary_path = root / "bssid_vocab.json"
    if not nodes_path.is_file():
        raise FileNotFoundError(f"fingerprint table not found: {nodes_path}")
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"AP vocabulary not found: {vocabulary_path}")

    frame = pd.read_csv(nodes_path)
    missing_metadata = REQUIRED_METADATA_COLUMNS.difference(frame.columns)
    if missing_metadata:
        joined = ", ".join(sorted(missing_metadata))
        raise ValueError(f"fingerprint table is missing required columns: {joined}")

    access_points = _load_access_point_columns(vocabulary_path)
    missing_access_points = [column for column in access_points if column not in frame.columns]
    if missing_access_points:
        preview = ", ".join(missing_access_points[:5])
        raise ValueError(f"fingerprint table is missing AP columns: {preview}")

    frame = frame.copy()
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["x", "y"]).reset_index(drop=True)

    if included_modes is not None:
        allowed = {str(mode) for mode in included_modes}
        frame = frame[frame["mode"].astype(str).isin(allowed)].reset_index(drop=True)
    if require_wifi:
        frame = frame[_boolean_series(frame["has_wifi"])].reset_index(drop=True)

    if frame.empty:
        raise ValueError("no fingerprint visits remain after filtering")
    if not np.isfinite(frame[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("fingerprint coordinates contain non-finite values")

    numeric_ap = frame.loc[:, access_points].apply(pd.to_numeric, errors="coerce")
    if numeric_ap.isna().any().any():
        raise ValueError("one or more AP columns contain non-numeric values")
    frame.loc[:, access_points] = numeric_ap

    return FingerprintDatabase(root, frame, access_points)
