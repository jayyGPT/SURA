"""Build a processed Wi-Fi and magnetic fingerprint database from raw MagWi files."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WIFI_FLOOR_DBM = -100.0
PHONE_TOKENS = ("S9+", "A8", "G7", "S8", "LG G6", "LG Q6", "G6", "Q6")


@dataclass(frozen=True)
class FingerprintBuildSummary:
    building: str
    magnetic_files: int
    wifi_files: int
    parsed_visits: int
    wifi_paired_visits: int
    unique_nodes: int
    access_points: int
    output_directory: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _parse_metadata(path: Path, building_root: Path) -> tuple[str, str, str, str]:
    try:
        components = path.relative_to(building_root).parts[:-1]
    except ValueError:
        components = path.parts[:-1]
    mode = components[0] if components else "Unknown"
    scenario = next((part for part in components if "scenario" in part.lower()), "NA")
    phone = next((part for part in components if part in PHONE_TOKENS), "Unknown")
    user = next(
        (part for part in components if part.replace(" ", "").lower().startswith("user")),
        "Unknown",
    )
    return mode, scenario, phone, user


def magnetic_rotation_invariant_statistics(frame: pd.DataFrame) -> dict[str, float]:
    """Return mean and standard deviation of four gravity-referenced features."""
    required = {"Mag_x", "Mag_y", "Mag_z", "Acc_x", "Acc_y", "Acc_z"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"magnetic file is missing columns: {', '.join(sorted(missing))}")

    magnetic = frame[["Mag_x", "Mag_y", "Mag_z"]].to_numpy(dtype=float)
    acceleration = frame[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(dtype=float)
    magnetic_norm = np.linalg.norm(magnetic, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    acceleration_norm[acceleration_norm == 0] = np.nan
    gravity = acceleration / acceleration_norm[:, None]

    vertical = np.sum(magnetic * gravity, axis=1)
    horizontal = np.sqrt(np.maximum(magnetic_norm**2 - vertical**2, 0.0))
    dip = np.arctan2(vertical, horizontal)

    result: dict[str, float] = {}
    for name, values in {
        "magN": magnetic_norm,
        "magV": vertical,
        "magH": horizontal,
        "dip": dip,
    }.items():
        finite = values[np.isfinite(values)]
        result[f"{name}_mean"] = float(finite.mean()) if finite.size else float("nan")
        result[f"{name}_std"] = float(finite.std()) if finite.size else float("nan")
    return result


def _wifi_basename(magnetic_name: str) -> str | None:
    if magnetic_name.startswith("IMU_"):
        return "WiFi_" + magnetic_name[len("IMU_") :]
    return None


def _read_wifi_scan(path: Path) -> dict[str, float] | None:
    """Read MagWi's BIFF8 Wi-Fi file even when it has a ``.csv`` suffix."""
    def attempt(reader):
        try:
            return reader()
        except Exception:  # pragma: no cover - depends on source file format
            return None

    frame = None
    for reader in (
        lambda: pd.read_excel(path, engine="xlrd"),
        lambda: pd.read_excel(path),
        lambda: pd.read_csv(path),
    ):
        frame = attempt(reader)
        if frame is not None:
            break
    if frame is None:
        return None

    frame.columns = [str(column).strip() for column in frame.columns]
    if "BSSID" not in frame.columns or "RSS" not in frame.columns:
        return None
    frame = frame.dropna(subset=["BSSID", "RSS"]).copy()
    frame["BSSID"] = frame["BSSID"].astype(str)
    frame["RSS"] = pd.to_numeric(frame["RSS"], errors="coerce")
    frame = frame.dropna(subset=["RSS"])
    if frame.empty:
        return None
    return frame.groupby("BSSID")["RSS"].max().astype(float).to_dict()


def _find_files(root: Path, suffixes: Iterable[str] = (".csv",)) -> list[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )


def build_fingerprint_database(
    *,
    raw_dataset_directory: str | Path,
    output_directory: str | Path,
    building: str = "IT Engineering",
    wifi_building: str | None = None,
    magnetic_static_subdirectory: str | Path = "Magnetic field dataset/Static Data",
    wifi_subdirectory: str | Path = "WiFi dataset",
    make_coverage_plot: bool = True,
    dry_run: bool = False,
) -> FingerprintBuildSummary:
    """Build ``nodes.csv`` and ``bssid_vocab.json`` for one building."""
    raw_root = Path(raw_dataset_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    wifi_building = wifi_building or building
    magnetic_root = raw_root / Path(magnetic_static_subdirectory) / building
    wifi_root = raw_root / Path(wifi_subdirectory) / wifi_building

    if not magnetic_root.is_dir():
        raise FileNotFoundError(f"magnetic static directory not found: {magnetic_root}")
    if not wifi_root.is_dir():
        raise FileNotFoundError(f"Wi-Fi directory not found: {wifi_root}")

    magnetic_files = _find_files(magnetic_root)
    wifi_files = _find_files(wifi_root)
    if dry_run:
        return FingerprintBuildSummary(
            building=building,
            magnetic_files=len(magnetic_files),
            wifi_files=len(wifi_files),
            parsed_visits=0,
            wifi_paired_visits=0,
            unique_nodes=0,
            access_points=0,
            output_directory=str(output),
        )
    if not magnetic_files:
        raise ValueError(f"no magnetic CSV files found under {magnetic_root}")

    wifi_by_name = {path.name: path for path in wifi_files}
    records: list[dict[str, object]] = []
    scans: list[dict[str, float] | None] = []
    bssid_frequency: dict[str, int] = {}

    required_rows = [
        "X-cord",
        "Y-cord",
        "Mag_x",
        "Mag_y",
        "Mag_z",
        "Acc_x",
        "Acc_y",
        "Acc_z",
    ]
    for magnetic_path in magnetic_files:
        try:
            frame = pd.read_csv(magnetic_path)
        except Exception:
            continue
        if any(column not in frame.columns for column in required_rows):
            continue
        frame = frame.dropna(subset=required_rows)
        if frame.empty:
            continue

        mode, scenario, phone, user = _parse_metadata(magnetic_path, magnetic_root)
        record: dict[str, object] = {
            "x": float(frame["X-cord"].iloc[0]),
            "y": float(frame["Y-cord"].iloc[0]),
            "mode": mode,
            "scenario": scenario,
            "phone": phone,
            "user": user,
            "n_mag_rows": int(len(frame)),
            "file": magnetic_path.name,
        }
        record.update(magnetic_rotation_invariant_statistics(frame))

        scan: dict[str, float] | None = None
        wifi_name = _wifi_basename(magnetic_path.name)
        if wifi_name and wifi_name in wifi_by_name:
            scan = _read_wifi_scan(wifi_by_name[wifi_name])
            if scan:
                for bssid in scan:
                    bssid_frequency[bssid] = bssid_frequency.get(bssid, 0) + 1

        records.append(record)
        scans.append(scan)

    if not records:
        raise ValueError("no valid magnetic node visits could be parsed")

    vocabulary = sorted(
        bssid_frequency,
        key=lambda bssid: (-bssid_frequency[bssid], bssid),
    )
    if not vocabulary:
        raise ValueError(
            "no matched Wi-Fi scans were parsed; verify raw paths and BIFF8 support"
        )
    bssid_index = {bssid: index for index, bssid in enumerate(vocabulary)}
    metadata = pd.DataFrame(records)
    metadata["n_ap"] = [len(scan) if scan else 0 for scan in scans]
    metadata["has_wifi"] = [bool(scan) for scan in scans]

    wifi_matrix = np.full(
        (len(records), len(vocabulary)),
        WIFI_FLOOR_DBM,
        dtype=np.float32,
    )
    for row, scan in enumerate(scans):
        if not scan:
            continue
        for bssid, rss in scan.items():
            wifi_matrix[row, bssid_index[bssid]] = float(rss)
    ap_columns = [f"AP_{index}" for index in range(len(vocabulary))]
    full = pd.concat(
        [metadata.reset_index(drop=True), pd.DataFrame(wifi_matrix, columns=ap_columns)],
        axis=1,
    )

    output.mkdir(parents=True, exist_ok=True)
    full.to_csv(output / "nodes.csv", index=False)
    with (output / "bssid_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "bssid_vocab": vocabulary,
                "wifi_floor": WIFI_FLOOR_DBM,
                "ap_columns": ap_columns,
            },
            handle,
            indent=2,
        )

    unique_nodes = metadata[["x", "y"]].round(1).drop_duplicates()
    if make_coverage_plot:
        counts = (
            metadata.assign(x_round=metadata["x"].round(1), y_round=metadata["y"].round(1))
            .groupby(["x_round", "y_round"])
            .size()
            .reset_index(name="visits")
        )
        figure, axis = plt.subplots(figsize=(12, 4))
        scatter = axis.scatter(
            counts["x_round"],
            counts["y_round"],
            c=counts["visits"],
            cmap="viridis",
            s=60,
            edgecolors="black",
            linewidths=0.3,
        )
        figure.colorbar(scatter, ax=axis, label="node visits")
        axis.set_title(f"{building} fingerprint coverage")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / "coverage.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    return FingerprintBuildSummary(
        building=building,
        magnetic_files=len(magnetic_files),
        wifi_files=len(wifi_files),
        parsed_visits=len(full),
        wifi_paired_visits=int(metadata["has_wifi"].sum()),
        unique_nodes=len(unique_nodes),
        access_points=len(vocabulary),
        output_directory=str(output),
    )
