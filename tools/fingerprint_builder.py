"""Build a processed Wi-Fi/magnetic fingerprint database from raw MagWi files.

For the IT Engineering data used in this project, the common localization frame is
anchored to the static magnetic survey coordinates. A Wi-Fi scan is attached to a
magnetic visit only when mode, scenario, phone, user, and the timestamped filename
match exactly after normalization. This avoids basename-only cross-device pairing.
The Wi-Fi workbook's own coordinate fields are preserved as audit metadata and are
never silently treated as the common localization coordinate.
"""

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
PAIRING_RULE = "exact_mode_scenario_phone_user_timestamp_filename"


@dataclass(frozen=True)
class FingerprintBuildSummary:
    building: str
    magnetic_files: int
    wifi_files: int
    magnetic_visits: int
    wifi_visits: int
    parsed_visits: int
    unique_nodes: int
    access_points: int
    output_directory: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9+]+", "", str(value).strip().lower())


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


def _pair_key(path: Path, root: Path) -> tuple[str, str, str, str, str]:
    mode, scenario, phone, user = _parse_metadata(path, root)
    return (
        _normalise_token(mode),
        _normalise_token(scenario),
        _normalise_token(phone),
        _normalise_token(user),
        path.name,
    )


def _expected_wifi_name(magnetic_name: str) -> str:
    return magnetic_name.replace("IMU_", "WiFi_", 1) if magnetic_name.startswith("IMU_") else magnetic_name


def magnetic_rotation_invariant_statistics(frame: pd.DataFrame) -> dict[str, float]:
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


def _read_wifi_visit(path: Path) -> tuple[float, float, dict[str, float]] | None:
    """Read one MagWi Wi-Fi workbook (BIFF8 data with a .csv suffix)."""
    def attempt(reader):
        try:
            return reader()
        except Exception:  # pragma: no cover - depends on source file support
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
    required = {"X-pos", "Y-pos", "BSSID", "RSS"}
    if not required.issubset(frame.columns):
        return None

    x_values = pd.to_numeric(frame["X-pos"], errors="coerce").dropna()
    y_values = pd.to_numeric(frame["Y-pos"], errors="coerce").dropna()
    if x_values.empty or y_values.empty:
        return None

    scan_frame = frame.dropna(subset=["BSSID", "RSS"]).copy()
    scan_frame["BSSID"] = scan_frame["BSSID"].astype(str).str.strip()
    scan_frame["RSS"] = pd.to_numeric(scan_frame["RSS"], errors="coerce")
    scan_frame = scan_frame[(scan_frame["BSSID"] != "") & scan_frame["RSS"].notna()]
    if scan_frame.empty:
        return None
    scan = scan_frame.groupby("BSSID")["RSS"].max().astype(float).to_dict()
    return float(x_values.iloc[0]), float(y_values.iloc[0]), scan


def _find_files(root: Path, suffixes: Iterable[str] = (".csv",)) -> list[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


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
    """Build one magnetic-survey-frame row per static magnetic visit."""
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
        return FingerprintBuildSummary(building, len(magnetic_files), len(wifi_files), 0, 0, 0, 0, 0, str(output))
    if not magnetic_files:
        raise ValueError(f"no magnetic CSV files found under {magnetic_root}")

    # Exact metadata-aware index. Duplicate keys are retained and therefore rejected.
    wifi_index: dict[tuple[str, str, str, str, str], list[Path]] = {}
    for wifi_path in wifi_files:
        wifi_index.setdefault(_pair_key(wifi_path, wifi_root), []).append(wifi_path)

    records: list[dict[str, object]] = []
    scans: list[dict[str, float] | None] = []
    bssid_frequency: dict[str, int] = {}
    audit = {
        "pairing_rule": PAIRING_RULE,
        "magnetic_files_total": len(magnetic_files),
        "wifi_files_total": len(wifi_files),
        "magnetic_visits_parsed": 0,
        "wifi_exact_candidates": 0,
        "wifi_exact_unique_attached": 0,
        "wifi_exact_duplicate_key_rejected": 0,
        "wifi_exact_unreadable_rejected": 0,
        "wifi_no_exact_match": 0,
    }

    required_rows = ["X-cord", "Y-cord", "Mag_x", "Mag_y", "Mag_z", "Acc_x", "Acc_y", "Acc_z"]
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
        expected_wifi = _expected_wifi_name(magnetic_path.name)
        key = (
            _normalise_token(mode),
            _normalise_token(scenario),
            _normalise_token(phone),
            _normalise_token(user),
            expected_wifi,
        )
        candidates = wifi_index.get(key, [])
        scan = None
        wifi_x_raw = float("nan")
        wifi_y_raw = float("nan")
        wifi_file = ""
        pairing_status = "no_exact_match"
        if candidates:
            audit["wifi_exact_candidates"] += 1
        if len(candidates) == 1:
            visit = _read_wifi_visit(candidates[0])
            if visit is not None:
                wifi_x_raw, wifi_y_raw, scan = visit
                wifi_file = candidates[0].name
                pairing_status = "exact_unique"
                audit["wifi_exact_unique_attached"] += 1
                for bssid in scan:
                    bssid_frequency[bssid] = bssid_frequency.get(bssid, 0) + 1
            else:
                pairing_status = "exact_unreadable"
                audit["wifi_exact_unreadable_rejected"] += 1
        elif len(candidates) > 1:
            pairing_status = "duplicate_exact_key_rejected"
            audit["wifi_exact_duplicate_key_rejected"] += 1
        else:
            audit["wifi_no_exact_match"] += 1

        record: dict[str, object] = {
            "x": float(frame["X-cord"].iloc[0]),
            "y": float(frame["Y-cord"].iloc[0]),
            "mode": mode,
            "scenario": scenario,
            "phone": phone,
            "user": user,
            "n_mag_rows": int(len(frame)),
            "file": magnetic_path.name,
            "wifi_file": wifi_file,
            "wifi_x_raw": wifi_x_raw,
            "wifi_y_raw": wifi_y_raw,
            "wifi_pairing_rule": PAIRING_RULE,
            "wifi_pairing_status": pairing_status,
            "n_ap": len(scan) if scan else 0,
            "has_wifi": bool(scan),
        }
        record.update(magnetic_rotation_invariant_statistics(frame))
        records.append(record)
        scans.append(scan)
        audit["magnetic_visits_parsed"] += 1

    if not records:
        raise ValueError("no valid magnetic fingerprint visits could be parsed")
    if not bssid_frequency:
        raise ValueError("no exact-metadata Wi-Fi scans were attached")

    vocabulary = sorted(bssid_frequency, key=lambda bssid: (-bssid_frequency[bssid], bssid))
    bssid_index = {bssid: index for index, bssid in enumerate(vocabulary)}
    metadata = pd.DataFrame(records)
    wifi_matrix = np.full((len(records), len(vocabulary)), WIFI_FLOOR_DBM, dtype=np.float32)
    for row, scan in enumerate(scans):
        if scan:
            for bssid, rss in scan.items():
                wifi_matrix[row, bssid_index[bssid]] = float(rss)
    ap_columns = [f"AP_{index}" for index in range(len(vocabulary))]
    full = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(wifi_matrix, columns=ap_columns)], axis=1)

    output.mkdir(parents=True, exist_ok=True)
    full.to_csv(output / "nodes.csv", index=False)
    with (output / "bssid_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump({"bssid_vocab": vocabulary, "wifi_floor": WIFI_FLOOR_DBM, "ap_columns": ap_columns}, handle, indent=2)
    audit.update({
        "rows_written": int(len(full)),
        "access_points": int(len(vocabulary)),
        "unique_magnetic_nodes": int(metadata[["x", "y"]].round(1).drop_duplicates().shape[0]),
    })
    (output / "pairing_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    unique_nodes = metadata[["x", "y"]].round(1).drop_duplicates()
    if make_coverage_plot:
        counts = metadata.assign(x_round=metadata["x"].round(1), y_round=metadata["y"].round(1)).groupby(["x_round", "y_round"]).size().reset_index(name="visits")
        figure, axis = plt.subplots(figsize=(12, 4))
        scatter = axis.scatter(counts["x_round"], counts["y_round"], c=counts["visits"], cmap="viridis", s=60, edgecolors="black", linewidths=0.3)
        figure.colorbar(scatter, ax=axis, label="magnetic survey visits")
        axis.set_title(f"{building} magnetic-survey-frame coverage")
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
        magnetic_visits=len(records),
        wifi_visits=int(metadata["has_wifi"].sum()),
        parsed_visits=len(full),
        unique_nodes=len(unique_nodes),
        access_points=len(vocabulary),
        output_directory=str(output),
    )
