#!/usr/bin/env python3
"""Train/evaluate a DualKalmanNet that fuses Wi-Fi heatmap, magnetic CNN, and PDR.

This is the code-accurate replacement for the older scalar magnetic-anomaly fusion.
The magnetic branch used by the KalmanNet is the 2-D position estimate produced by
``MagSequenceMatcher``. Its predicted log-variance is also provided to the GRU as a
confidence feature.

The experiment keeps the previous synthetic-corridor evaluation protocol so the new
fusion can be compared with the Wi-Fi-only KalmanNet under full and degraded Wi-Fi.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.magnetic_sequence_cnn import MagSequenceMatcher
from models.pdr import StepDetector
from models.wifi_heatmap import Grid, WifiHeatmapNet, encode_wifi, soft_argmax
from models.wifi_kalmannet import WiFiOnlyKalmanNet
from tools.fingerprint import FingerprintDatabase, load_fingerprint_database

# ---------------------------------------------------------------------------
# Configuration. Edit here for normal experiments; CLI flags override run size.
# ---------------------------------------------------------------------------
SEED = 0
INCLUDED_MODES = ["Navigation", "Call listening", "Swinging"]

SAMPLING_HZ = 16.7
HEADING_WHITE_NOISE_RAD = np.deg2rad(8.8)
HEADING_DRIFT_STD_RAD = np.deg2rad(0.5) / np.sqrt(SAMPLING_HZ)
STEP_LENGTH_M = 0.65
SPEED_MIN_MPS = 1.0
SPEED_MAX_MPS = 1.35
STEP_FREQ_MIN_HZ = 1.7
STEP_FREQ_MAX_HZ = 2.0
MINIMUM_PATH_M = 30.0
CORRIDOR_EPSILON_M = 1.6

DEFAULT_T_BINS = 160
DEFAULT_TRAIN_WALKS = 250
DEFAULT_TEST_WALKS = 60
DEFAULT_FUSION_EPOCHS = 150
FUSION_BATCH_SIZE = 32
FUSION_LEARNING_RATE = 2e-3
FUSION_WEIGHT_DECAY = 1e-5
FUSION_HIDDEN_SIZE = 64

DEFAULT_DATABASE = REPO_ROOT / "data" / "processed" / "fingerprint_db" / "it_engineering"
DEFAULT_WIFI_CHECKPOINT = (
    REPO_ROOT / "archive" / "legacy_experiments" / "Models" / "dl_models" / "best_wifi_heatmap.pth"
)
DEFAULT_MAG_CHECKPOINT = (
    REPO_ROOT / "archive" / "legacy_experiments" / "Models" / "dl_models" / "best_mag_sequence.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "runs" / "cnn_dual_kalmannet"


@dataclass(frozen=True)
class MagneticMap:
    values: np.ndarray  # [C, nx, ny]
    noise_std: np.ndarray  # [C]
    grid: Grid
    features: tuple[str, ...]


@dataclass(frozen=True)
class CorridorGraph:
    coordinates: np.ndarray
    adjacency: csr_matrix
    component_nodes: np.ndarray


@dataclass(frozen=True)
class Environment:
    wifi_model: WifiHeatmapNet
    wifi_grid: Grid
    wifi_coordinates: Tensor
    wifi_pool_nodes: np.ndarray
    wifi_pool: dict[tuple[float, float], np.ndarray]
    access_points: tuple[str, ...]
    magnetic_model: MagSequenceMatcher
    magnetic_window: int
    magnetic_map: MagneticMap
    corridor: CorridorGraph


class CNNMagneticDualKalmanNet(nn.Module):
    """Dual matrix-gain KalmanNet using Wi-Fi and magnetic CNN spatial fixes.

    Per-step GRU features (13 total):
      - Wi-Fi innovation: z_wifi - x_pred                      [2]
      - Magnetic-CNN innovation: z_mag - x_pred                [2]
      - Change in Wi-Fi fix                                    [2]
      - PDR motion                                              [2]
      - Previous state update                                   [2]
      - Wi-Fi availability mask                                 [1]
      - Magnetic availability mask                              [1]
      - Magnetic CNN log-variance (confidence feature)          [1]

    The head outputs two independent 2x2 gains. No scalar magnetic anomaly map or
    spatial anomaly gradient is used anywhere in this model.
    """

    def __init__(self, hidden_size: int = FUSION_HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(13, hidden_size)
        self.head = nn.Linear(hidden_size, 8)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            # Start with a moderate Wi-Fi correction and a conservative magnetic branch.
            self.head.bias.copy_(
                torch.tensor([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
            )

    def forward(
        self,
        motion: Tensor,
        wifi_fix: Tensor,
        wifi_mask: Tensor,
        magnetic_fix: Tensor,
        magnetic_log_variance: Tensor,
        magnetic_mask: Tensor,
    ) -> Tensor:
        if motion.ndim != 3 or motion.shape[-1] != 2:
            raise ValueError("motion must have shape [B,T,2]")
        if wifi_fix.shape != motion.shape or magnetic_fix.shape != motion.shape:
            raise ValueError("Wi-Fi and magnetic fixes must have shape [B,T,2]")
        scalar_shape = motion.shape[:-1] + (1,)
        if wifi_mask.shape != scalar_shape or magnetic_mask.shape != scalar_shape:
            raise ValueError("availability masks must have shape [B,T,1]")
        if magnetic_log_variance.shape != scalar_shape:
            raise ValueError("magnetic_log_variance must have shape [B,T,1]")

        batch, steps, _ = motion.shape
        hidden = motion.new_zeros(batch, self.hidden_size)
        state = motion.new_zeros(batch, 2)
        previous_wifi = wifi_fix[:, 0]
        previous_update = motion.new_zeros(batch, 2)
        outputs: list[Tensor] = []

        for step in range(steps):
            wifi_available = wifi_mask[:, step]
            mag_available = magnetic_mask[:, step]
            predicted = state + motion[:, step]

            wifi_innovation = (wifi_fix[:, step] - predicted) * wifi_available
            magnetic_innovation = (magnetic_fix[:, step] - predicted) * mag_available
            wifi_delta = (wifi_fix[:, step] - previous_wifi) * wifi_available

            # Keep the confidence feature numerically tame. The raw spatial innovation is
            # still used in the correction; the GRU learns how strongly variance matters.
            mag_confidence = magnetic_log_variance[:, step].clamp(-6.0, 8.0) * mag_available

            features = torch.cat(
                [
                    wifi_innovation,
                    magnetic_innovation,
                    wifi_delta,
                    motion[:, step],
                    previous_update,
                    wifi_available,
                    mag_available,
                    mag_confidence,
                ],
                dim=1,
            )
            hidden = self.cell(features, hidden)
            gains = self.head(hidden)
            wifi_gain = gains[:, :4].view(batch, 2, 2)
            magnetic_gain = gains[:, 4:].view(batch, 2, 2)

            wifi_correction = wifi_available * torch.bmm(
                wifi_gain, wifi_innovation.unsqueeze(-1)
            ).squeeze(-1)
            magnetic_correction = mag_available * torch.bmm(
                magnetic_gain, magnetic_innovation.unsqueeze(-1)
            ).squeeze(-1)
            updated = predicted + wifi_correction + magnetic_correction

            previous_update = updated - state
            previous_wifi = torch.where(
                wifi_available.bool(), wifi_fix[:, step], previous_wifi
            )
            state = updated
            outputs.append(state)

        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Environment + checkpoint loading
# ---------------------------------------------------------------------------
def _boolean_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0).to_numpy(dtype=bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
        .to_numpy(dtype=bool)
    )


def load_wifi_model(
    database: FingerprintDatabase,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[WifiHeatmapNet, Grid, Tensor, tuple[str, ...]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Wi-Fi checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state") or payload.get("model_state")
    if state is None:
        raise ValueError(f"unsupported Wi-Fi checkpoint format: {checkpoint_path}")

    ap_columns = tuple(str(value) for value in payload.get("ap_cols", database.access_point_columns))
    missing = [column for column in ap_columns if column not in database.frame.columns]
    if missing:
        raise ValueError(f"Wi-Fi checkpoint AP columns are missing from database: {missing[:5]}")

    wifi_rows = database.frame[_boolean_mask(database.frame["has_wifi"])].copy()
    grid = Grid(wifi_rows["x"].to_numpy(float), wifi_rows["y"].to_numpy(float), cell=1.0)

    grid_meta = payload.get("grid")
    if isinstance(grid_meta, dict):
        expected = (
            float(grid_meta["x0"]),
            float(grid_meta["y0"]),
            int(grid_meta["nx"]),
            int(grid_meta["ny"]),
            float(grid_meta["cell"]),
        )
        actual = (grid.x0, grid.y0, grid.nx, grid.ny, grid.cell)
        if actual != expected:
            raise ValueError(f"Wi-Fi checkpoint grid {expected} does not match database grid {actual}")

    model = WifiHeatmapNet(len(ap_columns), grid.n_cells).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    coordinates = torch.tensor(grid.coords, dtype=torch.float32, device=device)
    return model, grid, coordinates, ap_columns


def load_magnetic_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MagSequenceMatcher, int, tuple[str, ...]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"magnetic CNN checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"unsupported magnetic checkpoint format: {checkpoint_path}")

    # Historical checkpoint names were pos_head/var_head; the cleaned model names are
    # position_head/variance_head. The tensor shapes are otherwise identical.
    mapped_state = {}
    for key, value in state.items():
        key = key.replace("pos_head.", "position_head.")
        key = key.replace("var_head.", "variance_head.")
        mapped_state[key] = value

    features = tuple(payload.get("features") or ["magN", "magV", "magH", "dip"])
    window = int(payload.get("window_frames") or payload.get("window_size") or 84)
    model = MagSequenceMatcher(in_channels=len(features), hidden_size=128).to(device)
    model.load_state_dict(mapped_state, strict=True)
    model.eval()
    return model, window, features


def build_magnetic_map(
    database: FingerprintDatabase,
    grid: Grid,
    features: tuple[str, ...],
) -> MagneticMap:
    frame = database.frame.copy()
    maps: list[np.ndarray] = []
    noise_std: list[float] = []

    for feature in features:
        column = f"{feature}_mean"
        if column not in frame.columns:
            raise ValueError(f"fingerprint table is missing magnetic feature: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid = frame.dropna(subset=[column]).copy()
        if valid.empty:
            raise ValueError(f"magnetic feature has no finite values: {column}")

        centered = f"_centered_{feature}"
        valid[centered] = valid[column] - valid.groupby("phone")[column].transform("mean")
        valid["x_round"] = valid["x"].round(1)
        valid["y_round"] = valid["y"].round(1)
        node_mean = valid.groupby(["x_round", "y_round"])[centered].mean()
        coordinates = np.asarray(list(node_mean.index), dtype=float)
        values = node_mean.to_numpy(dtype=float)

        linear = griddata(coordinates, values, grid.coords, method="linear")
        nearest = griddata(coordinates, values, grid.coords, method="nearest")
        maps.append(np.where(np.isnan(linear), nearest, linear).reshape(grid.nx, grid.ny))

        node_std = valid.groupby(["x_round", "y_round"])[centered].std()
        finite = node_std[np.isfinite(node_std.to_numpy(dtype=float))]
        estimate = float(finite.median()) if not finite.empty else 1.0
        if not np.isfinite(estimate) or estimate <= 0:
            estimate = max(float(valid[centered].std()) * 0.05, 1e-3)
        noise_std.append(estimate)

    return MagneticMap(
        values=np.asarray(maps, dtype=np.float32),
        noise_std=np.asarray(noise_std, dtype=np.float32),
        grid=grid,
        features=features,
    )


def build_corridor_graph(database: FingerprintDatabase) -> CorridorGraph:
    coordinates = database.frame[["x", "y"]].round(1).drop_duplicates().to_numpy(float)
    tree = cKDTree(coordinates)
    pairs = list(tree.query_pairs(CORRIDOR_EPSILON_M))
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for left, right in pairs:
        distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        rows.extend((left, right))
        columns.extend((right, left))
        weights.extend((distance, distance))
    adjacency = csr_matrix((weights, (rows, columns)), shape=(len(coordinates), len(coordinates)))
    count, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=count)
    component = np.flatnonzero(labels == int(np.argmax(sizes)))
    return CorridorGraph(coordinates, adjacency, component)


def build_wifi_pool(
    database: FingerprintDatabase,
    ap_columns: tuple[str, ...],
) -> tuple[np.ndarray, dict[tuple[float, float], np.ndarray]]:
    frame = database.frame[_boolean_mask(database.frame["has_wifi"])].copy()
    frame["x_round"] = frame["x"].round(1)
    frame["y_round"] = frame["y"].round(1)
    pool: dict[tuple[float, float], np.ndarray] = {}
    for (x, y), indices in frame.groupby(["x_round", "y_round"]).groups.items():
        pool[(float(x), float(y))] = frame.loc[indices, list(ap_columns)].to_numpy(np.float32)
    nodes = np.asarray(list(pool.keys()), dtype=float)
    if not len(nodes):
        raise ValueError("no Wi-Fi scans available for synthetic measurement pool")
    return nodes, pool


def setup_environment(
    database_path: Path,
    wifi_checkpoint: Path,
    magnetic_checkpoint: Path,
    device: torch.device,
) -> Environment:
    database = load_fingerprint_database(
        database_path,
        included_modes=INCLUDED_MODES,
        require_wifi=False,
    )
    wifi_model, wifi_grid, wifi_coordinates, ap_columns = load_wifi_model(
        database, wifi_checkpoint, device
    )
    magnetic_model, magnetic_window, magnetic_features = load_magnetic_model(
        magnetic_checkpoint, device
    )

    # The archived magnetic CNN was trained on the same 1 m environment grid as
    # the Wi-Fi heatmap, so preserve that geometry exactly when generating windows.
    magnetic_map = build_magnetic_map(database, wifi_grid, magnetic_features)
    corridor = build_corridor_graph(database)
    pool_nodes, pool = build_wifi_pool(database, ap_columns)
    return Environment(
        wifi_model=wifi_model,
        wifi_grid=wifi_grid,
        wifi_coordinates=wifi_coordinates,
        wifi_pool_nodes=pool_nodes,
        wifi_pool=pool,
        access_points=ap_columns,
        magnetic_model=magnetic_model,
        magnetic_window=magnetic_window,
        magnetic_map=magnetic_map,
        corridor=corridor,
    )


# ---------------------------------------------------------------------------
# Synthetic trajectories and sensor generation
# ---------------------------------------------------------------------------
def sample_path(graph: CorridorGraph, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        source, destination = rng.choice(graph.component_nodes, 2, replace=False)
        distances, predecessors = dijkstra(
            graph.adjacency,
            directed=False,
            indices=int(source),
            return_predecessors=True,
        )
        if not np.isfinite(distances[destination]) or distances[destination] < MINIMUM_PATH_M:
            continue
        path = [int(destination)]
        while path[-1] != int(source):
            previous = int(predecessors[path[-1]])
            if previous < 0:
                break
            path.append(previous)
        if path[-1] == int(source):
            path.reverse()
            return graph.coordinates[np.asarray(path)]
    return None


def synthesize_walk(path: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    segment = np.diff(path, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    speed = float(rng.uniform(SPEED_MIN_MPS, SPEED_MAX_MPS))
    step_frequency = float(rng.uniform(STEP_FREQ_MIN_HZ, STEP_FREQ_MAX_HZ))
    frames = int((total / speed) * SAMPLING_HZ)
    if frames < 120:
        return None

    distance = np.linspace(0.0, total, frames)
    x = np.interp(distance, cumulative, path[:, 0])
    y = np.interp(distance, cumulative, path[:, 1])
    truth = np.column_stack((x, y))

    true_heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    true_heading = np.convolve(true_heading, np.ones(7) / 7.0, mode="same")
    drift = np.cumsum(rng.normal(0.0, HEADING_DRIFT_STD_RAD, frames))
    measured_heading = true_heading + drift + rng.normal(0.0, HEADING_WHITE_NOISE_RAD, frames)

    time = np.arange(frames) / SAMPLING_HZ
    acceleration_magnitude = (
        9.81
        + rng.uniform(0.8, 1.3) * np.sin(2 * np.pi * step_frequency * time)
        + rng.normal(0.0, 0.3, frames)
    )
    return truth, acceleration_magnitude, measured_heading


def bilinear_magnetic_sample(magnetic_map: MagneticMap, positions: np.ndarray) -> np.ndarray:
    grid = magnetic_map.grid
    ix = np.clip((positions[:, 0] - grid.x0) / grid.cell, 0, grid.nx - 1.001)
    iy = np.clip((positions[:, 1] - grid.y0) / grid.cell, 0, grid.ny - 1.001)
    x0 = np.floor(ix).astype(int)
    y0 = np.floor(iy).astype(int)
    x1 = np.minimum(x0 + 1, grid.nx - 1)
    y1 = np.minimum(y0 + 1, grid.ny - 1)
    fx = ix - x0
    fy = iy - y0
    values = magnetic_map.values
    v00 = values[:, x0, y0].T
    v10 = values[:, x1, y0].T
    v01 = values[:, x0, y1].T
    v11 = values[:, x1, y1].T
    return (
        v00 * ((1 - fx) * (1 - fy))[:, None]
        + v10 * (fx * (1 - fy))[:, None]
        + v01 * ((1 - fx) * fy)[:, None]
        + v11 * (fx * fy)[:, None]
    )


def wifi_position_fix(env: Environment, scan: np.ndarray, device: torch.device) -> np.ndarray:
    encoded = encode_wifi(scan[None, :])
    with torch.no_grad():
        logits = env.wifi_model(torch.from_numpy(encoded).to(device))
        position = soft_argmax(logits, env.wifi_coordinates)
    return position[0].detach().cpu().numpy()


def magnetic_fixes_for_bins(
    env: Environment,
    magnetic_frames: np.ndarray,
    endpoints: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixes = np.zeros((len(endpoints), 2), dtype=np.float32)
    log_variance = np.zeros((len(endpoints), 1), dtype=np.float32)
    mask = np.zeros((len(endpoints), 1), dtype=np.float32)

    windows: list[np.ndarray] = []
    target_bins: list[int] = []
    for bin_index, frame_index in enumerate(endpoints):
        end = int(frame_index) + 1
        if end < env.magnetic_window:
            continue
        windows.append(magnetic_frames[end - env.magnetic_window : end])
        target_bins.append(bin_index)

    if windows:
        batch = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(device)
        with torch.no_grad():
            predicted, predicted_log_variance = env.magnetic_model(batch)
        predicted = predicted.cpu().numpy()
        predicted_log_variance = predicted_log_variance.cpu().numpy()
        for row, bin_index in enumerate(target_bins):
            fixes[bin_index] = predicted[row]
            log_variance[bin_index] = predicted_log_variance[row]
            mask[bin_index] = 1.0

    return fixes, log_variance, mask


def build_sequence(
    walk: tuple[np.ndarray, np.ndarray, np.ndarray],
    env: Environment,
    wifi_tree: cKDTree,
    rng: np.random.Generator,
    device: torch.device,
    *,
    wifi_period_s: float,
    ap_dropout: float,
    bins: int,
) -> tuple[np.ndarray, ...]:
    truth, acceleration_magnitude, heading = walk
    frames = len(truth)
    wifi_stride = max(1, int(round(wifi_period_s * SAMPLING_HZ)))

    # PDR motion at raw IMU rate.
    detector = StepDetector()
    controls = np.zeros((frames, 2), dtype=np.float32)
    for index in range(frames):
        if detector.update(float(acceleration_magnitude[index])):
            controls[index] = STEP_LENGTH_M * np.array(
                [math.cos(heading[index]), math.sin(heading[index])], dtype=np.float32
            )

    # Sparse Wi-Fi heatmap fixes from real stored scan vectors near the true position.
    node_keys = [tuple(node) for node in env.wifi_pool_nodes]
    wifi_fixes: dict[int, np.ndarray] = {}
    for index in range(0, frames, wifi_stride):
        _, nearest = wifi_tree.query(truth[index])
        scans = env.wifi_pool[node_keys[int(nearest)]]
        scan = scans[int(rng.integers(len(scans)))].copy()
        if ap_dropout > 0:
            scan[rng.random(len(scan)) < ap_dropout] = -100.0
        wifi_fixes[index] = wifi_position_fix(env, scan, device)

    # Rotation-invariant magnetic sequence sampled from the surveyed 4-channel map.
    magnetic_clean = bilinear_magnetic_sample(env.magnetic_map, truth)
    magnetic_frames = magnetic_clean + rng.normal(
        0.0,
        env.magnetic_map.noise_std,
        size=magnetic_clean.shape,
    )

    edges = np.linspace(0, frames, bins + 1).astype(int)
    endpoints = np.maximum(edges[1:] - 1, 0)
    magnetic_fix, magnetic_logvar, magnetic_mask = magnetic_fixes_for_bins(
        env, magnetic_frames.astype(np.float32), endpoints, device
    )

    motion = np.zeros((bins, 2), dtype=np.float32)
    wifi_fix = np.zeros((bins, 2), dtype=np.float32)
    wifi_mask = np.zeros((bins, 1), dtype=np.float32)
    target = np.zeros((bins, 2), dtype=np.float32)
    start = truth[0].astype(np.float32)

    first_wifi_index = min(wifi_fixes)
    last_wifi = wifi_fixes[first_wifi_index].astype(np.float32)
    for bin_index in range(bins):
        left, right = int(edges[bin_index]), int(edges[bin_index + 1])
        motion[bin_index] = controls[left:right].sum(axis=0)
        fixes_in_bin = [fix for frame, fix in wifi_fixes.items() if left <= frame < right]
        if fixes_in_bin:
            last_wifi = np.mean(fixes_in_bin, axis=0).astype(np.float32)
            wifi_mask[bin_index] = 1.0
        wifi_fix[bin_index] = last_wifi
        target[bin_index] = truth[max(left, right - 1)]

    return (
        motion,
        wifi_fix,
        wifi_mask,
        magnetic_fix,
        magnetic_logvar,
        magnetic_mask,
        target,
        start,
    )


def make_dataset(
    walks: int,
    seed: int,
    env: Environment,
    device: torch.device,
    *,
    wifi_period_s: float,
    ap_dropout: float,
    bins: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    wifi_tree = cKDTree(env.wifi_pool_nodes)
    rows: list[tuple[np.ndarray, ...]] = []
    attempts = 0
    while len(rows) < walks and attempts < walks * 5:
        attempts += 1
        path = sample_path(env.corridor, rng)
        if path is None:
            continue
        walk = synthesize_walk(path, rng)
        if walk is None:
            continue
        rows.append(
            build_sequence(
                walk,
                env,
                wifi_tree,
                np.random.default_rng(seed * 10007 + attempts),
                device,
                wifi_period_s=wifi_period_s,
                ap_dropout=ap_dropout,
                bins=bins,
            )
        )
        if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == walks:
            print(f"    generated {len(rows)}/{walks} walks")

    if len(rows) != walks:
        raise RuntimeError(f"generated only {len(rows)}/{walks} requested walks")
    return tuple(np.stack([row[column] for row in rows]) for column in range(8))


# ---------------------------------------------------------------------------
# Fusion training/evaluation
# ---------------------------------------------------------------------------
def _tensor(array: np.ndarray, device: torch.device) -> Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def prepare_tensors(data: tuple[np.ndarray, ...], device: torch.device) -> tuple[Tensor, ...]:
    motion, wifi, wifi_mask, mag, mag_logvar, mag_mask, target, start = data
    start_expanded = start[:, None, :]
    return (
        _tensor(motion, device),
        _tensor(wifi - start_expanded, device),
        _tensor(wifi_mask, device),
        _tensor(mag - start_expanded, device),
        _tensor(mag_logvar, device),
        _tensor(mag_mask, device),
        _tensor(target - start_expanded, device),
    )


def train_filter(
    model: nn.Module,
    data: tuple[np.ndarray, ...],
    device: torch.device,
    *,
    epochs: int,
    uses_magnetic: bool,
) -> tuple[nn.Module, list[float]]:
    tensors = prepare_tensors(data, device)
    motion, wifi, wifi_mask, mag, mag_logvar, mag_mask, target = tensors
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FUSION_LEARNING_RATE,
        weight_decay=FUSION_WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()
    indices = np.arange(len(motion))
    history: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        total = 0.0
        batches = 0
        for left in range(0, len(indices), FUSION_BATCH_SIZE):
            batch = indices[left : left + FUSION_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            if uses_magnetic:
                output = model(
                    motion[batch],
                    wifi[batch],
                    wifi_mask[batch],
                    mag[batch],
                    mag_logvar[batch],
                    mag_mask[batch],
                )
            else:
                output = model(motion[batch], wifi[batch], wifi_mask[batch])
            loss = loss_fn(output, target[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1

        epoch_loss = total / max(batches, 1)
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch + 1 == epochs:
            print(f"      epoch {epoch + 1:03d}/{epochs}: train_mse={epoch_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_filter(
    model: nn.Module,
    data: tuple[np.ndarray, ...],
    device: torch.device,
    *,
    uses_magnetic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = prepare_tensors(data, device)
    motion, wifi, wifi_mask, mag, mag_logvar, mag_mask, target = tensors
    model.eval()
    with torch.no_grad():
        if uses_magnetic:
            output = model(motion, wifi, wifi_mask, mag, mag_logvar, mag_mask)
        else:
            output = model(motion, wifi, wifi_mask)
    errors = torch.linalg.norm(output - target, dim=2).cpu().numpy()
    return errors.mean(axis=1), output.cpu().numpy()


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "walks": int(len(values)),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "ci95_half_width_m": float(ci),
    }


def magnetic_measurement_summary(data: tuple[np.ndarray, ...]) -> dict[str, float]:
    _, _, _, mag, _, mag_mask, target, _ = data
    mask = mag_mask[..., 0].astype(bool)
    errors = np.linalg.norm(mag - target, axis=2)
    available = errors[mask]
    return {
        "availability_fraction": float(mask.mean()),
        "mean_error_m": float(available.mean()) if len(available) else float("nan"),
        "median_error_m": float(np.median(available)) if len(available) else float("nan"),
    }


def plot_cdf(results: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5), squeeze=False)
    for axis, (regime, arrays) in zip(axes[0], results.items()):
        for label, values in arrays.items():
            sorted_values = np.sort(values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axis.plot(sorted_values, cdf, linewidth=2, label=f"{label} ({values.mean():.2f} m)")
        axis.set_title(regime)
        axis.set_xlabel("Per-walk mean error (m)")
        axis.set_ylabel("CDF")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    env: Environment,
    device: torch.device,
    *,
    train_walks: int,
    test_walks: int,
    fusion_epochs: int,
    bins: int,
    output: Path,
    regimes_to_run: set[str],
) -> dict[str, object]:
    regimes = {
        "full": ("Full Wi-Fi (1 Hz)", 1.0, 0.0),
        "degraded": ("Degraded Wi-Fi (5 s, 40% AP drop)", 5.0, 0.4),
    }
    report: dict[str, object] = {}
    cdf_data: dict[str, dict[str, np.ndarray]] = {}

    for regime_key, (regime_name, wifi_period, dropout) in regimes.items():
        if regime_key not in regimes_to_run:
            continue
        print(f"\n{'=' * 72}\n{regime_name}\n{'=' * 72}")
        print("  generating training set")
        training = make_dataset(
            train_walks,
            seed=1,
            env=env,
            device=device,
            wifi_period_s=wifi_period,
            ap_dropout=dropout,
            bins=bins,
        )
        print("  generating test set")
        testing = make_dataset(
            test_walks,
            seed=2,
            env=env,
            device=device,
            wifi_period_s=wifi_period,
            ap_dropout=dropout,
            bins=bins,
        )
        print("  magnetic CNN measurement quality:", magnetic_measurement_summary(testing))

        torch.manual_seed(SEED)
        baseline = WiFiOnlyKalmanNet(hidden_size=FUSION_HIDDEN_SIZE).to(device)
        print("  training Wi-Fi-only KalmanNet")
        baseline, baseline_history = train_filter(
            baseline, training, device, epochs=fusion_epochs, uses_magnetic=False
        )
        baseline_errors, _ = evaluate_filter(
            baseline, testing, device, uses_magnetic=False
        )

        torch.manual_seed(SEED)
        dual = CNNMagneticDualKalmanNet(hidden_size=FUSION_HIDDEN_SIZE).to(device)
        print("  training CNN-output DualKalmanNet")
        dual, dual_history = train_filter(
            dual, training, device, epochs=fusion_epochs, uses_magnetic=True
        )
        dual_errors, _ = evaluate_filter(dual, testing, device, uses_magnetic=True)

        baseline_summary = summarize(baseline_errors)
        dual_summary = summarize(dual_errors)
        improvement = 100.0 * (
            float(baseline_summary["mean_m"]) - float(dual_summary["mean_m"])
        ) / float(baseline_summary["mean_m"])
        print(
            f"  Wi-Fi-only: {baseline_summary['mean_m']:.3f} +/- "
            f"{baseline_summary['ci95_half_width_m']:.3f} m "
            f"(median {baseline_summary['median_m']:.3f})"
        )
        print(
            f"  CNN Dual:   {dual_summary['mean_m']:.3f} +/- "
            f"{dual_summary['ci95_half_width_m']:.3f} m "
            f"(median {dual_summary['median_m']:.3f})"
        )
        print(f"  relative improvement from magnetic CNN: {improvement:+.1f}%")

        report[regime_key] = {
            "name": regime_name,
            "wifi_period_s": wifi_period,
            "ap_dropout": dropout,
            "magnetic_measurement": magnetic_measurement_summary(testing),
            "wifi_only_kalmannet": baseline_summary,
            "cnn_dual_kalmannet": dual_summary,
            "relative_improvement_percent": float(improvement),
            "baseline_training_loss": baseline_history,
            "dual_training_loss": dual_history,
        }
        cdf_data[regime_name] = {
            "Wi-Fi-only KalmanNet": baseline_errors,
            "CNN DualKalmanNet": dual_errors,
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if cdf_data:
        plot_cdf(cdf_data, output / "cdf.png")
    return report


def dry_run(env: Environment, device: torch.device, bins: int) -> None:
    print("Running one-walk end-to-end preflight...")
    data = make_dataset(
        1,
        seed=99,
        env=env,
        device=device,
        wifi_period_s=5.0,
        ap_dropout=0.4,
        bins=bins,
    )
    tensors = prepare_tensors(data, device)
    motion, wifi, wifi_mask, mag, mag_logvar, mag_mask, target = tensors
    dual = CNNMagneticDualKalmanNet().to(device)
    baseline = WiFiOnlyKalmanNet().to(device)
    with torch.no_grad():
        dual_out = dual(motion, wifi, wifi_mask, mag, mag_logvar, mag_mask)
        baseline_out = baseline(motion, wifi, wifi_mask)
    if dual_out.shape != target.shape or baseline_out.shape != target.shape:
        raise RuntimeError("fusion output shape does not match target")
    if not torch.isfinite(dual_out).all():
        raise RuntimeError("dual fusion produced non-finite values")
    print(
        json.dumps(
            {
                "dry_run": True,
                "motion_shape": list(motion.shape),
                "magnetic_window_frames": env.magnetic_window,
                "magnetic_measurement": magnetic_measurement_summary(data),
                "wifi_measurement_fraction": float(data[2].mean()),
                "dual_output_shape": list(dual_out.shape),
                "baseline_output_shape": list(baseline_out.shape),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--wifi-checkpoint", type=Path, default=DEFAULT_WIFI_CHECKPOINT)
    parser.add_argument("--mag-checkpoint", type=Path, default=DEFAULT_MAG_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-walks", type=int, default=DEFAULT_TRAIN_WALKS)
    parser.add_argument("--test-walks", type=int, default=DEFAULT_TEST_WALKS)
    parser.add_argument("--fusion-epochs", type=int, default=DEFAULT_FUSION_EPOCHS)
    parser.add_argument("--bins", type=int, default=DEFAULT_T_BINS)
    parser.add_argument("--regime", choices=["both", "full", "degraded"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="small real training run (12 train walks, 4 test walks, 8 epochs, 80 bins)",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if args.smoke:
        args.train_walks = 12
        args.test_walks = 4
        args.fusion_epochs = 8
        args.bins = 80

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"device: {device}")
    env = setup_environment(
        args.database.resolve(),
        args.wifi_checkpoint.resolve(),
        args.mag_checkpoint.resolve(),
        device,
    )
    print(
        f"environment: APs={len(env.access_points)}, corridor_nodes={len(env.corridor.component_nodes)}, "
        f"mag_window={env.magnetic_window}, mag_features={env.magnetic_map.features}"
    )

    if args.dry_run:
        dry_run(env, device, args.bins)
        return 0

    regimes = {"full", "degraded"} if args.regime == "both" else {args.regime}
    report = run_experiment(
        env,
        device,
        train_walks=args.train_walks,
        test_walks=args.test_walks,
        fusion_epochs=args.fusion_epochs,
        bins=args.bins,
        output=args.output.resolve(),
        regimes_to_run=regimes,
    )
    print("\nFinal metrics")
    print(json.dumps(report, indent=2))
    print(f"saved: {args.output.resolve() / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
