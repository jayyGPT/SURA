"""Synthetic, corridor-faithful windows for standalone magnetic CNN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from sura.models.wifi_heatmap import Grid

from .fingerprint import FingerprintDatabase


@dataclass(frozen=True)
class MagneticMap:
    values: np.ndarray
    noise_std: np.ndarray
    grid: Grid
    features: tuple[str, ...]


@dataclass(frozen=True)
class CorridorGraph:
    coordinates: np.ndarray
    adjacency: csr_matrix
    component_nodes: np.ndarray


def build_magnetic_map(
    database: FingerprintDatabase,
    *,
    features: Sequence[str],
    grid_cell_m: float = 1.0,
) -> MagneticMap:
    """Build per-feature device-centered maps over the surveyed grid."""
    frame = database.frame.copy()
    grid = Grid(
        frame["x"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
        cell=grid_cell_m,
    )
    maps: list[np.ndarray] = []
    noise: list[float] = []
    for feature in features:
        column = f"{feature}_mean"
        if column not in frame.columns:
            raise ValueError(f"fingerprint table is missing magnetic feature: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid = frame.dropna(subset=[column]).copy()
        if valid.empty:
            raise ValueError(f"magnetic feature has no finite values: {column}")

        centered_column = f"_centered_{feature}"
        valid[centered_column] = valid[column] - valid.groupby("phone")[column].transform(
            "mean"
        )
        valid["x_round"] = valid["x"].round(1)
        valid["y_round"] = valid["y"].round(1)
        node_mean = valid.groupby(["x_round", "y_round"])[centered_column].mean()
        coordinates = np.asarray(list(node_mean.index), dtype=float)
        values = node_mean.to_numpy(dtype=float)
        if len(coordinates) < 3:
            raise ValueError("at least three unique nodes are required for magnetic mapping")

        linear = griddata(coordinates, values, grid.coords, method="linear")
        nearest = griddata(coordinates, values, grid.coords, method="nearest")
        interpolated = np.where(np.isnan(linear), nearest, linear).reshape(grid.nx, grid.ny)
        maps.append(interpolated.astype(np.float32))

        node_std = valid.groupby(["x_round", "y_round"])[centered_column].std()
        finite_node_std = node_std[np.isfinite(node_std.to_numpy(dtype=float))]
        estimate = float(finite_node_std.median()) if not finite_node_std.empty else float("nan")
        if not np.isfinite(estimate) or estimate <= 0:
            global_std = float(valid[centered_column].std())
            estimate = max(global_std * 0.05, 1e-3) if np.isfinite(global_std) else 1.0
        noise.append(estimate)

    return MagneticMap(
        values=np.stack(maps, axis=0),
        noise_std=np.asarray(noise, dtype=np.float32),
        grid=grid,
        features=tuple(str(feature) for feature in features),
    )


def build_corridor_graph(
    database: FingerprintDatabase,
    *,
    epsilon_m: float = 1.6,
) -> CorridorGraph:
    """Connect surveyed nodes whose Euclidean separation is at most epsilon."""
    if epsilon_m <= 0:
        raise ValueError("epsilon_m must be positive")
    coordinates = (
        database.frame[["x", "y"]].round(1).drop_duplicates().to_numpy(dtype=float)
    )
    if len(coordinates) < 2:
        raise ValueError("at least two unique nodes are required")
    tree = cKDTree(coordinates)
    pairs = list(tree.query_pairs(epsilon_m))
    if not pairs:
        raise ValueError("epsilon graph has no edges")
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for left, right in pairs:
        distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        rows.extend((left, right))
        columns.extend((right, left))
        weights.extend((distance, distance))
    adjacency = csr_matrix((weights, (rows, columns)), shape=(len(coordinates), len(coordinates)))
    component_count, labels = connected_components(adjacency, directed=False)
    counts = np.bincount(labels, minlength=component_count)
    largest = int(np.argmax(counts))
    component_nodes = np.flatnonzero(labels == largest)
    if len(component_nodes) < 2:
        raise ValueError("largest corridor component contains fewer than two nodes")
    return CorridorGraph(coordinates, adjacency, component_nodes)


def sample_corridor_path(
    graph: CorridorGraph,
    generator: np.random.Generator,
    *,
    minimum_length_m: float = 30.0,
    attempts: int = 100,
) -> np.ndarray:
    """Sample a shortest path within the largest connected component."""
    nodes = graph.component_nodes
    for _ in range(attempts):
        start, end = generator.choice(nodes, size=2, replace=False)
        distances, predecessors = dijkstra(
            graph.adjacency,
            directed=False,
            indices=int(start),
            return_predecessors=True,
        )
        if not np.isfinite(distances[end]) or distances[end] < minimum_length_m:
            continue
        indices = [int(end)]
        current = int(end)
        while current != int(start):
            current = int(predecessors[current])
            if current < 0:
                break
            indices.append(current)
        if current == int(start):
            indices.reverse()
            return graph.coordinates[np.asarray(indices)]
    raise RuntimeError(
        f"could not sample a corridor path of at least {minimum_length_m:.1f} m"
    )


def interpolate_path(
    path: np.ndarray,
    *,
    sampling_hz: float,
    speed_mps: float,
) -> np.ndarray:
    """Sample a piecewise-linear corridor path at a constant speed."""
    if sampling_hz <= 0 or speed_mps <= 0:
        raise ValueError("sampling_hz and speed_mps must be positive")
    path = np.asarray(path, dtype=float)
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    length = float(cumulative[-1])
    if length <= 0:
        raise ValueError("path length must be positive")
    frames = max(2, int(np.ceil(length / speed_mps * sampling_hz)) + 1)
    distance = np.linspace(0.0, length, frames)
    x = np.interp(distance, cumulative, path[:, 0])
    y = np.interp(distance, cumulative, path[:, 1])
    return np.column_stack((x, y))


def bilinear_sample(magnetic_map: MagneticMap, positions: np.ndarray) -> np.ndarray:
    """Sample all magnetic channels at world coordinates."""
    positions = np.asarray(positions, dtype=float)
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


def generate_magnetic_windows(
    *,
    graph: CorridorGraph,
    magnetic_map: MagneticMap,
    walks: int,
    seed: int,
    window_frames: int,
    stride_frames: int,
    minimum_path_m: float,
    sampling_hz: float,
    speed_min_mps: float,
    speed_max_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate causal windows and final-frame position targets."""
    if walks <= 0 or window_frames <= 0 or stride_frames <= 0:
        raise ValueError("walks, window_frames, and stride_frames must be positive")
    generator = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    successful_walks = 0
    attempts = 0
    while successful_walks < walks and attempts < walks * 6:
        attempts += 1
        try:
            path = sample_corridor_path(
                graph,
                generator,
                minimum_length_m=minimum_path_m,
            )
        except RuntimeError:
            continue
        speed = float(generator.uniform(speed_min_mps, speed_max_mps))
        positions = interpolate_path(path, sampling_hz=sampling_hz, speed_mps=speed)
        if len(positions) <= window_frames:
            continue
        clean = bilinear_sample(magnetic_map, positions)
        observed = clean + generator.normal(
            0.0,
            magnetic_map.noise_std,
            size=clean.shape,
        )
        for end in range(window_frames, len(positions) + 1, stride_frames):
            windows.append(observed[end - window_frames : end])
            targets.append(positions[end - 1])
        successful_walks += 1

    if successful_walks < walks or not windows:
        raise RuntimeError(
            f"generated only {successful_walks}/{walks} requested magnetic walks"
        )
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )
