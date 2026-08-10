#!/usr/bin/env python3
"""Train the standalone magnetic sequence CNN."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fingerprint import load_fingerprint_database
from models.magnetic_sequence_cnn import MagSequenceMatcher, heteroscedastic_nll
from models.wifi_heatmap import Grid
from common import (
    create_run_directory,
    current_git_commit,
    default_run_name,
    position_errors,
    resolve_device,
    save_error_cdf,
    save_training_curve,
    seed_everything,
    summarize_errors,
    write_json,
)


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


def build_magnetic_map(database, features: list[str], grid_cell_m: float = 1.0) -> MagneticMap:
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

        centered = f"_centered_{feature}"
        valid[centered] = valid[column] - valid.groupby("phone")[column].transform("mean")
        valid["x_round"] = valid["x"].round(1)
        valid["y_round"] = valid["y"].round(1)
        node_mean = valid.groupby(["x_round", "y_round"])[centered].mean()
        coordinates = np.asarray(list(node_mean.index), dtype=float)
        values = node_mean.to_numpy(dtype=float)

        if len(coordinates) < 3:
            raise ValueError("at least three unique nodes are required for magnetic mapping")

        linear = griddata(coordinates, values, grid.coords, method="linear")
        nearest = griddata(coordinates, values, grid.coords, method="nearest")
        interpolated = np.where(np.isnan(linear), nearest, linear).reshape(grid.nx, grid.ny)
        maps.append(interpolated.astype(np.float32))

        node_std = valid.groupby(["x_round", "y_round"])[centered].std()
        finite_std = node_std[np.isfinite(node_std.to_numpy(dtype=float))]
        estimate = float(finite_std.median()) if not finite_std.empty else float("nan")
        if not np.isfinite(estimate) or estimate <= 0:
            global_std = float(valid[centered].std())
            estimate = max(global_std * 0.05, 1e-3) if np.isfinite(global_std) else 1.0
        noise.append(estimate)

    return MagneticMap(
        values=np.stack(maps, axis=0),
        noise_std=np.asarray(noise, dtype=np.float32),
        grid=grid,
        features=tuple(features),
    )


def build_corridor_graph(database, epsilon_m: float = 1.6) -> CorridorGraph:
    coordinates = (
        database.frame[["x", "y"]].round(1).drop_duplicates().to_numpy(dtype=float)
    )
    tree = cKDTree(coordinates)
    pairs = list(tree.query_pairs(epsilon_m))
    if not pairs:
        raise ValueError("corridor graph has no edges")

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for left, right in pairs:
        distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
        rows.extend((left, right))
        columns.extend((right, left))
        weights.extend((distance, distance))

    adjacency = csr_matrix(
        (weights, (rows, columns)),
        shape=(len(coordinates), len(coordinates)),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    counts = np.bincount(labels, minlength=component_count)
    component_nodes = np.flatnonzero(labels == int(np.argmax(counts)))
    return CorridorGraph(coordinates, adjacency, component_nodes)


def sample_corridor_path(
    graph: CorridorGraph,
    generator: np.random.Generator,
    minimum_length_m: float,
    attempts: int = 100,
) -> np.ndarray:
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


def interpolate_path(path: np.ndarray, sampling_hz: float, speed_mps: float) -> np.ndarray:
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    length = float(cumulative[-1])
    frames = max(2, int(np.ceil(length / speed_mps * sampling_hz)) + 1)
    distance = np.linspace(0.0, length, frames)
    x = np.interp(distance, cumulative, path[:, 0])
    y = np.interp(distance, cumulative, path[:, 1])
    return np.column_stack((x, y))


def bilinear_sample(magnetic_map: MagneticMap, positions: np.ndarray) -> np.ndarray:
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
    generator = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    successful_walks = 0
    attempts = 0

    while successful_walks < walks and attempts < walks * 6:
        attempts += 1
        try:
            path = sample_corridor_path(graph, generator, minimum_path_m)
        except RuntimeError:
            continue

        speed = float(generator.uniform(speed_min_mps, speed_max_mps))
        positions = interpolate_path(path, sampling_hz, speed)
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

# ---------------------------------------------------------------------------
# Configuration
# Edit these values directly for experiments. There is no separate config file.
# ---------------------------------------------------------------------------
SEED = 42
FEATURES = ["magN", "magV", "magH", "dip"]
GRID_CELL_M = 1.0
WINDOW_FRAMES = 84
WINDOW_SWEEP = [50, 84, 134, 167]
HIDDEN_SIZE = 128
POSITION_DROPOUT = 0.2
MINIMUM_VARIANCE = 0.01
BATCH_SIZE = 128
NUM_WORKERS = 0
DEFAULT_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 8
SCHEDULER_FACTOR = 0.5

EPSILON_M = 1.6
MINIMUM_PATH_M = 30.0
SAMPLING_HZ = 16.7
SPEED_MIN_MPS = 1.0
SPEED_MAX_MPS = 1.35
TRAIN_WALKS = 300
TEST_WALKS = 60
TEST_SEED = 200
TRAIN_STRIDE_FRAMES = 5
TEST_STRIDE_FRAMES = 10

INCLUDED_MODES = ["Navigation", "Call listening", "Swinging"]

DEFAULT_DATABASE = REPO_ROOT / "data" / "processed" / "fingerprint_db" / "it_engineering"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "runs"


def config_dict() -> dict[str, object]:
    return {
        "seed": SEED,
        "features": FEATURES,
        "grid_cell_m": GRID_CELL_M,
        "window_frames": WINDOW_FRAMES,
        "window_sweep_frames": WINDOW_SWEEP,
        "hidden_size": HIDDEN_SIZE,
        "position_dropout": POSITION_DROPOUT,
        "minimum_variance": MINIMUM_VARIANCE,
        "batch_size": BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,
        "epsilon_m": EPSILON_M,
        "minimum_path_m": MINIMUM_PATH_M,
        "sampling_hz": SAMPLING_HZ,
        "speed_min_mps": SPEED_MIN_MPS,
        "speed_max_mps": SPEED_MAX_MPS,
        "train_walks": TRAIN_WALKS,
        "test_walks": TEST_WALKS,
        "test_seed": TEST_SEED,
        "train_stride_frames": TRAIN_STRIDE_FRAMES,
        "test_stride_frames": TEST_STRIDE_FRAMES,
        "included_modes": INCLUDED_MODES,
    }


def train_window(
    window_frames: int,
    magnetic_map,
    graph,
    run_directory: Path,
    device: torch.device,
    epochs: int,
) -> dict[str, object]:
    seed_everything(SEED)

    common_generation = {
        "graph": graph,
        "magnetic_map": magnetic_map,
        "window_frames": window_frames,
        "minimum_path_m": MINIMUM_PATH_M,
        "sampling_hz": SAMPLING_HZ,
        "speed_min_mps": SPEED_MIN_MPS,
        "speed_max_mps": SPEED_MAX_MPS,
    }
    train_x, train_y = generate_magnetic_windows(
        **common_generation,
        walks=TRAIN_WALKS,
        seed=SEED,
        stride_frames=TRAIN_STRIDE_FRAMES,
    )
    test_x, test_y = generate_magnetic_windows(
        **common_generation,
        walks=TEST_WALKS,
        seed=TEST_SEED,
        stride_frames=TEST_STRIDE_FRAMES,
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=NUM_WORKERS,
    )
    model = MagSequenceMatcher(
        in_channels=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        position_dropout=POSITION_DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )
    test_x_tensor = torch.from_numpy(test_x).to(device)

    best_mean = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, log_variance = model(batch_x)
            loss = heteroscedastic_nll(
                predicted,
                log_variance,
                batch_y,
                minimum_variance=MINIMUM_VARIANCE,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

        model.eval()
        with torch.no_grad():
            predicted, _ = model(test_x_tensor)
        summary = summarize_errors(position_errors(predicted.cpu().numpy(), test_y))
        scheduler.step(summary.mean_m)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": running_loss / max(batches, 1),
                "test_mean_error_m": summary.mean_m,
            }
        )
        print(
            f"[window={window_frames}] epoch {epoch + 1:03d}/{epochs} "
            f"loss={history[-1]['training_loss']:.4f} "
            f"mean_error={summary.mean_m:.3f} m"
        )

        if summary.mean_m < best_mean:
            best_mean = summary.mean_m
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training did not produce a model state")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted, log_variance = model(test_x_tensor)

    prediction = predicted.cpu().numpy()
    log_variance_array = log_variance.cpu().numpy()
    errors = position_errors(prediction, test_y)
    summary = summarize_errors(errors)

    output = run_directory / f"window_{window_frames}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "window_frames": window_frames,
            "features": FEATURES,
            "config": config_dict(),
            "metrics": summary.as_dict(),
        },
        checkpoint,
    )
    np.savez_compressed(
        output / "predictions.npz",
        predicted=prediction,
        truth=test_y,
        log_variance=log_variance_array,
    )
    write_json(output / "history.json", {"epochs": history})
    write_json(output / "metrics.json", summary.as_dict())
    save_error_cdf(
        errors,
        output / "error_cdf.png",
        f"Magnetic CNN ({window_frames} frames)",
    )
    save_training_curve(
        history,
        output / "training_curve.png",
        f"Magnetic CNN ({window_frames} frames)",
    )

    return {
        "window_frames": window_frames,
        "train_windows": int(len(train_x)),
        "test_windows": int(len(test_x)),
        "checkpoint": str(checkpoint),
        "metrics": summary.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check data/model setup without training",
    )
    args = parser.parse_args()

    database = load_fingerprint_database(
        args.database,
        included_modes=INCLUDED_MODES,
        require_wifi=False,
    )
    windows = WINDOW_SWEEP if args.sweep else [WINDOW_FRAMES]
    magnetic_map = build_magnetic_map(
        database,
        features=FEATURES,
        grid_cell_m=GRID_CELL_M,
    )
    graph = build_corridor_graph(database, epsilon_m=EPSILON_M)
    device = resolve_device(args.device)

    preflight = {
        "model": "magnetic_sequence_cnn",
        "database": database.summary(),
        "features": FEATURES,
        "window_frames": windows,
        "magnetic_map_shape": list(magnetic_map.values.shape),
        "corridor_nodes": int(len(graph.component_nodes)),
        "device": str(device),
        "epochs": args.epochs,
        "config": config_dict(),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **preflight}, indent=2))
        return 0

    name = args.run_name or default_run_name("magnetic")
    run_directory = create_run_directory(args.output, "magnetic_sequence", name)
    results = [
        train_window(window, magnetic_map, graph, run_directory, device, args.epochs)
        for window in windows
    ]
    best = min(results, key=lambda item: float(item["metrics"]["mean_m"]))
    payload = {
        "run_name": name,
        "run_directory": str(run_directory),
        "git_commit": current_git_commit(),
        **preflight,
        "results": results,
        "best_window_frames": best["window_frames"],
        "best_checkpoint": best["checkpoint"],
    }
    write_json(run_directory / "run.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
