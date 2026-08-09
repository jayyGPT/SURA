"""End-to-end standalone training for the magnetic sequence CNN."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sura.config import load_yaml
from sura.data.fingerprint import load_fingerprint_database
from sura.data.magnetic_windows import (
    build_corridor_graph,
    build_magnetic_map,
    generate_magnetic_windows,
)
from sura.data.paths import configured_data_path, experiment_runs_root, fingerprint_database
from sura.evaluation.metrics import position_errors, summarize_errors
from sura.models.magnetic_sequence_cnn import MagSequenceMatcher, heteroscedastic_nll

from .common import (
    create_run_directory,
    current_git_commit,
    default_run_name,
    resolve_device,
    seed_everything,
    write_json,
)


def _train_window(
    *,
    window_frames: int,
    magnetic_map,
    graph,
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    run_directory: Path,
    device: torch.device,
    epochs_override: int | None,
) -> dict[str, Any]:
    seed = int(model_config.get("seed", 42))
    seed_everything(seed)
    features = [str(value) for value in model_config.get("features", [])]
    trajectory = model_config.get("trajectory", {})
    sampling_hz = float(
        trajectory.get("sampling_hz", dataset_config.get("sampling_hz", 16.7))
    )
    common_generation = {
        "graph": graph,
        "magnetic_map": magnetic_map,
        "window_frames": window_frames,
        "minimum_path_m": float(trajectory.get("minimum_path_m", 30.0)),
        "sampling_hz": sampling_hz,
        "speed_min_mps": float(trajectory.get("speed_min_mps", 1.0)),
        "speed_max_mps": float(trajectory.get("speed_max_mps", 1.35)),
    }
    train_x, train_y = generate_magnetic_windows(
        **common_generation,
        walks=int(trajectory.get("train_walks", 300)),
        seed=seed,
        stride_frames=int(trajectory.get("train_stride_frames", 5)),
    )
    test_x, test_y = generate_magnetic_windows(
        **common_generation,
        walks=int(trajectory.get("test_walks", 60)),
        seed=int(trajectory.get("test_seed", seed + 158)),
        stride_frames=int(trajectory.get("test_stride_frames", 10)),
    )

    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(model_config.get("batch_size", 128)),
        shuffle=True,
        generator=loader_generator,
        num_workers=int(model_config.get("num_workers", 0)),
    )
    model = MagSequenceMatcher(
        in_channels=len(features),
        hidden_size=int(model_config.get("hidden_size", 128)),
        position_dropout=float(model_config.get("position_dropout", 0.2)),
    ).to(device)
    optimizer_config = model_config.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 1e-3)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
    )
    scheduler_config = model_config.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(scheduler_config.get("patience", 8)),
        factor=float(scheduler_config.get("factor", 0.5)),
    )
    epochs = int(epochs_override or model_config.get("epochs", 60))
    minimum_variance = float(model_config.get("minimum_variance", 0.01))
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
                minimum_variance=minimum_variance,
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
    summary = summarize_errors(position_errors(prediction, test_y))

    output = run_directory / f"window_{window_frames}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "window_frames": window_frames,
            "features": features,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "metrics": summary.as_dict(),
        },
        checkpoint_path,
    )
    np.savez_compressed(
        output / "predictions.npz",
        predicted=prediction,
        truth=test_y,
        log_variance=log_variance_array,
    )
    write_json(output / "history.json", {"epochs": history})
    return {
        "window_frames": window_frames,
        "train_windows": int(len(train_x)),
        "test_windows": int(len(test_x)),
        "checkpoint": str(checkpoint_path),
        "metrics": summary.as_dict(),
    }


def train_magnetic_sequence(
    *,
    model_config_path: str | Path,
    dataset_config_path: str | Path,
    data_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    device: str = "auto",
    run_name: str | None = None,
    epochs: int | None = None,
    sweep: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train the configured window or perform the configured window-size sweep."""
    model_config = load_yaml(model_config_path)
    dataset_config = load_yaml(dataset_config_path)
    included_modes = [str(value) for value in dataset_config.get("included_modes", [])]
    building = str(dataset_config.get("building", "it_engineering"))
    configured_database = dataset_config.get("fingerprint_db")
    database_path = (
        configured_data_path(str(configured_database), data_directory)
        if configured_database
        else fingerprint_database(building, data_directory)
    )
    database = load_fingerprint_database(
        database_path,
        included_modes=included_modes or None,
        require_wifi=False,
    )
    features = [str(value) for value in model_config.get("features", [])]
    if not features:
        raise ValueError("magnetic configuration must define at least one feature")
    window_values = (
        [int(value) for value in model_config.get("window_sweep_frames", [])]
        if sweep
        else [int(model_config.get("window_frames", 84))]
    )
    if not window_values:
        raise ValueError("no magnetic window sizes were configured")

    magnetic_map = build_magnetic_map(
        database,
        features=features,
        grid_cell_m=float(model_config.get("grid_cell_m", 1.0)),
    )
    trajectory = model_config.get("trajectory", {})
    graph = build_corridor_graph(
        database,
        epsilon_m=float(trajectory.get("epsilon_m", 1.6)),
    )
    preflight = {
        "model": "magnetic_sequence_cnn",
        "database": database.summary(),
        "features": features,
        "window_frames": window_values,
        "magnetic_map_shape": list(magnetic_map.values.shape),
        "corridor_nodes": int(len(graph.component_nodes)),
        "device": str(resolve_device(device)),
        "epochs": int(epochs or model_config.get("epochs", 60)),
    }
    if dry_run:
        return {"dry_run": True, **preflight}

    base = output_directory or experiment_runs_root()
    name = run_name or default_run_name("magnetic")
    run_directory = create_run_directory(base, "magnetic_sequence", name)
    resolved_device = resolve_device(device)
    results = [
        _train_window(
            window_frames=window,
            magnetic_map=magnetic_map,
            graph=graph,
            model_config=model_config,
            dataset_config=dataset_config,
            run_directory=run_directory,
            device=resolved_device,
            epochs_override=epochs,
        )
        for window in window_values
    ]
    best = min(results, key=lambda item: item["metrics"]["mean_m"])
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
    return payload
