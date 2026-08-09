"""End-to-end training for the Wi-Fi probability-heatmap model."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sura.config import load_yaml
from sura.data.fingerprint import FingerprintDatabase, load_fingerprint_database
from sura.data.paths import configured_data_path, experiment_runs_root, fingerprint_database
from sura.evaluation.metrics import position_errors, summarize_errors
from sura.models.wifi_heatmap import (
    Grid,
    WifiHeatmapNet,
    encode_wifi,
    heatmap_covariance,
    kl_divergence_loss,
    soft_argmax,
)

from .common import (
    create_run_directory,
    current_git_commit,
    default_run_name,
    resolve_device,
    seed_everything,
    write_json,
)


def _split_indices(
    database: FingerprintDatabase,
    *,
    split: str,
    seed: int,
    test_fraction: float,
    held_out_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame = database.frame
    if split == "phone":
        test = frame["phone"].astype(str).eq(held_out_device).to_numpy()
        if not test.any():
            raise ValueError(f"held-out device has no visits: {held_out_device}")
    elif split == "random":
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        generator = np.random.default_rng(seed)
        order = generator.permutation(len(frame))
        count = min(len(frame) - 1, max(1, int(round(len(frame) * test_fraction))))
        test = np.zeros(len(frame), dtype=bool)
        test[order[:count]] = True
    else:
        raise ValueError("split must be 'random' or 'phone'")
    train = ~test
    if not train.any() or not test.any():
        raise ValueError("training and test partitions must both be non-empty")
    return np.flatnonzero(train), np.flatnonzero(test)


def _train_one_split(
    *,
    database: FingerprintDatabase,
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    split: str,
    run_directory: Path,
    device: torch.device,
    epochs_override: int | None,
) -> dict[str, Any]:
    seed = int(model_config.get("seed", 0))
    seed_everything(seed)
    frame = database.frame
    grid = Grid(
        frame["x"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
        cell=float(model_config.get("grid_cell_m", 1.0)),
    )
    sigma = float(model_config.get("gaussian_sigma_m", 2.0))
    access_points = list(database.access_point_columns)
    encoded = encode_wifi(
        frame[access_points].to_numpy(dtype=float),
        absent_floor=float(model_config.get("rss_absent_dbm", -100.0)),
        clip_min=float(model_config.get("rss_clip_min_dbm", -90.0)),
        clip_max=float(model_config.get("rss_clip_max_dbm", -30.0)),
    )
    targets = np.stack(
        [
            grid.gaussian_target(float(x), float(y), sigma=sigma)
            for x, y in frame[["x", "y"]].to_numpy(dtype=float)
        ]
    )
    positions = frame[["x", "y"]].to_numpy(dtype=np.float32)

    train_index, test_index = _split_indices(
        database,
        split=split,
        seed=seed,
        test_fraction=float(model_config.get("test_fraction", 0.2)),
        held_out_device=str(dataset_config.get("held_out_device", "S9+")),
    )

    batch_size = int(model_config.get("batch_size", 64))
    train_dataset = TensorDataset(
        torch.from_numpy(encoded[train_index]),
        torch.from_numpy(targets[train_index]),
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(model_config.get("num_workers", 0)),
        generator=loader_generator,
    )

    model = WifiHeatmapNet(
        len(access_points),
        grid.n_cells,
        hidden_size=int(model_config.get("hidden_size", 256)),
        dropout=float(model_config.get("dropout", 0.3)),
    ).to(device)
    optimizer_config = model_config.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 1e-3)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
    )
    epochs = int(epochs_override or model_config.get("epochs", 80))
    coordinates = torch.tensor(grid.coords, dtype=torch.float32, device=device)
    test_x = torch.from_numpy(encoded[test_index]).to(device)
    test_positions = positions[test_index]

    best_mean = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_x, batch_target in loader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = kl_divergence_loss(model(batch_x), batch_target)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

        model.eval()
        with torch.no_grad():
            prediction = soft_argmax(model(test_x), coordinates).cpu().numpy()
        summary = summarize_errors(position_errors(prediction, test_positions))
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
        logits = model(test_x)
        prediction = soft_argmax(logits, coordinates).cpu().numpy()
        covariance = heatmap_covariance(logits, coordinates).cpu().numpy()
    summary = summarize_errors(position_errors(prediction, test_positions))

    split_directory = run_directory / split
    split_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = split_directory / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "grid": asdict(grid.spec),
            "access_point_columns": access_points,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "split": split,
            "metrics": summary.as_dict(),
        },
        checkpoint_path,
    )
    np.savez_compressed(
        split_directory / "predictions.npz",
        predicted=prediction,
        truth=test_positions,
        covariance=covariance,
        test_indices=test_index,
    )
    write_json(split_directory / "history.json", {"epochs": history})

    return {
        "split": split,
        "train_visits": int(len(train_index)),
        "test_visits": int(len(test_index)),
        "checkpoint": str(checkpoint_path),
        "metrics": summary.as_dict(),
    }


def train_wifi_heatmap(
    *,
    model_config_path: str | Path,
    dataset_config_path: str | Path,
    data_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    split: str = "both",
    device: str = "auto",
    run_name: str | None = None,
    epochs: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train random-split, held-device, or both Wi-Fi heatmap evaluations."""
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
        require_wifi=True,
    )
    requested_splits = ["random", "phone"] if split == "both" else [split]
    preflight = {
        "model": "wifi_heatmap",
        "database": database.summary(),
        "splits": requested_splits,
        "device": str(resolve_device(device)),
        "epochs": int(epochs or model_config.get("epochs", 80)),
    }
    if dry_run:
        return {"dry_run": True, **preflight}

    base = output_directory or experiment_runs_root()
    name = run_name or default_run_name("wifi")
    run_directory = create_run_directory(base, "wifi_heatmap", name)
    resolved_device = resolve_device(device)
    results = [
        _train_one_split(
            database=database,
            model_config=model_config,
            dataset_config=dataset_config,
            split=current_split,
            run_directory=run_directory,
            device=resolved_device,
            epochs_override=epochs,
        )
        for current_split in requested_splits
    ]
    payload = {
        "run_name": name,
        "run_directory": str(run_directory),
        "git_commit": current_git_commit(),
        **preflight,
        "results": results,
    }
    write_json(run_directory / "run.json", payload)
    return payload
