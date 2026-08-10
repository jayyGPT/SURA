#!/usr/bin/env python3
"""Train the Wi-Fi probability-heatmap model."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fingerprint import FingerprintDatabase, load_fingerprint_database
from models.wifi_heatmap import (
    Grid,
    WifiHeatmapNet,
    encode_wifi,
    heatmap_covariance,
    kl_divergence_loss,
    soft_argmax,
)
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

# ---------------------------------------------------------------------------
# Configuration
# Edit these values directly for experiments. There is no separate config file.
# ---------------------------------------------------------------------------
SEED = 0
GRID_CELL_M = 1.0
GAUSSIAN_SIGMA_M = 2.0
RSS_ABSENT_DBM = -100.0
RSS_CLIP_MIN_DBM = -90.0
RSS_CLIP_MAX_DBM = -30.0
HIDDEN_SIZE = 256
DROPOUT = 0.3
BATCH_SIZE = 64
NUM_WORKERS = 0
DEFAULT_EPOCHS = 80
TEST_FRACTION = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HELD_OUT_DEVICE = "S9+"
INCLUDED_MODES = ["Navigation", "Call listening", "Swinging"]

DEFAULT_DATABASE = REPO_ROOT / "data" / "processed" / "fingerprint_db" / "it_engineering"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "runs"


def config_dict() -> dict[str, object]:
    return {
        "seed": SEED,
        "grid_cell_m": GRID_CELL_M,
        "gaussian_sigma_m": GAUSSIAN_SIGMA_M,
        "rss_absent_dbm": RSS_ABSENT_DBM,
        "rss_clip_min_dbm": RSS_CLIP_MIN_DBM,
        "rss_clip_max_dbm": RSS_CLIP_MAX_DBM,
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "test_fraction": TEST_FRACTION,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "held_out_device": HELD_OUT_DEVICE,
        "included_modes": INCLUDED_MODES,
    }


def split_indices(database: FingerprintDatabase, split: str) -> tuple[np.ndarray, np.ndarray]:
    frame = database.frame
    if split == "phone":
        test = frame["phone"].astype(str).eq(HELD_OUT_DEVICE).to_numpy()
        if not test.any():
            raise ValueError(f"held-out device has no visits: {HELD_OUT_DEVICE}")
    elif split == "random":
        generator = np.random.default_rng(SEED)
        order = generator.permutation(len(frame))
        count = min(len(frame) - 1, max(1, int(round(len(frame) * TEST_FRACTION))))
        test = np.zeros(len(frame), dtype=bool)
        test[order[:count]] = True
    else:
        raise ValueError("split must be 'random' or 'phone'")

    train = ~test
    if not train.any() or not test.any():
        raise ValueError("training and test partitions must both be non-empty")
    return np.flatnonzero(train), np.flatnonzero(test)


def train_one_split(
    database: FingerprintDatabase,
    split: str,
    run_directory: Path,
    device: torch.device,
    epochs: int,
) -> dict[str, object]:
    seed_everything(SEED)
    frame = database.frame

    grid = Grid(
        frame["x"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
        cell=GRID_CELL_M,
    )
    access_points = list(database.access_point_columns)
    encoded = encode_wifi(
        frame[access_points].to_numpy(dtype=float),
        absent_floor=RSS_ABSENT_DBM,
        clip_min=RSS_CLIP_MIN_DBM,
        clip_max=RSS_CLIP_MAX_DBM,
    )
    targets = np.stack(
        [
            grid.gaussian_target(float(x), float(y), sigma=GAUSSIAN_SIGMA_M)
            for x, y in frame[["x", "y"]].to_numpy(dtype=float)
        ]
    )
    positions = frame[["x", "y"]].to_numpy(dtype=np.float32)
    train_index, test_index = split_indices(database, split)

    train_dataset = TensorDataset(
        torch.from_numpy(encoded[train_index]),
        torch.from_numpy(targets[train_index]),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = WifiHeatmapNet(
        len(access_points),
        grid.n_cells,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

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
        print(
            f"[{split}] epoch {epoch + 1:03d}/{epochs} "
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
        logits = model(test_x)
        prediction = soft_argmax(logits, coordinates).cpu().numpy()
        covariance = heatmap_covariance(logits, coordinates).cpu().numpy()

    errors = position_errors(prediction, test_positions)
    summary = summarize_errors(errors)

    output = run_directory / split
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "grid": asdict(grid.spec),
            "access_point_columns": access_points,
            "config": config_dict(),
            "split": split,
            "metrics": summary.as_dict(),
        },
        checkpoint,
    )
    np.savez_compressed(
        output / "predictions.npz",
        predicted=prediction,
        truth=test_positions,
        covariance=covariance,
        test_indices=test_index,
    )
    write_json(output / "history.json", {"epochs": history})
    write_json(output / "metrics.json", summary.as_dict())
    save_error_cdf(errors, output / "error_cdf.png", f"Wi-Fi heatmap ({split})")
    save_training_curve(history, output / "training_curve.png", f"Wi-Fi heatmap ({split})")

    return {
        "split": split,
        "train_visits": int(len(train_index)),
        "test_visits": int(len(test_index)),
        "checkpoint": str(checkpoint),
        "metrics": summary.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("both", "random", "phone"), default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check data/model setup without training",
    )
    args = parser.parse_args()

    database = load_fingerprint_database(
        args.database,
        included_modes=INCLUDED_MODES,
        require_wifi=True,
    )
    requested_splits = ["random", "phone"] if args.split == "both" else [args.split]
    device = resolve_device(args.device)

    preflight = {
        "model": "wifi_heatmap",
        "database": database.summary(),
        "splits": requested_splits,
        "device": str(device),
        "epochs": args.epochs,
        "config": config_dict(),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **preflight}, indent=2))
        return 0

    name = args.run_name or default_run_name("wifi")
    run_directory = create_run_directory(args.output, "wifi_heatmap", name)
    results = [
        train_one_split(database, split, run_directory, device, args.epochs)
        for split in requested_splits
    ]
    payload = {
        "run_name": name,
        "run_directory": str(run_directory),
        "git_commit": current_git_commit(),
        **preflight,
        "results": results,
    }
    write_json(run_directory / "run.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
