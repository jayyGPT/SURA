"""Small helpers shared by the training scripts."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(value)


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def default_run_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def create_run_directory(base: str | Path, model: str, run_name: str) -> Path:
    path = Path(base).expanduser().resolve() / model / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


@dataclass(frozen=True)
class ErrorSummary:
    count: int
    mean_m: float
    median_m: float
    p90_m: float
    maximum_m: float
    ci95_half_width_m: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def position_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if predicted.shape != truth.shape or predicted.ndim < 2 or predicted.shape[-1] != 2:
        raise ValueError("predicted and truth must have matching [...,2] shapes")
    return np.linalg.norm(predicted - truth, axis=-1)


def summarize_errors(errors: np.ndarray) -> ErrorSummary:
    values = np.asarray(errors, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("errors contains no finite values")
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return ErrorSummary(
        count=len(values),
        mean_m=float(values.mean()),
        median_m=float(np.median(values)),
        p90_m=float(np.percentile(values, 90)),
        maximum_m=float(values.max()),
        ci95_half_width_m=ci,
    )


def save_error_cdf(errors: np.ndarray, output: str | Path, title: str) -> None:
    values = np.sort(np.asarray(errors, dtype=float).ravel())
    values = values[np.isfinite(values)]
    if not len(values):
        return
    cdf = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, cdf)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_curve(
    history: list[dict[str, float | int]],
    output: str | Path,
    title: str,
    *,
    error_key: str = "test_mean_error_m",
    error_label: str = "Test mean error (m)",
) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["training_loss"]) for row in history]
    errors = [float(row[error_key]) for row in history]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, losses, label="training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, errors, linestyle="--", label=error_label.lower())
    ax2.set_ylabel(error_label)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
