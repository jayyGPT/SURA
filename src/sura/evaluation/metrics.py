"""Shared localization metrics used by experiments and paper figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


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
