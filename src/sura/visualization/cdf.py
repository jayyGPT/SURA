"""CDF data preparation kept separate from plotting style."""

from __future__ import annotations

import numpy as np


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(values, dtype=float).ravel()
    samples = np.sort(samples[np.isfinite(samples)])
    if len(samples) == 0:
        raise ValueError("values contains no finite samples")
    probabilities = np.arange(1, len(samples) + 1, dtype=float) / len(samples)
    return samples, probabilities
