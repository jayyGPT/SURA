"""Wi-Fi probability-heatmap measurement model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GridSpec:
    """Serializable geometry of a regular two-dimensional grid."""

    x0: float
    y0: float
    nx: int
    ny: int
    cell: float


class Grid:
    """Regular two-dimensional grid covering surveyed coordinates."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, cell: float = 1.0) -> None:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys) or len(xs) == 0:
            raise ValueError("xs and ys must be non-empty one-dimensional arrays of equal length")
        if cell <= 0:
            raise ValueError("cell must be positive")

        self.x0 = float(np.floor(xs.min()))
        self.x1 = float(np.ceil(xs.max()))
        self.y0 = float(np.floor(ys.min()))
        self.y1 = float(np.ceil(ys.max()))
        self.cell = float(cell)
        self.nx = int(round((self.x1 - self.x0) / self.cell)) + 1
        self.ny = int(round((self.y1 - self.y0) / self.cell)) + 1

        gx = self.x0 + np.arange(self.nx) * self.cell
        gy = self.y0 + np.arange(self.ny) * self.cell
        self.gxx, self.gyy = np.meshgrid(gx, gy, indexing="ij")
        self.coords = np.stack([self.gxx.ravel(), self.gyy.ravel()], axis=1)
        self.n_cells = self.nx * self.ny

    @property
    def spec(self) -> GridSpec:
        return GridSpec(self.x0, self.y0, self.nx, self.ny, self.cell)

    def gaussian_target(self, x: float, y: float, sigma: float = 2.0) -> np.ndarray:
        """Return a normalized Gaussian soft target over grid cells."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        d2 = (self.gxx - x) ** 2 + (self.gyy - y) ** 2
        target = np.exp(-d2 / (2.0 * sigma**2)).ravel()
        total = float(target.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Gaussian target could not be normalized")
        return (target / total).astype(np.float32)


def encode_wifi(
    rss: np.ndarray,
    *,
    absent_floor: float = -100.0,
    clip_min: float = -90.0,
    clip_max: float = -30.0,
) -> np.ndarray:
    """Clip RSSI values, scale them to [0, 1], and map absent APs to zero."""
    if clip_min >= clip_max:
        raise ValueError("clip_min must be smaller than clip_max")
    raw = np.asarray(rss, dtype=float)
    encoded = np.clip(raw, clip_min, clip_max)
    encoded = (encoded - clip_min) / (clip_max - clip_min)
    encoded[raw <= absent_floor] = 0.0
    return encoded.astype(np.float32)


class WifiHeatmapNet(nn.Module):
    """MLP mapping an N-AP RSSI vector to M spatial-cell logits."""

    def __init__(
        self,
        n_access_points: int,
        n_cells: int,
        *,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_access_points <= 0 or n_cells <= 0 or hidden_size <= 0:
            raise ValueError("network dimensions must be positive")
        self.net = nn.Sequential(
            nn.Linear(n_access_points, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_cells),
        )

    def forward(self, rss: Tensor) -> Tensor:
        if rss.ndim != 2:
            raise ValueError("rss must have shape [batch, access_points]")
        return self.net(rss)


def soft_argmax(logits: Tensor, coordinates: Tensor) -> Tensor:
    """Convert heatmap logits to a continuous probability-weighted position."""
    if logits.ndim != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("expected logits [B,M] and coordinates [M,2]")
    if logits.shape[1] != coordinates.shape[0]:
        raise ValueError("logit cell count does not match coordinate count")
    return torch.softmax(logits, dim=1) @ coordinates


def heatmap_covariance(logits: Tensor, coordinates: Tensor) -> Tensor:
    """Return the probability-weighted 2x2 covariance for each heatmap."""
    probabilities = torch.softmax(logits, dim=1)
    mean = probabilities @ coordinates
    delta = coordinates.unsqueeze(0) - mean.unsqueeze(1)
    outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)
    return (probabilities.unsqueeze(-1).unsqueeze(-1) * outer).sum(dim=1)


def kl_divergence_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-9) -> Tensor:
    """Compute D_KL(target || prediction) for Gaussian soft labels."""
    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    log_probability = torch.log_softmax(logits, dim=1)
    return torch.sum(target * (torch.log(target + epsilon) - log_probability), dim=1).mean()
