"""Rotation-invariant magnetic sequence position model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

MAGNETIC_FEATURES = ("magN", "magV", "magH", "dip")


class MagSequenceMatcher(nn.Module):
    """1D CNN producing a two-dimensional fix and heteroscedastic log variance."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 128,
        *,
        position_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_size <= 0:
            raise ValueError("network dimensions must be positive")
        if not 0 <= position_dropout < 1:
            raise ValueError("position_dropout must be in [0, 1)")

        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(position_dropout),
            nn.Linear(64, 2),
        )
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, channels]")
        if sequence.shape[2] != self.in_channels:
            raise ValueError("sequence channel count does not match model input")
        feature = self.encoder(sequence.transpose(1, 2)).squeeze(-1)
        return self.position_head(feature), self.variance_head(feature)


def heteroscedastic_nll(
    predicted_position: Tensor,
    log_variance: Tensor,
    true_position: Tensor,
    *,
    minimum_variance: float = 0.01,
) -> Tensor:
    """Gaussian negative log likelihood with a learned scalar position variance."""
    if predicted_position.shape != true_position.shape or predicted_position.shape[-1] != 2:
        raise ValueError("predicted_position and true_position must both have shape [B,2]")
    if log_variance.shape != predicted_position.shape[:-1] + (1,):
        raise ValueError("log_variance must have shape [B,1]")
    variance = torch.exp(log_variance).clamp(min=minimum_variance)
    squared_error = ((predicted_position - true_position) ** 2).sum(dim=-1, keepdim=True)
    return (0.5 * squared_error / variance + 0.5 * log_variance).mean()


def variance_from_log_variance(log_variance: Tensor, minimum: float = 0.01) -> Tensor:
    """Convert network log variance to a numerically safe variance."""
    return torch.exp(log_variance).clamp(min=minimum)
