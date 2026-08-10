"""Legacy anomaly-gradient DualKalmanNet retained as a reproducibility baseline.

The active research milestone will replace this scalar anomaly pathway with the
MagSequenceMatcher's two-dimensional position and uncertainty outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MagneticAnomalyMap:
    """Regular scalar magnetic map and its spatial gradients."""

    values: Tensor
    gradient_x: Tensor
    gradient_y: Tensor
    x0: float
    y0: float
    cell: float

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must be a two-dimensional grid")
        if self.gradient_x.shape != self.values.shape or self.gradient_y.shape != self.values.shape:
            raise ValueError("gradient grids must match values")
        if self.cell <= 0:
            raise ValueError("cell must be positive")


def _bilinear_sample(grid: Tensor, position: Tensor, *, x0: float, y0: float, cell: float) -> Tensor:
    """Sample a regular [nx,ny] grid at world positions [B,2]."""
    nx, ny = grid.shape
    ix = ((position[:, 0] - x0) / cell).clamp(0, nx - 1.001)
    iy = ((position[:, 1] - y0) / cell).clamp(0, ny - 1.001)
    x_lo = ix.floor().long()
    y_lo = iy.floor().long()
    x_hi = (x_lo + 1).clamp(max=nx - 1)
    y_hi = (y_lo + 1).clamp(max=ny - 1)
    fx = ix - x_lo
    fy = iy - y_lo
    return (
        grid[x_lo, y_lo] * (1 - fx) * (1 - fy)
        + grid[x_hi, y_lo] * fx * (1 - fy)
        + grid[x_lo, y_hi] * (1 - fx) * fy
        + grid[x_hi, y_hi] * fx * fy
    )


class AnomalyDualKalmanNet(nn.Module):
    """Legacy GRU filter with Wi-Fi and scalar magnetic-anomaly corrections."""

    def __init__(self, magnetic_map: MagneticAnomalyMap, hidden_size: int = 64) -> None:
        super().__init__()
        self.x0 = float(magnetic_map.x0)
        self.y0 = float(magnetic_map.y0)
        self.cell_size = float(magnetic_map.cell)
        self.hidden_size = hidden_size
        self.register_buffer("map_values", magnetic_map.values.float())
        self.register_buffer("map_gradient_x", magnetic_map.gradient_x.float())
        self.register_buffer("map_gradient_y", magnetic_map.gradient_y.float())

        self.cell = nn.GRUCell(13, hidden_size)
        self.head = nn.Linear(hidden_size, 8)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(
                torch.tensor([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
            )

    def _map_sample(self, grid: Tensor, position: Tensor) -> Tensor:
        return _bilinear_sample(
            grid,
            position,
            x0=self.x0,
            y0=self.y0,
            cell=self.cell_size,
        )

    def forward(
        self,
        motion: Tensor,
        wifi_fix: Tensor,
        wifi_mask: Tensor,
        magnetic_observation: Tensor,
        start_absolute: Tensor,
        magnetic_mask: Tensor | None = None,
    ) -> Tensor:
        if motion.ndim != 3 or motion.shape[-1] != 2:
            raise ValueError("motion must have shape [B,T,2]")
        if wifi_fix.shape != motion.shape:
            raise ValueError("wifi_fix must have shape [B,T,2]")
        expected_scalar = motion.shape[:-1] + (1,)
        if wifi_mask.shape != expected_scalar or magnetic_observation.shape != expected_scalar:
            raise ValueError("masks and magnetic observations must have shape [B,T,1]")
        if start_absolute.shape != (motion.shape[0], 2):
            raise ValueError("start_absolute must have shape [B,2]")
        if magnetic_mask is None:
            magnetic_mask = torch.ones_like(wifi_mask)
        elif magnetic_mask.shape != expected_scalar:
            raise ValueError("magnetic_mask must have shape [B,T,1]")

        batch, steps, _ = motion.shape
        hidden = motion.new_zeros(batch, self.hidden_size)
        state = motion.new_zeros(batch, 2)
        previous_wifi = wifi_fix[:, 0]
        previous_update = motion.new_zeros(batch, 2)
        outputs: list[Tensor] = []

        for step in range(steps):
            wifi_available = wifi_mask[:, step]
            mag_available = magnetic_mask[:, step]
            predicted = state + motion[:, step]

            wifi_innovation = (wifi_fix[:, step] - predicted) * wifi_available
            wifi_delta = (wifi_fix[:, step] - previous_wifi) * wifi_available

            absolute_prediction = (predicted + start_absolute).detach()
            map_value = self._map_sample(self.map_values, absolute_prediction).unsqueeze(-1)
            gradient = torch.stack(
                [
                    self._map_sample(self.map_gradient_x, absolute_prediction),
                    self._map_sample(self.map_gradient_y, absolute_prediction),
                ],
                dim=1,
            )
            magnetic_innovation = (
                magnetic_observation[:, step] - map_value
            ) * mag_available

            features = torch.cat(
                [
                    wifi_innovation,
                    magnetic_innovation,
                    gradient,
                    wifi_delta,
                    motion[:, step],
                    previous_update,
                    wifi_available,
                    mag_available,
                ],
                dim=1,
            )
            hidden = self.cell(features, hidden)
            gains = self.head(hidden)
            wifi_gain = gains[:, :4].view(batch, 2, 2)
            magnetic_gain = gains[:, 4:].view(batch, 2, 2)

            wifi_correction = wifi_available * torch.bmm(
                wifi_gain, wifi_innovation.unsqueeze(-1)
            ).squeeze(-1)
            magnetic_direction = magnetic_innovation * gradient
            magnetic_correction = mag_available * torch.bmm(
                magnetic_gain, magnetic_direction.unsqueeze(-1)
            ).squeeze(-1)
            updated = predicted + wifi_correction + magnetic_correction

            previous_update = updated - state
            previous_wifi = torch.where(
                wifi_available.bool(), wifi_fix[:, step], previous_wifi
            )
            state = updated
            outputs.append(state)

        return torch.stack(outputs, dim=1)
