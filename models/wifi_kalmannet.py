"""Wi-Fi-only matrix-gain KalmanNet baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WiFiOnlyKalmanNet(nn.Module):
    """Fuse PDR motion and sparse Wi-Fi fixes using a learned 2x2 gain."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(9, hidden_size)
        self.head = nn.Linear(hidden_size, 4)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.5, 0.0, 0.0, 0.5]))

    def forward(self, motion: Tensor, wifi_fix: Tensor, wifi_mask: Tensor) -> Tensor:
        if motion.ndim != 3 or motion.shape[-1] != 2:
            raise ValueError("motion must have shape [B,T,2]")
        if wifi_fix.shape != motion.shape:
            raise ValueError("wifi_fix must have shape [B,T,2]")
        if wifi_mask.shape != motion.shape[:-1] + (1,):
            raise ValueError("wifi_mask must have shape [B,T,1]")

        batch, steps, _ = motion.shape
        hidden = motion.new_zeros(batch, self.hidden_size)
        state = motion.new_zeros(batch, 2)
        previous_wifi = wifi_fix[:, 0]
        previous_update = motion.new_zeros(batch, 2)
        outputs: list[Tensor] = []

        for step in range(steps):
            mask = wifi_mask[:, step]
            predicted = state + motion[:, step]
            innovation = (wifi_fix[:, step] - predicted) * mask
            wifi_delta = (wifi_fix[:, step] - previous_wifi) * mask
            features = torch.cat(
                [innovation, wifi_delta, motion[:, step], previous_update, mask], dim=1
            )
            hidden = self.cell(features, hidden)
            gain = self.head(hidden).view(batch, 2, 2)
            correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1) * mask
            updated = predicted + correction

            previous_update = updated - state
            previous_wifi = torch.where(mask.bool(), wifi_fix[:, step], previous_wifi)
            state = updated
            outputs.append(state)

        return torch.stack(outputs, dim=1)
