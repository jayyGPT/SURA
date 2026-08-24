"""Causal step detection and two-dimensional pedestrian dead reckoning."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StepDetectorConfig:
    sampling_hz: float = 16.7
    threshold: float = 0.6
    refractory_seconds: float = 0.3
    ema_alpha: float = 0.98
    initial_gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.sampling_hz <= 0:
            raise ValueError("sampling_hz must be positive")
        if not 0 < self.ema_alpha < 1:
            raise ValueError("ema_alpha must be between zero and one")


class StepDetector:
    """Online detector using an EMA gravity baseline and refractory threshold."""

    def __init__(self, config: StepDetectorConfig | None = None) -> None:
        self.config = config or StepDetectorConfig()
        self.refractory_frames = int(
            self.config.refractory_seconds * self.config.sampling_hz
        )
        self.reset()

    def reset(self) -> None:
        self.mean = self.config.initial_gravity
        self.frame = 0
        self.last_step = -10**9

    def update(self, acceleration_magnitude: float) -> bool:
        if not np.isfinite(acceleration_magnitude):
            raise ValueError("acceleration_magnitude must be finite")
        self.frame += 1
        alpha = self.config.ema_alpha
        self.mean = alpha * self.mean + (1.0 - alpha) * acceleration_magnitude
        high_pass = acceleration_magnitude - self.mean
        if (
            high_pass > self.config.threshold
            and self.frame - self.last_step > self.refractory_frames
        ):
            self.last_step = self.frame
            return True
        return False
