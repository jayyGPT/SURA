"""Causal step detection and two-dimensional pedestrian dead reckoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


def pdr_controls(
    acceleration: np.ndarray,
    heading: np.ndarray,
    *,
    heading_offset: float,
    step_length: float,
    detector: StepDetector | None = None,
) -> np.ndarray:
    """Convert accelerometer and heading streams into per-frame displacement vectors."""
    acceleration = np.asarray(acceleration, dtype=float)
    heading = np.asarray(heading, dtype=float)
    if acceleration.ndim != 2 or acceleration.shape[1] != 3:
        raise ValueError("acceleration must have shape [time,3]")
    if heading.ndim != 1 or len(heading) != len(acceleration):
        raise ValueError("heading must have shape [time]")
    if step_length <= 0:
        raise ValueError("step_length must be positive")

    step_detector = detector or StepDetector()
    magnitude = np.linalg.norm(acceleration, axis=1)
    corrected_heading = heading + heading_offset
    controls = np.zeros((len(acceleration), 2), dtype=float)
    for index, sample in enumerate(magnitude):
        if step_detector.update(float(sample)):
            controls[index] = step_length * np.array(
                [np.cos(corrected_heading[index]), np.sin(corrected_heading[index])]
            )
    return controls


def fit_heading_offset(true_positions: np.ndarray, device_heading: np.ndarray) -> float:
    """Estimate the circular-mean rotation from device heading to the map frame."""
    positions = np.asarray(true_positions, dtype=float)
    heading = np.asarray(device_heading, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("true_positions must have shape [time,2]")
    if heading.ndim != 1 or len(heading) != len(positions):
        raise ValueError("device_heading must have shape [time]")
    delta = np.gradient(positions, axis=0)
    true_heading = np.arctan2(delta[:, 1], delta[:, 0])
    difference = true_heading - heading
    return float(np.arctan2(np.mean(np.sin(difference)), np.mean(np.cos(difference))))


def calibrate_step_length(
    walks: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    heading_offset: float,
    detector_config: StepDetectorConfig | None = None,
) -> float:
    """Estimate mean step length from acceleration/heading streams with true positions."""
    estimates: list[float] = []
    for sensor_stream, true_positions in walks:
        sensor_stream = np.asarray(sensor_stream, dtype=float)
        true_positions = np.asarray(true_positions, dtype=float)
        if sensor_stream.ndim != 2 or sensor_stream.shape[1] != 4:
            raise ValueError("sensor_stream must contain ax, ay, az, heading")
        unit_controls = pdr_controls(
            sensor_stream[:, :3],
            sensor_stream[:, 3],
            heading_offset=heading_offset,
            step_length=1.0,
            detector=StepDetector(detector_config),
        )
        step_count = int(np.count_nonzero(np.any(unit_controls != 0, axis=1)))
        path_length = float(np.linalg.norm(np.diff(true_positions, axis=0), axis=1).sum())
        if step_count > 0:
            estimates.append(path_length / step_count)
    if not estimates:
        raise ValueError("no steps were detected in the calibration walks")
    return float(np.mean(estimates))
