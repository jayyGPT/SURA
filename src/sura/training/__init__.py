"""Reproducible training workflows for canonical SURA models."""

from .magnetic import train_magnetic_sequence
from .wifi import train_wifi_heatmap

__all__ = ["train_magnetic_sequence", "train_wifi_heatmap"]
