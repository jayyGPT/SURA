#!/usr/bin/env python3
"""Ablate the temporal Wi-Fi-fix delta from both KalmanNet variants.

The comparison is paired and protocol-identical: each with-delta / no-delta pair is
trained on the same generated trajectories, with the same initialization seed and
same minibatch shuffle seed. The only architectural difference is the two-scalar
feature delta_z_wifi = z_wifi,t - z_wifi,previous.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.wifi_kalmannet import WiFiOnlyKalmanNet
from train.kalmannet_wifiheatmap_magneticCNN_pdr import (
    CNNMagneticDualKalmanNet,
    DEFAULT_DATABASE,
    DEFAULT_FUSION_EPOCHS,
    DEFAULT_MAG_CHECKPOINT,
    DEFAULT_T_BINS,
    DEFAULT_TEST_WALKS,
    DEFAULT_TRAIN_WALKS,
    DEFAULT_WIFI_CHECKPOINT,
    FUSION_HIDDEN_SIZE,
    FUSION_TEST_SEED,
    FUSION_TRAIN_SEED,
    SEED,
    evaluate_filter,
    make_dataset,
    setup_environment,
    summarize,
    train_filter,
    validate_trajectory_split,
)

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "wifi_delta_ablation"


class WiFiOnlyNoDeltaKalmanNet(nn.Module):
    """Wi-Fi-only KalmanNet with delta_z_wifi removed (7 GRU inputs)."""

    def __init__(self, hidden_size: int = FUSION_HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(7, hidden_size)
        self.head = nn.Linear(hidden_size, 4)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.5, 0.0, 0.0, 0.5]))

    def forward(self, motion: Tensor, wifi_fix: Tensor, wifi_mask: Tensor) -> Tensor:
        batch, steps, _ = motion.shape
        hidden = motion.new_zeros(batch, self.hidden_size)
        state = motion.new_zeros(batch, 2)
        previous_update = motion.new_zeros(batch, 2)
        outputs: list[Tensor] = []

        for step in range(steps):
            mask = wifi_mask[:, step]
            predicted = state + motion[:, step]
            innovation = (wifi_fix[:, step] - predicted) * mask
            features = torch.cat(
                [innovation, motion[:, step], previous_update, mask], dim=1
            )
            hidden = self.cell(features, hidden)
            gain = self.head(hidden).view(batch, 2, 2)
            correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1) * mask
            updated = predicted + correction
            previous_update = updated - state
            state = updated
            outputs.append(state)

        return torch.stack(outputs, dim=1)


class CNNMagneticDualNoDeltaKalmanNet(nn.Module):
    """CNN DualKalmanNet with delta_z_wifi removed (11 GRU inputs)."""

    def __init__(
        self,
        hidden_size: int = FUSION_HIDDEN_SIZE,
        magnetic_reference_log_variance: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.magnetic_reference_log_variance = float(magnetic_reference_log_variance)
        self.cell = nn.GRUCell(11, hidden_size)
        self.head = nn.Linear(hidden_size, 8)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(
                torch.tensor([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
            )

    def forward(
        self,
        motion: Tensor,
        wifi_fix: Tensor,
        wifi_mask: Tensor,
        magnetic_fix: Tensor,
        magnetic_log_variance: Tensor,
        magnetic_mask: Tensor,
    ) -> Tensor:
        batch, steps, _ = motion.shape
        hidden = motion.new_zeros(batch, self.hidden_size)
        state = motion.new_zeros(batch, 2)
        previous_update = motion.new_zeros(batch, 2)
        outputs: list[Tensor] = []

        for step in range(steps):
            wifi_available = wifi_mask[:, step]
            mag_available = magnetic_mask[:, step]
            predicted = state + motion[:, step]
            wifi_innovation = (wifi_fix[:, step] - predicted) * wifi_available
            magnetic_innovation = (magnetic_fix[:, step] - predicted) * mag_available
            mag_confidence = magnetic_log_variance[:, step].clamp(-6.0, 8.0) * mag_available
            relative_log_variance = (
                magnetic_log_variance[:, step] - self.magnetic_reference_log_variance
            ).clamp(-8.0, 8.0)
            magnetic_weight = 1.0 / (1.0 + torch.exp(relative_log_variance))

            features = torch.cat(
                [
                    wifi_innovation,
                    magnetic_innovation,
                    motion[:, step],
                    previous_update,
                    wifi_available,
                    mag_available,
                    mag_confidence,
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
            magnetic_correction = magnetic_weight * mag_available * torch.bmm(
                magnetic_gain, magnetic_innovation.unsqueeze(-1)
            ).squeeze(-1)
            updated = predicted + wifi_correction + magnetic_correction
            previous_update = updated - state
            state = updated
            outputs.append(state)

        return torch.stack(outputs, dim=1)


def paired_difference(with_delta: np.ndarray, without_delta: np.ndarray) -> dict[str, float]:
    """Report no-delta minus with-delta per-walk error; negative favors removal."""
    diff = np.asarray(without_delta, dtype=float) - np.asarray(with_delta, dtype=float)
    ci = 1.96 * diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
    return {
        "mean_difference_m_no_delta_minus_with_delta": float(diff.mean()),
        "median_difference_m": float(np.median(diff)),
        "ci95_half_width_m": float(ci),
        "ci95_low_m": float(diff.mean() - ci),
        "ci95_high_m": float(diff.mean() + ci),
        "walks_no_delta_better": int(np.sum(diff < 0)),
        "walks_with_delta_better": int(np.sum(diff > 0)),
        "walks_tied": int(np.sum(diff == 0)),
    }


def train_eval(model_factory, data_train, data_test, device, *, magnetic: bool, epochs: int):
    # Reset before model construction AND before train_filter shuffles its indices.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_factory().to(device)
    model, history = train_filter(
        model, data_train, device, epochs=epochs, uses_magnetic=magnetic
    )
    errors, _ = evaluate_filter(model, data_test, device, uses_magnetic=magnetic)
    return errors, history


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = setup_environment(args.database, args.wifi_checkpoint, args.mag_checkpoint, device)
    regimes = {
        "full": ("Full Wi-Fi (1 Hz)", 1.0, 0.0),
        "degraded": ("Degraded Wi-Fi (5 s, 40% AP drop)", 5.0, 0.4),
    }
    report: dict[str, object] = {
        "question": "Does delta_z_wifi improve the learned fusion filters?",
        "difference_sign_convention": "no_delta_minus_with_delta; negative favors removing delta_z_wifi",
        "protocol": {
            "train_walks": args.train_walks,
            "test_walks": args.test_walks,
            "fusion_epochs": args.fusion_epochs,
            "bins": args.bins,
            "train_seed": FUSION_TRAIN_SEED,
            "test_seed": FUSION_TEST_SEED,
            "model_and_shuffle_seed_reset_before_every_training": SEED,
            "model_seed_is_reset_before_construction": True,
            "device": str(device),
        },
        "regimes": {},
    }

    for key, (name, wifi_period, dropout) in regimes.items():
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        training = make_dataset(
            args.train_walks,
            FUSION_TRAIN_SEED,
            env,
            device,
            wifi_period_s=wifi_period,
            ap_dropout=dropout,
            bins=args.bins,
        )
        testing = make_dataset(
            args.test_walks,
            FUSION_TEST_SEED,
            env,
            device,
            wifi_period_s=wifi_period,
            ap_dropout=dropout,
            bins=args.bins,
        )
        split_audit = validate_trajectory_split(training, testing)

        training_mag_available = training[5][..., 0] > 0.5
        training_logvar = training[4][..., 0][training_mag_available]
        ref_logvar = float(np.median(training_logvar))

        print("  Wi-Fi-only with delta")
        wifi_with, wifi_with_hist = train_eval(
            lambda: WiFiOnlyKalmanNet(FUSION_HIDDEN_SIZE),
            training, testing, device, magnetic=False, epochs=args.fusion_epochs,
        )
        print("  Wi-Fi-only without delta")
        wifi_without, wifi_without_hist = train_eval(
            lambda: WiFiOnlyNoDeltaKalmanNet(FUSION_HIDDEN_SIZE),
            training, testing, device, magnetic=False, epochs=args.fusion_epochs,
        )
        print("  CNN Dual with delta")
        dual_with, dual_with_hist = train_eval(
            lambda: CNNMagneticDualKalmanNet(FUSION_HIDDEN_SIZE, ref_logvar),
            training, testing, device, magnetic=True, epochs=args.fusion_epochs,
        )
        print("  CNN Dual without delta")
        dual_without, dual_without_hist = train_eval(
            lambda: CNNMagneticDualNoDeltaKalmanNet(FUSION_HIDDEN_SIZE, ref_logvar),
            training, testing, device, magnetic=True, epochs=args.fusion_epochs,
        )

        regime_report = {
            "name": name,
            "wifi_period_s": wifi_period,
            "ap_dropout": dropout,
            "trajectory_split_audit": split_audit,
            "magnetic_reference_log_variance_training": ref_logvar,
            "wifi_only": {
                "with_delta_9_inputs": summarize(wifi_with),
                "without_delta_7_inputs": summarize(wifi_without),
                "paired_difference": paired_difference(wifi_with, wifi_without),
                "with_delta_training_loss": wifi_with_hist,
                "without_delta_training_loss": wifi_without_hist,
            },
            "cnn_dual_relative_variance": {
                "with_delta_13_inputs": summarize(dual_with),
                "without_delta_11_inputs": summarize(dual_without),
                "paired_difference": paired_difference(dual_with, dual_without),
                "with_delta_training_loss": dual_with_hist,
                "without_delta_training_loss": dual_without_hist,
            },
        }
        report["regimes"][key] = regime_report

        print("  Wi-Fi paired diff:", regime_report["wifi_only"]["paired_difference"])
        print("  Dual paired diff:", regime_report["cnn_dual_relative_variance"]["paired_difference"])

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Wi-Fi temporal-delta ablation",
        "",
        "Only `delta_z_wifi = z_wifi,t - z_wifi,previous` is removed. Negative paired differences favor the no-delta model.",
        "",
        "| Regime | Model | With delta mean | No delta mean | Paired mean diff (no-with) | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in ("full", "degraded"):
        rr = report["regimes"][key]
        for label, model_key, with_key, without_key in (
            ("Wi-Fi-only", "wifi_only", "with_delta_9_inputs", "without_delta_7_inputs"),
            ("CNN Dual + rel. variance", "cnn_dual_relative_variance", "with_delta_13_inputs", "without_delta_11_inputs"),
        ):
            block = rr[model_key]
            diff = block["paired_difference"]
            lines.append(
                f"| {rr['name']} | {label} | {block[with_key]['mean_m']:.4f} m | "
                f"{block[without_key]['mean_m']:.4f} m | "
                f"{diff['mean_difference_m_no_delta_minus_with_delta']:+.4f} m | "
                f"[{diff['ci95_low_m']:+.4f}, {diff['ci95_high_m']:+.4f}] m |"
            )
    lines += [
        "",
        "Interpretation should be based on the paired differences above; this file does not automatically choose the final architecture.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--wifi-checkpoint", type=Path, default=DEFAULT_WIFI_CHECKPOINT)
    parser.add_argument("--mag-checkpoint", type=Path, default=DEFAULT_MAG_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-walks", type=int, default=DEFAULT_TRAIN_WALKS)
    parser.add_argument("--test-walks", type=int, default=DEFAULT_TEST_WALKS)
    parser.add_argument("--fusion-epochs", type=int, default=DEFAULT_FUSION_EPOCHS)
    parser.add_argument("--bins", type=int, default=DEFAULT_T_BINS)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
