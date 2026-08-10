#!/usr/bin/env python3
"""Compare magnetic CNN predicted variance with actual position error.

Run from the repository root:

    python benchmarks/analyze_magnetic_variance.py

The script reuses the same 60-walk synthetic test protocol as the CNN-output
DualKalmanNet experiment and writes a compact calibration report/CSV/plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train"))

from kalmannet_wifiheatmap_magneticCNN_pdr import (  # noqa: E402
    DEFAULT_DATABASE,
    DEFAULT_MAG_CHECKPOINT,
    DEFAULT_T_BINS,
    DEFAULT_WIFI_CHECKPOINT,
    make_dataset,
    setup_environment,
)

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "magnetic_variance_calibration"


def _safe_correlation(func, x: np.ndarray, y: np.ndarray) -> float:
    result = func(x, y)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def analyze_regime(
    name: str,
    data: tuple[np.ndarray, ...],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    _, _, _, magnetic_fix, log_variance, magnetic_mask, target, _ = data
    available = magnetic_mask[..., 0] > 0.5

    error = np.linalg.norm(magnetic_fix - target, axis=-1)[available].astype(float)
    logvar = log_variance[..., 0][available].astype(float)
    finite = np.isfinite(error) & np.isfinite(logvar)
    error = error[finite]
    logvar = logvar[finite]

    # The CNN outputs log(sigma^2). Clipping here is only numerical protection for
    # exponentiation in the analysis; the raw log-variance is preserved separately.
    variance = np.exp(np.clip(logvar, -20.0, 20.0))
    sigma = np.sqrt(variance)

    samples = pd.DataFrame(
        {
            "regime": name,
            "predicted_log_variance": logvar,
            "predicted_variance_m2": variance,
            "predicted_sigma_m": sigma,
            "actual_position_error_m": error,
        }
    )
    samples["uncertainty_quartile"] = pd.qcut(
        samples["predicted_sigma_m"],
        q=4,
        labels=["Q1 most confident", "Q2", "Q3", "Q4 least confident"],
        duplicates="drop",
    )

    quartiles = (
        samples.groupby("uncertainty_quartile", observed=True)
        .agg(
            samples=("actual_position_error_m", "size"),
            mean_predicted_sigma_m=("predicted_sigma_m", "mean"),
            median_predicted_sigma_m=("predicted_sigma_m", "median"),
            mean_actual_error_m=("actual_position_error_m", "mean"),
            median_actual_error_m=("actual_position_error_m", "median"),
            rmse_radius_m=("actual_position_error_m", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )

    # For an isotropic 2-D Gaussian, E[||e||^2] = 2*sigma^2. This gives a useful
    # absolute calibration check in addition to rank/correlation checks.
    empirical_per_axis_variance = float(np.mean(error**2) / 2.0)
    predicted_variance_mean = float(np.mean(variance))
    q1_error = float(quartiles.iloc[0]["mean_actual_error_m"])
    q4_error = float(quartiles.iloc[-1]["mean_actual_error_m"])

    summary = {
        "regime": name,
        "samples": int(len(error)),
        "mean_actual_error_m": float(np.mean(error)),
        "median_actual_error_m": float(np.median(error)),
        "mean_predicted_sigma_m": float(np.mean(sigma)),
        "median_predicted_sigma_m": float(np.median(sigma)),
        "pearson_logvariance_vs_error": _safe_correlation(pearsonr, logvar, error),
        "spearman_uncertainty_vs_error": _safe_correlation(spearmanr, sigma, error),
        "q4_to_q1_mean_error_ratio": q4_error / q1_error,
        "predicted_variance_mean_m2": predicted_variance_mean,
        "empirical_per_axis_variance_m2": empirical_per_axis_variance,
        "variance_calibration_ratio_predicted_over_empirical": (
            predicted_variance_mean / empirical_per_axis_variance
            if empirical_per_axis_variance > 0
            else float("nan")
        ),
        "fraction_error_within_1sigma": float(np.mean(error <= sigma)),
        "fraction_error_within_2sigma": float(np.mean(error <= 2.0 * sigma)),
    }
    return summary, quartiles, samples


def plot_deciles(samples: pd.DataFrame, output: Path) -> None:
    frames = []
    for regime, group in samples.groupby("regime"):
        work = group.copy()
        work["decile"] = pd.qcut(work["predicted_sigma_m"], q=10, labels=False, duplicates="drop")
        frame = (
            work.groupby("decile", observed=True)
            .agg(
                predicted_sigma_m=("predicted_sigma_m", "mean"),
                actual_error_m=("actual_position_error_m", "mean"),
            )
            .reset_index()
        )
        frame["regime"] = regime
        frames.append(frame)

    deciles = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for regime, frame in deciles.groupby("regime"):
        ax.plot(
            frame["predicted_sigma_m"],
            frame["actual_error_m"],
            marker="o",
            label=regime,
        )
    ax.set_xlabel("Mean CNN-predicted sigma in uncertainty decile (m)")
    ax.set_ylabel("Mean actual magnetic position error (m)")
    ax.set_title("Does predicted magnetic uncertainty track actual error?")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    summaries: list[dict[str, object]],
    quartiles: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Magnetic CNN variance calibration",
        "",
        "This benchmark compares the magnetic CNN's predicted `log(sigma^2)` with its actual 2-D position error on the same synthetic 60-walk test protocol used by the CNN-output DualKalmanNet experiment.",
        "",
        "## Correlation summary",
        "",
        "| Regime | Samples | Mean error | Mean predicted sigma | Spearman uncertainty/error | Pearson logvar/error | Q4/Q1 mean-error ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['regime']} | {row['samples']} | {row['mean_actual_error_m']:.3f} m | "
            f"{row['mean_predicted_sigma_m']:.3f} m | {row['spearman_uncertainty_vs_error']:.3f} | "
            f"{row['pearson_logvariance_vs_error']:.3f} | {row['q4_to_q1_mean_error_ratio']:.2f}x |"
        )

    lines += ["", "## Uncertainty quartiles", ""]
    for regime in quartiles["regime"].unique():
        lines += [f"### {regime}", "", "| Quartile | Samples | Predicted sigma | Mean actual error | Median actual error | Radial RMSE |", "|---|---:|---:|---:|---:|---:|"]
        subset = quartiles[quartiles["regime"] == regime]
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['uncertainty_quartile']} | {int(row['samples'])} | "
                f"{row['mean_predicted_sigma_m']:.3f} m | {row['mean_actual_error_m']:.3f} m | "
                f"{row['median_actual_error_m']:.3f} m | {row['rmse_radius_m']:.3f} m |"
            )
        lines.append("")

    lines += [
        "## Absolute calibration check",
        "",
        "For an isotropic 2-D Gaussian, a calibrated scalar variance would approximately satisfy `E[||error||^2] = 2 sigma^2`. The JSON output records the mean predicted variance, empirical per-axis variance, and their ratio. Rank correlation is the more important quantity for deciding whether uncertainty gating is useful; absolute calibration can be corrected later if necessary.",
        "",
        "The generated `magnetic_variance_calibration.png` plots mean predicted sigma against mean actual error by uncertainty decile.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--wifi-checkpoint", type=Path, default=DEFAULT_WIFI_CHECKPOINT)
    parser.add_argument("--mag-checkpoint", type=Path, default=DEFAULT_MAG_CHECKPOINT)
    parser.add_argument("--walks", type=int, default=60)
    parser.add_argument("--bins", type=int, default=DEFAULT_T_BINS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)

    np.random.seed(0)
    torch.manual_seed(0)
    env = setup_environment(
        args.database.resolve(),
        args.wifi_checkpoint.resolve(),
        args.mag_checkpoint.resolve(),
        device,
    )

    regimes = [
        ("Full Wi-Fi (1 Hz)", 1.0, 0.0),
        ("Degraded Wi-Fi (5 s, 40% AP drop)", 5.0, 0.4),
    ]
    summaries = []
    quartile_frames = []
    sample_frames = []

    for name, period, dropout in regimes:
        print(f"Generating {args.walks} test walks: {name}")
        data = make_dataset(
            args.walks,
            seed=2,
            env=env,
            device=device,
            wifi_period_s=period,
            ap_dropout=dropout,
            bins=args.bins,
        )
        summary, quartiles, samples = analyze_regime(name, data)
        quartiles.insert(0, "regime", name)
        summaries.append(summary)
        quartile_frames.append(quartiles)
        sample_frames.append(samples)
        print(json.dumps(summary, indent=2))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_quartiles = pd.concat(quartile_frames, ignore_index=True)
    all_samples = pd.concat(sample_frames, ignore_index=True)

    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    all_quartiles.to_csv(output / "quartiles.csv", index=False)
    all_samples.to_csv(output / "samples.csv", index=False)
    plot_deciles(all_samples, output / "magnetic_variance_calibration.png")
    write_markdown(summaries, all_quartiles, output / "README.md")

    print(f"Saved calibration results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
