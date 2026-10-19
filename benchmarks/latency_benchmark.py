#!/usr/bin/env python3
"""Per-step inference latency benchmark with independent repeats and reproducibility logging.

Measures per-step wall-clock latency for:
  - Wi-Fi-only KalmanNet
  - CNN Dual KalmanNet + relative confidence
  - Wi-Fi + Magnetic KNN (K=5 and K=20)

Follows PyTorch benchmarking best practices:
  - Explicit warm-up policy (single-sample queries for KNN, warmup walks for KalmanNet)
  - Explicit thread settings (torch.set_num_threads, OMP/MKL num threads)
  - Complete software versions, CPU topology, and power mode metadata logging
  - Independent benchmark repeats to quantify repeat-to-repeat reproducibility (mean, std, CV%, 95% CI)
  - Documented method evaluation order

Usage:
    python benchmarks/latency_benchmark.py
    python benchmarks/latency_benchmark.py --repeats 3 --threads 8
    python benchmarks/latency_benchmark.py --regime full --order full_first
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy.stats
import sklearn
from sklearn.neighbors import KNeighborsRegressor
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.wifi_kalmannet import WiFiOnlyKalmanNet
from train.kalmannet_wifiheatmap_magneticCNN_pdr import (
    CNNMagneticDualKalmanNet,
    DEFAULT_DATABASE,
    DEFAULT_MAG_CHECKPOINT,
    DEFAULT_OUTPUT,
    DEFAULT_T_BINS,
    DEFAULT_TEST_WALKS,
    DEFAULT_TRAIN_WALKS,
    DEFAULT_WIFI_CHECKPOINT,
    FUSION_FINAL_TEST_SEED,
    FUSION_HIDDEN_SIZE,
    FUSION_TRAIN_SEED,
    make_dataset,
    prepare_tensors,
    setup_environment,
)

METHOD_ORDER = [
    "wifi_only_kalmannet",
    "cnn_dual_kalmannet",
    "knn_k5",
    "knn_k20",
]


# ---------------------------------------------------------------------------
# Metadata collector
# ---------------------------------------------------------------------------

def collect_system_metadata(threads: int | None, device: torch.device) -> dict:
    """Collect hardware, software, concurrency, and power settings."""
    cpu_info = {"logical_cores": os.cpu_count()}
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object -Property Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        if isinstance(data, list):
            data = data[0]
        cpu_info["name"] = data.get("Name", platform.processor()).strip()
        cpu_info["physical_cores"] = data.get("NumberOfCores")
        cpu_info["logical_cores"] = data.get("NumberOfLogicalProcessors", os.cpu_count())
    except Exception:
        cpu_info["name"] = platform.processor()

    power_scheme = "Unknown"
    try:
        res = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        power_scheme = res.stdout.strip()
    except Exception:
        pass

    battery_status = "Unknown"
    try:
        res = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance -ClassName Win32_Battery | Select-Object -ExpandProperty BatteryStatus",
            ],
            capture_output=True,
            text=True,
        )
        code = res.stdout.strip()
        if code == "1":
            battery_status = "Discharging (Battery)"
        elif code == "2":
            battery_status = "AC Connected (Mains)"
        else:
            battery_status = f"Status code {code}"
    except Exception:
        pass

    config_str = torch.__config__.show()
    blas_backend = "Intel MKL (AVX2)" if "mkl" in config_str.lower() else "Standard"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "cpu": cpu_info,
        "power_scheme": power_scheme,
        "battery_status": battery_status,
        "thread_settings": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "default"),
        },
        "blas_backend": blas_backend,
    }


# ---------------------------------------------------------------------------
# Per-step benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_wifi_kalmannet(model: WiFiOnlyKalmanNet, tensors: tuple, walk_idx: int) -> list[float]:
    """Time each GRU step for one walk. Returns list of per-step ms."""
    motion, wifi, wifi_mask = tensors[0], tensors[1], tensors[2]
    m = motion[walk_idx : walk_idx + 1]
    w = wifi[walk_idx : walk_idx + 1]
    wm = wifi_mask[walk_idx : walk_idx + 1]

    steps = m.shape[1]
    hidden = m.new_zeros(1, model.hidden_size)
    state = m.new_zeros(1, 2)
    previous_wifi = w[:, 0]
    previous_update = m.new_zeros(1, 2)
    timings: list[float] = []

    for step in range(steps):
        t0 = time.perf_counter()
        mask = wm[:, step]
        predicted = state + m[:, step]
        innovation = (w[:, step] - predicted) * mask
        wifi_delta = (w[:, step] - previous_wifi) * mask
        features = torch.cat(
            [innovation, wifi_delta, m[:, step], previous_update, mask], dim=1
        )
        hidden = model.cell(features, hidden)
        gain = model.head(hidden).view(1, 2, 2)
        correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1) * mask
        updated = predicted + correction
        previous_update = updated - state
        previous_wifi = torch.where(mask.bool(), w[:, step], previous_wifi)
        state = updated
        timings.append((time.perf_counter() - t0) * 1000.0)

    return timings


def benchmark_dual_kalmannet(model: CNNMagneticDualKalmanNet, tensors: tuple, walk_idx: int) -> list[float]:
    """Time each GRU step for one walk. Returns list of per-step ms."""
    motion, wifi, wifi_mask, mag, mag_logvar, mag_mask = tensors[:6]
    m = motion[walk_idx : walk_idx + 1]
    w = wifi[walk_idx : walk_idx + 1]
    wm = wifi_mask[walk_idx : walk_idx + 1]
    mg = mag[walk_idx : walk_idx + 1]
    mlv = mag_logvar[walk_idx : walk_idx + 1]
    mm = mag_mask[walk_idx : walk_idx + 1]

    steps = m.shape[1]
    hidden = m.new_zeros(1, model.hidden_size)
    state = m.new_zeros(1, 2)
    previous_wifi = w[:, 0]
    previous_update = m.new_zeros(1, 2)
    timings: list[float] = []

    for step in range(steps):
        t0 = time.perf_counter()
        wifi_available = wm[:, step]
        mag_available = mm[:, step]
        predicted = state + m[:, step]
        wifi_innovation = (w[:, step] - predicted) * wifi_available
        magnetic_innovation = (mg[:, step] - predicted) * mag_available
        wifi_delta = (w[:, step] - previous_wifi) * wifi_available
        mag_confidence = mlv[:, step].clamp(-6.0, 8.0) * mag_available
        relative_lv = (
            mlv[:, step] - model.magnetic_reference_log_variance
        ).clamp(-8.0, 8.0)
        magnetic_weight = 1.0 / (1.0 + torch.exp(relative_lv))
        features = torch.cat(
            [
                wifi_innovation,
                magnetic_innovation,
                wifi_delta,
                m[:, step],
                previous_update,
                wifi_available,
                mag_available,
                mag_confidence,
            ],
            dim=1,
        )
        hidden = model.cell(features, hidden)
        gains = model.head(hidden)
        wifi_gain = gains[:, :4].view(1, 2, 2)
        magnetic_gain = gains[:, 4:].view(1, 2, 2)
        wifi_correction = wifi_available * torch.bmm(
            wifi_gain, wifi_innovation.unsqueeze(-1)
        ).squeeze(-1)
        magnetic_correction = magnetic_weight * mag_available * torch.bmm(
            magnetic_gain, magnetic_innovation.unsqueeze(-1)
        ).squeeze(-1)
        updated = predicted + wifi_correction + magnetic_correction
        previous_update = updated - state
        previous_wifi = torch.where(
            wifi_available.bool(), w[:, step], previous_wifi
        )
        state = updated
        timings.append((time.perf_counter() - t0) * 1000.0)

    return timings


def trajectory_features(data, reference_logvar: float):
    """Extract per-step features for KNN (same logic as wifi_mag_knn.py)."""
    _, wifi, wifi_mask, mag, mag_logvar, mag_mask, target, _ = data
    mag_available = mag_mask[..., 0] > 0.5
    wifi_available = wifi_mask[..., 0] > 0.5
    mag_filled = mag.copy()
    mag_filled[~mag_available] = wifi[~mag_available]
    logvar = np.clip(mag_logvar[..., 0], -6.0, 8.0)
    logvar[~mag_available] = reference_logvar
    features = np.concatenate(
        (
            wifi,
            mag_filled,
            logvar[..., None],
            wifi_available[..., None].astype(np.float32),
            mag_available[..., None].astype(np.float32),
        ),
        axis=2,
    )
    walks, bins, channels = features.shape
    return features.reshape(walks * bins, channels), target.reshape(walks * bins, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--wifi-checkpoint", type=Path, default=DEFAULT_WIFI_CHECKPOINT)
    parser.add_argument("--mag-checkpoint", type=Path, default=DEFAULT_MAG_CHECKPOINT)
    parser.add_argument("--kalmannet-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "runs" / "latency_log.csv",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-walks", type=int, default=DEFAULT_TRAIN_WALKS)
    parser.add_argument("--test-walks", type=int, default=DEFAULT_TEST_WALKS)
    parser.add_argument("--bins", type=int, default=DEFAULT_T_BINS)
    parser.add_argument("--regime", choices=["both", "full", "degraded"], default="both")
    parser.add_argument(
        "--order",
        choices=["full_first", "degraded_first"],
        default="full_first",
        help="Order of regimes when benchmarking both (full_first or degraded_first)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of independent benchmark repeat passes across the test set",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="PyTorch intra-op threads (default: system threads)",
    )
    parser.add_argument(
        "--warmup-walks",
        type=int,
        default=1,
        help="Warm-up walks for KalmanNet before recording",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=50,
        help="Warm-up single-queries for KNN before recording",
    )
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    metadata = collect_system_metadata(args.threads, device)
    metadata["benchmark_protocol"] = {
        "repeats": args.repeats,
        "threads": metadata["thread_settings"]["torch_num_threads"],
        "warmup_policy": {
            "kalmannet_warmup_walks": args.warmup_walks,
            "knn_warmup_queries": args.warmup_queries,
        },
        "method_order": METHOD_ORDER,
        "test_walks": args.test_walks,
        "bins_per_walk": args.bins,
        "steps_per_walk_method": args.test_walks * args.bins,
    }

    print(f"Platform: {metadata['platform']}")
    print(f"CPU: {metadata['cpu'].get('name', 'Unknown')}")
    print(f"Threads: PyTorch={metadata['thread_settings']['torch_num_threads']}, Interop={metadata['thread_settings']['torch_interop_threads']}")
    print(f"Power Scheme: {metadata['power_scheme']}")
    print(f"Battery: {metadata['battery_status']}")
    print(f"Device: {device}")
    print(f"Repeats: {args.repeats} independent evaluation passes")
    print(f"Method Order: {METHOD_ORDER}")

    env = setup_environment(
        args.database.resolve(),
        args.wifi_checkpoint.resolve(),
        args.mag_checkpoint.resolve(),
        device,
    )

    regimes_defs = {
        "full": ("Full Wi-Fi (1 Hz)", 1.0, 0.0),
        "degraded": ("Degraded Wi-Fi (5 s, 40% AP drop)", 5.0, 0.4),
    }
    if args.regime == "both":
        order_keys = (
            ["degraded", "full"]
            if args.order == "degraded_first"
            else ["full", "degraded"]
        )
        regimes = {k: regimes_defs[k] for k in order_keys}
    else:
        regimes = {args.regime: regimes_defs[args.regime]}

    metadata["benchmark_protocol"]["regime_order"] = list(regimes.keys())
    print(f"Regime Order: {list(regimes.keys())}")

    rows: list[tuple[int, str, str, int, int, float]] = []

    for regime_key, (regime_name, wifi_period, dropout) in regimes.items():
        print(f"\n{'=' * 72}\n{regime_name}\n{'=' * 72}")

        # -- Generate datasets ------------------------------------------------
        print("  generating training set (for KNN fitting)...")
        training = make_dataset(
            args.train_walks,
            seed=FUSION_TRAIN_SEED,
            env=env,
            device=device,
            wifi_period_s=wifi_period,
            ap_dropout=dropout,
            bins=args.bins,
        )
        print("  generating test set...")
        testing = make_dataset(
            args.test_walks,
            seed=FUSION_FINAL_TEST_SEED,
            env=env,
            device=device,
            wifi_period_s=wifi_period,
            ap_dropout=dropout,
            bins=args.bins,
        )

        # -- Load KalmanNet weights -------------------------------------------
        regime_dir = args.kalmannet_dir / regime_key
        wifi_kn_path = regime_dir / "wifi_only_kalmannet.pt"
        dual_path = regime_dir / "cnn_dual_relative_confidence.pt"

        baseline = dual = None
        if wifi_kn_path.is_file():
            baseline = WiFiOnlyKalmanNet(hidden_size=FUSION_HIDDEN_SIZE).to(device)
            baseline.load_state_dict(
                torch.load(wifi_kn_path, map_location=device, weights_only=False)["state_dict"]
            )
            baseline.eval()
        else:
            print(f"  WARNING: {wifi_kn_path} not found")

        if dual_path.is_file():
            dual_payload = torch.load(dual_path, map_location=device, weights_only=False)
            dual = CNNMagneticDualKalmanNet(
                hidden_size=FUSION_HIDDEN_SIZE,
                magnetic_reference_log_variance=dual_payload.get(
                    "magnetic_reference_log_variance", 0.0
                ),
            ).to(device)
            dual.load_state_dict(dual_payload["state_dict"])
            dual.eval()
        else:
            print(f"  WARNING: {dual_path} not found")

        tensors = prepare_tensors(testing, device)
        num_walks = args.test_walks
        num_bins = args.bins

        # -- Prepare KNN models -----------------------------------------------
        training_mag_available = training[5][..., 0] > 0.5
        reference_logvar = (
            float(np.median(training[4][..., 0][training_mag_available]))
            if training_mag_available.any()
            else 0.0
        )
        x_train, y_train = trajectory_features(training, reference_logvar)
        x_test, _ = trajectory_features(testing, reference_logvar)
        mean = x_train.mean(axis=0).astype(np.float32)
        std = x_train.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        x_train_scaled = ((x_train - mean) / std).astype(np.float32)
        x_test_scaled = ((x_test - mean) / std).astype(np.float32)

        knn_models: dict[str, KNeighborsRegressor] = {}
        for k in (5, 20):
            knn = KNeighborsRegressor(
                n_neighbors=k, metric="euclidean", weights="uniform", algorithm="auto"
            )
            knn.fit(x_train_scaled, y_train)
            knn_models[f"knn_k{k}"] = knn
            print(f"  fitted KNN K={k} on {len(x_train_scaled)} training points")

        # -- Warm-up ----------------------------------------------------------
        print(f"  warming up ({args.warmup_walks} walks for KalmanNet, {args.warmup_queries} queries for KNN)...")
        if baseline is not None:
            with torch.no_grad():
                for w_idx in range(min(args.warmup_walks, num_walks)):
                    benchmark_wifi_kalmannet(baseline, tensors, w_idx)
        if dual is not None:
            with torch.no_grad():
                for w_idx in range(min(args.warmup_walks, num_walks)):
                    benchmark_dual_kalmannet(dual, tensors, w_idx)
        for i in range(min(args.warmup_queries, len(x_test_scaled))):
            knn_models["knn_k5"].predict(x_test_scaled[i : i + 1])
            knn_models["knn_k20"].predict(x_test_scaled[i : i + 1])

        # -- Independent Repeat Loops -----------------------------------------
        for repeat_idx in range(1, args.repeats + 1):
            print(f"\n  --- Repeat {repeat_idx}/{args.repeats} ---")

            for method_name in METHOD_ORDER:
                if method_name == "wifi_only_kalmannet" and baseline is not None:
                    print(f"    benchmarking {method_name} ({num_walks} walks)...")
                    with torch.no_grad():
                        for walk in range(num_walks):
                            timings = benchmark_wifi_kalmannet(baseline, tensors, walk)
                            for step, ms in enumerate(timings):
                                rows.append((repeat_idx, method_name, regime_key, walk, step, ms))

                elif method_name == "cnn_dual_kalmannet" and dual is not None:
                    print(f"    benchmarking {method_name} ({num_walks} walks)...")
                    with torch.no_grad():
                        for walk in range(num_walks):
                            timings = benchmark_dual_kalmannet(dual, tensors, walk)
                            for step, ms in enumerate(timings):
                                rows.append((repeat_idx, method_name, regime_key, walk, step, ms))

                elif method_name in knn_models:
                    model = knn_models[method_name]
                    print(f"    benchmarking {method_name} ({num_walks} walks)...")
                    for walk in range(num_walks):
                        base = walk * num_bins
                        for step in range(num_bins):
                            t0 = time.perf_counter()
                            model.predict(x_test_scaled[base + step : base + step + 1])
                            rows.append(
                                (
                                    repeat_idx,
                                    method_name,
                                    regime_key,
                                    walk,
                                    step,
                                    (time.perf_counter() - t0) * 1000.0,
                                )
                            )

    # -- Write CSV -------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "method", "regime", "walk", "step", "latency_ms"])
        writer.writerows(rows)
    print(f"\nLogged {len(rows)} timings to {args.output}")

    # -- Statistical Analysis & Repeat-to-Repeat Reproducibility ---------------
    reproducibility_report: dict[str, dict] = {}
    regimes_seen = sorted(set(r[2] for r in rows))
    methods_seen = [m for m in METHOD_ORDER if any(r[1] == m for r in rows)]

    print(f"\n{'=' * 95}")
    print("SUMMARY & REPEAT-TO-REPEAT REPRODUCIBILITY (per-step latency in ms)")
    print("=" * 95)

    for regime in regimes_seen:
        print(f"\n[Regime: {regime}]")
        reproducibility_report[regime] = {}

        for method in methods_seen:
            method_rows = [r for r in rows if r[1] == method and r[2] == regime]
            if not method_rows:
                continue

            repeat_means: list[float] = []
            repeat_medians: list[float] = []
            for r_idx in range(1, args.repeats + 1):
                r_vals = np.array([r[5] for r in method_rows if r[0] == r_idx])
                if len(r_vals) > 0:
                    repeat_means.append(float(r_vals.mean()))
                    repeat_medians.append(float(np.median(r_vals)))

            pooled_vals = np.array([r[5] for r in method_rows])
            r_means_arr = np.array(repeat_means)

            inter_mean = float(r_means_arr.mean())
            inter_std = float(r_means_arr.std(ddof=1)) if len(r_means_arr) > 1 else 0.0
            cv_percent = (inter_std / inter_mean * 100.0) if inter_mean > 0 else 0.0

            ci95 = 0.0
            if len(r_means_arr) > 1:
                t_crit = float(scipy.stats.t.ppf(0.975, df=len(r_means_arr) - 1))
                ci95 = float(t_crit * (inter_std / np.sqrt(len(r_means_arr))))

            reproducibility_report[regime][method] = {
                "repeat_means_ms": [round(m, 5) for m in repeat_means],
                "repeat_medians_ms": [round(m, 5) for m in repeat_medians],
                "inter_repeat_mean_ms": round(inter_mean, 5),
                "inter_repeat_std_ms": round(inter_std, 5),
                "coefficient_of_variation_pct": round(cv_percent, 3),
                "ci95_ms": round(ci95, 5),
                "pooled_mean_ms": round(float(pooled_vals.mean()), 5),
                "pooled_median_ms": round(float(np.median(pooled_vals)), 5),
                "pooled_std_ms": round(float(pooled_vals.std()), 5),
                "pooled_p95_ms": round(float(np.percentile(pooled_vals, 95)), 5),
                "pooled_p99_ms": round(float(np.percentile(pooled_vals, 99)), 5),
                "pooled_max_ms": round(float(pooled_vals.max()), 5),
                "pooled_min_ms": round(float(pooled_vals.min()), 5),
            }

            rep_str = ", ".join(f"R{i+1}={m:.4f}" for i, m in enumerate(repeat_means))
            print(
                f"  {method:<24s} | Inter-Repeat Mean={inter_mean:.5f} ms "
                f"| Std={inter_std:.5f} ms | CV={cv_percent:.2f}% | 95% CI=[{inter_mean - ci95:.5f}, {inter_mean + ci95:.5f}]\n"
                f"  {'':<24s} | Repeats: ({rep_str})\n"
                f"  {'':<24s} | Pooled: Median={np.median(pooled_vals):.5f} ms, P95={np.percentile(pooled_vals, 95):.5f} ms, Max={pooled_vals.max():.5f} ms\n"
            )

    metadata["results"] = reproducibility_report

    # -- Write Metadata JSON ---------------------------------------------------
    meta_path = args.output.with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reproducibility metadata to {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
