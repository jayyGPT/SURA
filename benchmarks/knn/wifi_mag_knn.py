#!/usr/bin/env python3
"""Current Wi-Fi + magnetic KNN baselines for the SURA/MagWi experiments.

Two protocols are intentionally kept separate:

1. ``static``: classical fingerprint KNN on the real processed MagWi node visits.
   All non-S9+ phones are training data and S9+ is held out for final evaluation.
   K and the Wi-Fi/magnetic block weight are selected using leave-one-phone-out
   validation inside the training phones only.

2. ``trajectory``: a non-temporal KNN fusion baseline on the exact synthetic
   250-train/60-test walk protocol used by the current KalmanNet experiment.
   Its inputs are the current Wi-Fi heatmap fix, magnetic-CNN fix/log-variance,
   and availability masks. It never sees PDR motion or recurrent history.

The trajectory protocol is the one that can be overlaid fairly with the current
KalmanNet CDFs. The static protocol is a real held-out-device fingerprinting
benchmark and should be reported separately rather than mixed with trajectory
errors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.wifi_kalmannet import WiFiOnlyKalmanNet
from tools.fingerprint import load_fingerprint_database
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
    INCLUDED_MODES,
    evaluate_filter,
    make_dataset,
    setup_environment,
    summarize,
    train_filter,
)

SEED = 0
HELD_OUT_PHONE = "S9+"
K_VALUES = (1, 3, 5, 7, 9, 11, 15, 20)
HYBRID_WIFI_WEIGHTS = (0.25, 0.50, 0.75)
MAG_COLUMNS = ("magN_mean", "magV_mean", "magH_mean", "dip_mean")


def _bool_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0).to_numpy(dtype=bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
        .to_numpy(dtype=bool)
    )


def _euclidean_errors(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(target) - np.asarray(prediction), axis=1)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    ci = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "samples": int(len(values)),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "max_m": float(values.max()),
        "ci95_half_width_m": float(ci),
    }


def _encode_wifi(matrix: np.ndarray) -> np.ndarray:
    """Match the canonical Wi-Fi preprocessing while retaining AP identity."""
    values = np.asarray(matrix, dtype=np.float32)
    absent = values <= -99.5
    encoded = (np.clip(values, -90.0, -30.0) + 90.0) / 60.0
    encoded[absent] = 0.0
    return encoded.astype(np.float32)


def _mag_scaler(magnetic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(magnetic, dtype=np.float64).mean(axis=0)
    std = np.asarray(magnetic, dtype=np.float64).std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _static_features(
    wifi: np.ndarray,
    magnetic: np.ndarray,
    *,
    mag_mean: np.ndarray,
    mag_std: np.ndarray,
    wifi_weight: float,
) -> np.ndarray:
    wifi_block = _encode_wifi(wifi)
    magnetic_block = (np.asarray(magnetic, dtype=np.float32) - mag_mean) / mag_std

    # Equalize feature-count effects before applying the modality weight. Without
    # this, ~250 AP dimensions dominate four magnetic features purely by dimension.
    wifi_block = wifi_block / math.sqrt(max(wifi_block.shape[1], 1))
    magnetic_block = magnetic_block / math.sqrt(max(magnetic_block.shape[1], 1))
    return np.hstack(
        (
            math.sqrt(wifi_weight) * wifi_block,
            math.sqrt(1.0 - wifi_weight) * magnetic_block,
        )
    ).astype(np.float32)


def _valid_k_values(train_count: int) -> list[int]:
    return [k for k in K_VALUES if k <= train_count]


def _select_static_hyperparameters(
    wifi: np.ndarray,
    magnetic: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    wifi_weights: tuple[float, ...],
) -> tuple[int, float, list[dict[str, float | int]]]:
    phones = np.unique(groups)
    if len(phones) < 2:
        raise ValueError("need at least two training phones for phone-held-out CV")
    splitter = GroupKFold(n_splits=len(phones))
    records: list[dict[str, float | int]] = []

    for wifi_weight in wifi_weights:
        for k in _valid_k_values(len(target)):
            fold_errors: list[float] = []
            possible = True
            for fit_index, validation_index in splitter.split(target, groups=groups):
                if k > len(fit_index):
                    possible = False
                    break
                mag_mean, mag_std = _mag_scaler(magnetic[fit_index])
                x_fit = _static_features(
                    wifi[fit_index], magnetic[fit_index],
                    mag_mean=mag_mean, mag_std=mag_std, wifi_weight=wifi_weight,
                )
                x_validation = _static_features(
                    wifi[validation_index], magnetic[validation_index],
                    mag_mean=mag_mean, mag_std=mag_std, wifi_weight=wifi_weight,
                )
                model = KNeighborsRegressor(n_neighbors=k, metric="euclidean", weights="uniform")
                model.fit(x_fit, target[fit_index])
                predicted = model.predict(x_validation)
                fold_errors.append(float(_euclidean_errors(target[validation_index], predicted).mean()))
            if possible and fold_errors:
                records.append(
                    {
                        "k": int(k),
                        "wifi_weight": float(wifi_weight),
                        "mean_phone_cv_error_m": float(np.mean(fold_errors)),
                    }
                )
    if not records:
        raise RuntimeError("no valid static KNN hyperparameter candidates")
    best = min(records, key=lambda row: row["mean_phone_cv_error_m"])
    return int(best["k"]), float(best["wifi_weight"]), records


def _fit_static_variant(
    train_wifi: np.ndarray,
    train_mag: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_wifi: np.ndarray,
    test_mag: np.ndarray,
    test_y: np.ndarray,
    *,
    weights: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    k, wifi_weight, cv = _select_static_hyperparameters(
        train_wifi, train_mag, train_y, train_groups, weights
    )
    mag_mean, mag_std = _mag_scaler(train_mag)
    x_train = _static_features(
        train_wifi, train_mag, mag_mean=mag_mean, mag_std=mag_std, wifi_weight=wifi_weight
    )
    x_test = _static_features(
        test_wifi, test_mag, mag_mean=mag_mean, mag_std=mag_std, wifi_weight=wifi_weight
    )
    model = KNeighborsRegressor(n_neighbors=k, metric="euclidean", weights="uniform")
    model.fit(x_train, train_y)
    prediction = model.predict(x_test)
    errors = _euclidean_errors(test_y, prediction)
    return prediction, errors, {
        "k": k,
        "wifi_weight": wifi_weight,
        "magnetic_weight": 1.0 - wifi_weight,
        "phone_group_cv": cv,
        "metrics": _summary(errors),
    }


def run_static(database: Path, output: Path, held_out_phone: str) -> dict[str, object]:
    db = load_fingerprint_database(database, included_modes=INCLUDED_MODES, require_wifi=False)
    frame = db.frame.copy()
    wifi_ok = _bool_mask(frame["has_wifi"])
    for column in MAG_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_mag = np.isfinite(frame[list(MAG_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    valid = frame.loc[wifi_ok & finite_mag].copy().reset_index(drop=True)

    phones = sorted(str(value) for value in valid["phone"].dropna().unique())
    if held_out_phone not in phones:
        raise ValueError(f"held-out phone {held_out_phone!r} not found; available phones: {phones}")
    train = valid[valid["phone"].astype(str) != held_out_phone].reset_index(drop=True)
    test = valid[valid["phone"].astype(str) == held_out_phone].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("static KNN train/test split is empty")

    ap_columns = list(db.access_point_columns)
    train_wifi = train[ap_columns].to_numpy(np.float32)
    test_wifi = test[ap_columns].to_numpy(np.float32)
    train_mag = train[list(MAG_COLUMNS)].to_numpy(np.float32)
    test_mag = test[list(MAG_COLUMNS)].to_numpy(np.float32)
    train_y = train[["x", "y"]].to_numpy(np.float32)
    test_y = test[["x", "y"]].to_numpy(np.float32)
    train_groups = train["phone"].astype(str).to_numpy()

    variants: dict[str, object] = {}
    prediction_rows = test[["x", "y", "phone", "mode", "scenario", "file"]].copy()
    curves: dict[str, np.ndarray] = {}
    for name, weights in {
        "wifi_only_knn": (1.0,),
        "magnetic_only_knn": (0.0,),
        "wifi_mag_hybrid_knn": HYBRID_WIFI_WEIGHTS,
    }.items():
        prediction, errors, report = _fit_static_variant(
            train_wifi, train_mag, train_y, train_groups,
            test_wifi, test_mag, test_y, weights=weights,
        )
        variants[name] = report
        curves[name] = errors
        prediction_rows[f"{name}_pred_x"] = prediction[:, 0]
        prediction_rows[f"{name}_pred_y"] = prediction[:, 1]
        prediction_rows[f"{name}_error_m"] = errors

    static_dir = output / "static_heldout_device"
    static_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_csv(static_dir / "predictions.csv", index=False)
    payload = {
        "protocol": "real_static_fingerprints_heldout_phone",
        "database": str(database),
        "held_out_phone": held_out_phone,
        "training_phones": sorted(str(v) for v in train["phone"].unique()),
        "modes": sorted(str(v) for v in valid["mode"].unique()),
        "train_visits": int(len(train)),
        "test_visits": int(len(test)),
        "access_points": int(len(ap_columns)),
        "magnetic_features": list(MAG_COLUMNS),
        "variants": variants,
    }
    (static_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    labels = {
        "wifi_only_knn": "Wi-Fi KNN",
        "magnetic_only_knn": "Magnetic KNN",
        "wifi_mag_hybrid_knn": "Wi-Fi + Magnetic KNN",
    }
    for key in ("wifi_only_knn", "magnetic_only_knn", "wifi_mag_hybrid_knn"):
        values = np.sort(curves[key])
        cdf = np.arange(1, len(values) + 1) / len(values)
        mean = float(curves[key].mean())
        ax.plot(values, cdf, linewidth=2.3, label=f"{labels[key]} ({mean:.2f} m)")
    ax.set_xlabel("Position error (m)", fontsize=14)
    ax.set_ylabel("CDF", fontsize=14)
    ax.set_title(f"Static fingerprint baseline - held-out {held_out_phone}", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(static_dir / "cdf.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return payload


def _trajectory_features(data: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, wifi, wifi_mask, mag, mag_logvar, mag_mask, target, _ = data
    mag_available = mag_mask[..., 0] > 0.5
    wifi_available = wifi_mask[..., 0] > 0.5
    mag_filled = mag.copy()
    # Before an 84-frame magnetic window exists, use the current Wi-Fi coordinate as a
    # neutral coordinate imputation and let the explicit mask identify missing magnetics.
    mag_filled[~mag_available] = wifi[~mag_available]
    logvar = np.clip(mag_logvar[..., 0], -6.0, 8.0)
    if mag_available.any():
        reference_logvar = float(np.median(logvar[mag_available]))
    else:
        reference_logvar = 0.0
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
    groups = np.repeat(np.arange(walks), bins)
    return features.reshape(walks * bins, channels), target.reshape(walks * bins, 2), groups


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _select_trajectory_k(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[int, list[dict[str, float | int]]]:
    splitter = GroupKFold(n_splits=5)
    records: list[dict[str, float | int]] = []
    for k in _valid_k_values(len(y)):
        fold_errors: list[float] = []
        for fit_index, validation_index in splitter.split(x, groups=groups):
            if k > len(fit_index):
                continue
            mean, std = _standardize_fit(x[fit_index])
            model = KNeighborsRegressor(
                n_neighbors=k, metric="euclidean", weights="uniform", algorithm="auto", n_jobs=-1
            )
            model.fit((x[fit_index] - mean) / std, y[fit_index])
            pred = model.predict((x[validation_index] - mean) / std)
            fold_errors.append(float(_euclidean_errors(y[validation_index], pred).mean()))
        if fold_errors:
            records.append({"k": int(k), "mean_group_cv_error_m": float(np.mean(fold_errors))})
    best = min(records, key=lambda row: row["mean_group_cv_error_m"])
    return int(best["k"]), records


def _fit_trajectory_knn(training: tuple[np.ndarray, ...], testing: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_train, y_train, train_groups = _trajectory_features(training)
    x_test, y_test, _ = _trajectory_features(testing)
    k, cv = _select_trajectory_k(x_train, y_train, train_groups)
    mean, std = _standardize_fit(x_train)
    model = KNeighborsRegressor(
        n_neighbors=k, metric="euclidean", weights="uniform", algorithm="auto", n_jobs=-1
    )
    model.fit((x_train - mean) / std, y_train)
    predicted_flat = model.predict((x_test - mean) / std)
    walks, bins = testing[6].shape[:2]
    predicted = predicted_flat.reshape(walks, bins, 2)
    errors_per_bin = np.linalg.norm(predicted - testing[6], axis=2)
    per_walk_mean = errors_per_bin.mean(axis=1)
    return per_walk_mean, predicted, {
        "k": k,
        "group_cv": cv,
        "features": [
            "wifi_fix_x", "wifi_fix_y", "magnetic_cnn_x", "magnetic_cnn_y",
            "magnetic_log_variance", "wifi_available", "magnetic_available",
        ],
        "uses_pdr": False,
        "uses_temporal_history": False,
        "metrics": summarize(per_walk_mean),
    }


def _plot_trajectory_cdf(
    output: Path,
    title: str,
    baseline_errors: np.ndarray,
    knn_errors: np.ndarray,
    dual_errors: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, values in (
        ("Wi-Fi-only KalmanNet", baseline_errors),
        ("Wi-Fi + Magnetic KNN", knn_errors),
        ("CNN Dual + relative variance", dual_errors),
    ):
        ordered = np.sort(values)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, linewidth=2.4, label=f"{label} ({values.mean():.2f} m)")
    ax.set_xlabel("Per-walk mean error (m)", fontsize=14)
    ax.set_ylabel("CDF", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)



def _trajectory_turn_score(target: np.ndarray) -> float:
    displacement = np.diff(np.asarray(target, dtype=float), axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    displacement = displacement[lengths > 1e-4]
    if len(displacement) < 2:
        return 0.0
    heading = np.unwrap(np.arctan2(displacement[:, 1], displacement[:, 0]))
    return float(np.abs(np.diff(heading)).sum())


def _select_representative_walk(
    baseline_errors: np.ndarray,
    dual_errors: np.ndarray,
    target: np.ndarray,
) -> tuple[int, dict[str, float | int | str]]:
    """Pick a typical-but-geometrically-informative walk without cherry-picking.

    Candidate walks are restricted to the interquartile range of per-walk
    improvements (Wi-Fi-only error minus weighted-Dual error). Within that
    central 50% we choose the path with the largest accumulated heading change,
    so the figure contains several corridor turns while remaining a typical
    performance case rather than a best-case example.
    """
    improvement = np.asarray(baseline_errors) - np.asarray(dual_errors)
    q25, q75 = np.percentile(improvement, [25, 75])
    candidates = np.flatnonzero((improvement >= q25) & (improvement <= q75))
    if not len(candidates):
        candidates = np.arange(len(improvement))
    scores = np.asarray([_trajectory_turn_score(path) for path in target], dtype=float)
    index = int(candidates[np.argmax(scores[candidates])])
    return index, {
        "selection_rule": "max turn score among walks in the interquartile improvement range",
        "walk_index_zero_based": index,
        "improvement_q25_m": float(q25),
        "improvement_q75_m": float(q75),
        "selected_improvement_m": float(improvement[index]),
        "selected_turn_score_rad": float(scores[index]),
    }


def _plot_representative_trajectory(
    output: Path,
    index: int,
    cache: dict[str, dict[str, np.ndarray]],
    corridor_coordinates: np.ndarray,
) -> None:
    """Plot an actual degraded-Wi-Fi test walk using current model outputs."""
    data = cache["degraded"]
    start = data["start"][index]
    truth = data["target"][index]
    pdr = start[None, :] + np.cumsum(data["motion"][index], axis=0)
    wifi = start[None, :] + data["wifi_prediction"][index]
    dual = start[None, :] + data["dual_prediction"][index]
    wifi_updates = data["wifi_mask"][index, :, 0] > 0.5

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    corridor = np.asarray(corridor_coordinates)
    ax.scatter(corridor[:, 0], corridor[:, 1], s=10, marker="s", color="0.88", zorder=0)
    ax.plot(truth[:, 0], truth[:, 1], "k--", linewidth=2.6, label="Ground truth", zorder=5)
    ax.plot(pdr[:, 0], pdr[:, 1], color="0.55", linestyle=":", linewidth=2.0,
            label="PDR only", zorder=2)
    ax.plot(wifi[:, 0], wifi[:, 1], linewidth=2.2, label="Wi-Fi-only KalmanNet", zorder=3)
    ax.plot(dual[:, 0], dual[:, 1], linewidth=2.5,
            label="CNN Dual + relative variance", zorder=4)
    ax.scatter(
        truth[wifi_updates, 0], truth[wifi_updates, 1],
        s=28, facecolors="none", edgecolors="0.25", linewidths=1.0,
        label="Wi-Fi update time", zorder=6,
    )
    ax.scatter(truth[0, 0], truth[0, 1], s=55, marker="o", facecolors="white",
               edgecolors="black", linewidths=1.5, zorder=7)
    ax.scatter(truth[-1, 0], truth[-1, 1], s=60, marker="X", color="black", zorder=7)

    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_title("Representative degraded-Wi-Fi test trajectory", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.28)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=10, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)

def run_trajectory(
    database: Path,
    wifi_checkpoint: Path,
    magnetic_checkpoint: Path,
    output: Path,
    *,
    device: torch.device,
    train_walks: int,
    test_walks: int,
    epochs: int,
    bins: int,
) -> dict[str, object]:
    env = setup_environment(database, wifi_checkpoint, magnetic_checkpoint, device)
    regimes = {
        "full": ("Full Wi-Fi (1 Hz)", 1.0, 0.0),
        "degraded": ("Degraded Wi-Fi (5 s, 40% AP drop)", 5.0, 0.4),
    }
    trajectory_dir = output / "trajectory_protocol"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "same_synthetic_walk_protocol_as_current_kalmannet",
        "train_walks": train_walks,
        "test_walks": test_walks,
        "fusion_epochs": epochs,
        "bins": bins,
        "regimes": {},
    }
    trajectory_cache: dict[str, dict[str, np.ndarray]] = {}

    for key, (name, wifi_period, dropout) in regimes.items():
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        training = make_dataset(
            train_walks, seed=1, env=env, device=device,
            wifi_period_s=wifi_period, ap_dropout=dropout, bins=bins,
        )
        testing = make_dataset(
            test_walks, seed=2, env=env, device=device,
            wifi_period_s=wifi_period, ap_dropout=dropout, bins=bins,
        )

        print("  tuning/fitting non-temporal Wi-Fi + magnetic KNN")
        knn_errors, knn_prediction, knn_report = _fit_trajectory_knn(training, testing)

        torch.manual_seed(SEED)
        baseline = WiFiOnlyKalmanNet(hidden_size=FUSION_HIDDEN_SIZE).to(device)
        print("  training Wi-Fi-only KalmanNet for matched CDF")
        baseline, _ = train_filter(baseline, training, device, epochs=epochs, uses_magnetic=False)
        baseline_errors, baseline_prediction = evaluate_filter(
            baseline, testing, device, uses_magnetic=False
        )

        training_mag_available = training[5][..., 0] > 0.5
        reference_logvar = float(np.median(training[4][..., 0][training_mag_available]))
        torch.manual_seed(SEED)
        dual = CNNMagneticDualKalmanNet(
            hidden_size=FUSION_HIDDEN_SIZE,
            magnetic_reference_log_variance=reference_logvar,
        ).to(device)
        print("  training variance-weighted CNN DualKalmanNet for matched CDF")
        dual, _ = train_filter(dual, training, device, epochs=epochs, uses_magnetic=True)
        dual_errors, dual_prediction = evaluate_filter(dual, testing, device, uses_magnetic=True)

        regime_dir = trajectory_dir / key
        regime_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            regime_dir / "predictions_and_errors.npz",
            wifi_only_per_walk_error=baseline_errors,
            knn_per_walk_error=knn_errors,
            dual_weighted_per_walk_error=dual_errors,
            wifi_only_prediction=baseline_prediction,
            knn_prediction=knn_prediction,
            dual_weighted_prediction=dual_prediction,
            target=testing[6],
            start=testing[7],
            motion=testing[0],
            wifi_mask=testing[2],
            magnetic_mask=testing[5],
        )
        _plot_trajectory_cdf(
            regime_dir / "cdf.png", name, baseline_errors, knn_errors, dual_errors
        )
        trajectory_cache[key] = {
            "target": testing[6],
            "start": testing[7],
            "motion": testing[0],
            "wifi_mask": testing[2],
            "wifi_prediction": baseline_prediction,
            "dual_prediction": dual_prediction,
            "wifi_error": baseline_errors,
            "dual_error": dual_errors,
        }

        report["regimes"][key] = {
            "name": name,
            "wifi_period_s": wifi_period,
            "ap_dropout": dropout,
            "wifi_only_kalmannet": summarize(baseline_errors),
            "wifi_mag_knn": knn_report,
            "cnn_dual_relative_variance": summarize(dual_errors),
            "magnetic_reference_log_variance_training": reference_logvar,
        }

    if "degraded" in trajectory_cache:
        degraded = trajectory_cache["degraded"]
        representative_index, selection = _select_representative_walk(
            degraded["wifi_error"], degraded["dual_error"], degraded["target"]
        )
        selection.update(
            {
                "wifi_only_mean_error_m": float(degraded["wifi_error"][representative_index]),
                "weighted_dual_mean_error_m": float(degraded["dual_error"][representative_index]),
                "regime": "Degraded Wi-Fi (5 s, 40% AP drop)",
            }
        )
        report["representative_trajectory"] = selection
        (trajectory_dir / "representative_trajectory.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        _plot_representative_trajectory(
            trajectory_dir / "representative_trajectory.png",
            representative_index,
            trajectory_cache,
            env.corridor.coordinates,
        )

    (trajectory_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--wifi-checkpoint", type=Path, default=DEFAULT_WIFI_CHECKPOINT)
    parser.add_argument("--magnetic-checkpoint", type=Path, default=DEFAULT_MAG_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks" / "knn" / "current_results")
    parser.add_argument("--held-out-phone", default=HELD_OUT_PHONE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--protocol", choices=("static", "trajectory", "both"), default="both")
    parser.add_argument("--train-walks", type=int, default=DEFAULT_TRAIN_WALKS)
    parser.add_argument("--test-walks", type=int, default=DEFAULT_TEST_WALKS)
    parser.add_argument("--fusion-epochs", type=int, default=DEFAULT_FUSION_EPOCHS)
    parser.add_argument("--bins", type=int, default=DEFAULT_T_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    final: dict[str, object] = {}
    if args.protocol in {"static", "both"}:
        print("Running real static held-out-device KNN benchmark...")
        final["static"] = run_static(args.database, args.output, args.held_out_phone)
    if args.protocol in {"trajectory", "both"}:
        print("Running matched trajectory-protocol KNN benchmark...")
        final["trajectory"] = run_trajectory(
            args.database,
            args.wifi_checkpoint,
            args.magnetic_checkpoint,
            args.output,
            device=device,
            train_walks=args.train_walks,
            test_walks=args.test_walks,
            epochs=args.fusion_epochs,
            bins=args.bins,
        )
    (args.output / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
