from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep the benchmark source consistent with the rendered figure.
source = Path("benchmarks/knn/wifi_mag_knn.py")
text = source.read_text(encoding="utf-8")
old = '''    fig, ax = plt.subplots(figsize=(8.4, 4.8))\n    corridor = np.asarray(corridor_coordinates)\n    ax.scatter(corridor[:, 0], corridor[:, 1], s=10, marker="s", color="0.88", zorder=0)\n    ax.plot(truth[:, 0], truth[:, 1], "k--", linewidth=2.6, label="Ground truth", zorder=5)\n'''
new = '''    fig, ax = plt.subplots(figsize=(8.4, 4.8))\n    combined = np.vstack((truth, pdr, wifi, dual))\n    x_min, y_min = combined.min(axis=0) - 3.0\n    x_max, y_max = combined.max(axis=0) + 3.0\n    corridor = np.asarray(corridor_coordinates)\n    local = corridor[\n        (corridor[:, 0] >= x_min) & (corridor[:, 0] <= x_max)\n        & (corridor[:, 1] >= y_min) & (corridor[:, 1] <= y_max)\n    ]\n    ax.scatter(local[:, 0], local[:, 1], s=12, marker="s", color="0.88", zorder=0)\n    ax.plot(truth[:, 0], truth[:, 1], "k--", linewidth=2.6, label="Ground truth", zorder=5)\n'''
if old not in text:
    raise SystemExit("trajectory plotting source block not found")
text = text.replace(old, new, 1)
old_axis = '''    ax.grid(alpha=0.28)\n    ax.set_aspect("equal", adjustable="datalim")\n    ax.legend(fontsize=10, ncol=2, loc="best")\n'''
new_axis = '''    ax.grid(alpha=0.28)\n    ax.set_xlim(x_min, x_max)\n    ax.set_ylim(y_min, y_max)\n    ax.set_aspect("equal", adjustable="box")\n    ax.legend(fontsize=10, ncol=2, loc="best")\n'''
if old_axis not in text:
    raise SystemExit("trajectory axis block not found")
text = text.replace(old_axis, new_axis, 1)
source.write_text(text, encoding="utf-8")

# Regenerate the already-tested representative figure without retraining.
root = Path("benchmarks/knn/current_results/trajectory_protocol")
meta = json.loads((root / "representative_trajectory.json").read_text(encoding="utf-8"))
index = int(meta["walk_index_zero_based"])
data = np.load(root / "degraded" / "predictions_and_errors.npz")
start = data["start"][index]
truth = data["target"][index]
pdr = start[None, :] + np.cumsum(data["motion"][index], axis=0)
wifi = start[None, :] + data["wifi_only_prediction"][index]
dual = start[None, :] + data["dual_weighted_prediction"][index]
wifi_updates = data["wifi_mask"][index, :, 0] > 0.5

nodes = pd.read_csv("data/processed/fingerprint_db/it_engineering/nodes.csv", usecols=["x", "y"])
corridor = nodes[["x", "y"]].round(1).drop_duplicates().to_numpy(float)
combined = np.vstack((truth, pdr, wifi, dual))
x_min, y_min = combined.min(axis=0) - 3.0
x_max, y_max = combined.max(axis=0) + 3.0
local = corridor[(corridor[:,0] >= x_min) & (corridor[:,0] <= x_max) & (corridor[:,1] >= y_min) & (corridor[:,1] <= y_max)]

fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.scatter(local[:,0], local[:,1], s=12, marker="s", color="0.88", zorder=0)
ax.plot(truth[:,0], truth[:,1], "k--", linewidth=2.6, label="Ground truth", zorder=5)
ax.plot(pdr[:,0], pdr[:,1], color="0.55", linestyle=":", linewidth=2.0, label="PDR only", zorder=2)
ax.plot(wifi[:,0], wifi[:,1], linewidth=2.2, label="Wi-Fi-only KalmanNet", zorder=3)
ax.plot(dual[:,0], dual[:,1], linewidth=2.5, label="CNN Dual + relative variance", zorder=4)
ax.scatter(truth[wifi_updates,0], truth[wifi_updates,1], s=28, facecolors="none", edgecolors="0.25", linewidths=1.0, label="Wi-Fi update time", zorder=6)
ax.scatter(truth[0,0], truth[0,1], s=55, marker="o", facecolors="white", edgecolors="black", linewidths=1.5, zorder=7)
ax.scatter(truth[-1,0], truth[-1,1], s=60, marker="X", color="black", zorder=7)
ax.set_xlabel("x (m)", fontsize=14)
ax.set_ylabel("y (m)", fontsize=14)
ax.set_title("Representative degraded-Wi-Fi test trajectory", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.28)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=10, ncol=2, loc="best")
fig.tight_layout()
fig.savefig(root / "representative_trajectory.png", dpi=240, bbox_inches="tight")
plt.close(fig)
