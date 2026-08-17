from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks' / 'knn' / 'current_results' / 'trajectory_protocol'

# Keep the reproducible benchmark source terminology consistent for future reruns.
source_path = ROOT / 'benchmarks' / 'knn' / 'wifi_mag_knn.py'
source = source_path.read_text(encoding='utf-8')
source = source.replace('CNN Dual + relative variance', 'CNN Dual + relative uncertainty')
source = source.replace('magnetic-CNN fix/log-variance', 'magnetic-CNN fix/log-uncertainty score')
source_path.write_text(source, encoding='utf-8')

# Regenerate trajectory CDFs from saved per-walk errors only; no model retraining.
for key, title in {
    'full': 'Full Wi-Fi (1 Hz)',
    'degraded': 'Degraded Wi-Fi (5 s, 40% AP drop)',
}.items():
    d = np.load(RESULTS / key / 'predictions_and_errors.npz')
    curves = (
        ('Wi-Fi-only KalmanNet', d['wifi_only_per_walk_error']),
        ('Wi-Fi + Magnetic KNN', d['knn_per_walk_error']),
        ('CNN Dual + relative uncertainty', d['dual_weighted_per_walk_error']),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, values in curves:
        values = np.asarray(values, dtype=float)
        ordered = np.sort(values)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, linewidth=2.4, label=f'{label} ({values.mean():.2f} m)')
    ax.set_xlabel('Per-walk mean error (m)', fontsize=14)
    ax.set_ylabel('CDF', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / key / 'cdf.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

# Regenerate the representative degraded trajectory from the saved predictions.
metrics = json.loads((RESULTS / 'metrics.json').read_text(encoding='utf-8'))
index = int(metrics['representative_trajectory']['walk_index_zero_based'])
d = np.load(RESULTS / 'degraded' / 'predictions_and_errors.npz')
start = d['start'][index]
truth = d['target'][index]
pdr = start[None, :] + np.cumsum(d['motion'][index], axis=0)
wifi = start[None, :] + d['wifi_only_prediction'][index]
dual = start[None, :] + d['dual_weighted_prediction'][index]
wifi_updates = d['wifi_mask'][index, :, 0] > 0.5

# Recover surveyed-node coordinates directly from the processed database for map context.
coords = set()
nodes_path = ROOT / 'data' / 'processed' / 'fingerprint_db' / 'it_engineering' / 'nodes.csv'
with nodes_path.open(newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        try:
            coords.add((round(float(row['x']), 1), round(float(row['y']), 1)))
        except (KeyError, TypeError, ValueError):
            continue
corridor = np.asarray(sorted(coords), dtype=float)

fig, ax = plt.subplots(figsize=(8.4, 4.8))
combined = np.vstack((truth, pdr, wifi, dual))
x_min, y_min = combined.min(axis=0) - 3.0
x_max, y_max = combined.max(axis=0) + 3.0
if corridor.size:
    local = corridor[
        (corridor[:, 0] >= x_min) & (corridor[:, 0] <= x_max)
        & (corridor[:, 1] >= y_min) & (corridor[:, 1] <= y_max)
    ]
    ax.scatter(local[:, 0], local[:, 1], s=12, marker='s', color='0.88', zorder=0)
ax.plot(truth[:, 0], truth[:, 1], 'k--', linewidth=2.6, label='Ground truth', zorder=5)
ax.plot(pdr[:, 0], pdr[:, 1], color='0.55', linestyle=':', linewidth=2.0, label='PDR only', zorder=2)
ax.plot(wifi[:, 0], wifi[:, 1], linewidth=2.2, label='Wi-Fi-only KalmanNet', zorder=3)
ax.plot(dual[:, 0], dual[:, 1], linewidth=2.5, label='CNN Dual + relative uncertainty', zorder=4)
ax.scatter(truth[wifi_updates, 0], truth[wifi_updates, 1], s=28, facecolors='none', edgecolors='0.25', linewidths=1.0, label='Wi-Fi update time', zorder=6)
ax.scatter(truth[0, 0], truth[0, 1], s=55, marker='o', facecolors='white', edgecolors='black', linewidths=1.5, zorder=7)
ax.scatter(truth[-1, 0], truth[-1, 1], s=60, marker='X', color='black', zorder=7)
ax.set_xlabel('x (m)', fontsize=14)
ax.set_ylabel('y (m)', fontsize=14)
ax.set_title('Representative degraded-Wi-Fi test trajectory', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.grid(alpha=0.28)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal', adjustable='box')
ax.legend(fontsize=10, ncol=2, loc='best')
fig.tight_layout()
fig.savefig(RESULTS / 'representative_trajectory.png', dpi=240, bbox_inches='tight')
plt.close(fig)
