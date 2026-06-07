"""
Stage 3 (real-sensor demo): causal IMU pedestrian dead-reckoning on REAL IT
Engineering continuous walks, overlaid on the real corridor map.

Why only this: the continuous recordings contain IMU + magnetometer but NO WiFi
(WiFi in this dataset is always static node scans), and they carry only a START
coordinate -- no per-frame trajectory ground truth. So we cannot run the WiFi
correction or score against truth on real walks. What IS real and demonstrable:
the causal motion model running on genuine sensors.

Pipeline (strictly causal):
  - Online step detection on real |acc|; each step advances the position by a
    fixed step length along heading = Orn_z + offset (world frame).
  - Anchored at the real start node from the file.
  - The unknown map<->compass rotation offset is resolved by MAP MATCHING: choose
    the offset whose dead-reckoned track sits closest to the surveyed corridor
    nodes (an honest map constraint, the only correction available here).

Qualitative assessment: does the track follow the corridor geometry, and how far
does pure DR drift off the surveyed nodes?
"""
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from stage3_ekf_fusion import StepDetector, FS
DB_DIR = "../Datasets/fingerprint_db/it_engineering"
STEP_LEN = 0.68  # m, calibrated earlier


def load_continuous(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    need = ["Acc_x", "Acc_y", "Acc_z", "Orn_z"]
    if not all(c in df.columns for c in need):
        return None
    xc = "X-cord" if "X-cord" in df.columns else None
    yc = "Y-cord" if "Y-cord" in df.columns else None
    start = (float(df[xc].dropna().iloc[0]), float(df[yc].dropna().iloc[0])) if xc and yc else (90.0, 24.0)
    return df, start


def pdr_track(df, start, offset, step_len=STEP_LEN):
    acc = df[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(float)
    accmag = np.linalg.norm(acc, axis=1)
    head = df["Orn_z"].to_numpy(float) + offset
    det = StepDetector()
    pos = np.array(start, float)
    pts = [pos.copy()]
    nsteps = 0
    for t in range(len(df)):
        if det.update(accmag[t]):
            pos = pos + step_len * np.array([np.cos(head[t]), np.sin(head[t])])
            nsteps += 1
        pts.append(pos.copy())
    return np.array(pts), nsteps


def best_offset(df, start, tree):
    """Resolve the map<->compass rotation by minimising mean distance to corridor nodes."""
    best = (1e9, 0.0, None)
    for off in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        trk, _ = pdr_track(df, start, off)
        d, _ = tree.query(trk)
        m = d.mean()
        if m < best[0]:
            best = (m, off, trk)
    return best


def main():
    nodes = pd.read_csv(f"{DB_DIR}/nodes.csv")[["x", "y"]].round(1).drop_duplicates().to_numpy()
    tree = cKDTree(nodes)

    files = sorted(glob.glob("../Datasets/Magnetic field dataset/Continuous Data/"
                             "IT Engineering/Navigation/**/IMU_*.csv", recursive=True))
    walks = []
    for f in files:
        r = load_continuous(f)
        if r is None:
            continue
        df, start = r
        if len(df) < 300:
            continue
        walks.append((f, df, start))
    print(f"Usable real IT continuous walks: {len(walks)}")

    # demo on up to 4 walks
    sel = walks[:4]
    fig, axes = plt.subplots(1, len(sel), figsize=(5 * len(sel), 5))
    if len(sel) == 1:
        axes = [axes]
    for ax, (f, df, start) in zip(axes, sel):
        mdist, off, trk = best_offset(df, start, tree)
        _, nsteps = pdr_track(df, start, off)
        dist = nsteps * STEP_LEN
        d2node, _ = tree.query(trk)
        lab = f.split("Navigation")[1][:24].replace("\\", "/")
        print(f"  {lab:26s} steps={nsteps:4d} dist={dist:5.1f}m "
              f"mean-to-corridor={mdist:.2f}m p90={np.percentile(d2node,90):.2f}m offset={off:.2f}rad")

        ax.scatter(nodes[:, 0], nodes[:, 1], s=8, c="lightgray", label="corridor nodes")
        ax.plot(trk[:, 0], trk[:, 1], "g-", lw=1.2, label=f"PDR ({nsteps} steps)")
        ax.scatter([start[0]], [start[1]], c="gold", s=120, ec="k", zorder=5, label="start")
        ax.scatter([trk[-1, 0]], [trk[-1, 1]], c="red", s=80, marker="X", zorder=5, label="end")
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(fontsize=6)
        ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    plt.suptitle("Stage 3 real-sensor demo: causal IMU dead-reckoning on real IT walks "
                 "(map-aligned, NO WiFi correction available)", fontsize=11)
    plt.tight_layout()
    plt.savefig("../Datasets/stage3_real_pdr_demo.png", dpi=170, bbox_inches="tight")
    print("\nSaved -> Datasets/stage3_real_pdr_demo.png")


if __name__ == "__main__":
    main()
