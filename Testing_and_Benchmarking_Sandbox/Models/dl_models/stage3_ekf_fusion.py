"""
Stage 3a: Causal EKF fusion of WiFi-heatmap anchor + IMU pedestrian dead-reckoning.

Real-time / causal by construction:
  - PREDICTION (every frame, IMU): online step detection on |acc|; each step advances
    position by step_length L along heading = Orn_z + offset (world frame). Drifts slowly.
  - MEASUREMENT (~1 Hz, WiFi): the walk's true position -> nearest surveyed node's REAL
    WiFi scan -> Stage-2 heatmap net -> position fix + a covariance R read directly from
    the heatmap's spread (honest measurement noise). Drift-free but noisy.
  - A standard 2D Kalman update fuses them. Almost no trainable params -> cannot memorise
    the route. The few calibration scalars (heading offset, L, Q, R-scale) are fit on the
    TRAIN walks (A8/G7/S8) and frozen for the S9+ test (forward + reversed).

The environment (heatmap) model is trained on static visits EXCLUDING the S9+ device,
so the test exercises a new device + new walk through a KNOWN environment.
"""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from stage2_wifi_heatmap import (Grid, WifiHeatmapNet, encode_wifi, soft_argmax,
                                 DB_DIR, KEEP_MODES)
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

FS = 16.7                       # sample rate (Hz)
WIFI_PERIOD_S = 1.0             # WiFi scan cadence
TRAIN_PHONES = ["A8", "G7", "S8"]
TEST_PHONE = "S9+"


# --------------------------------------------------------------------------- #
# Environment model + WiFi map (built from the static DB, excluding test device)
# --------------------------------------------------------------------------- #
def build_env(df_db, ap_cols, grid):
    """Train the WiFi heatmap net on non-test-device static visits."""
    sub = df_db[df_db["has_wifi"] & df_db["mode"].isin(KEEP_MODES) &
                (df_db["phone"] != TEST_PHONE)].reset_index(drop=True)
    X = encode_wifi(sub[ap_cols].to_numpy(float))
    Y = np.stack([grid.gaussian_target(x, y) for x, y in sub[["x", "y"]].values])
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)), batch_size=64, shuffle=True)
    net = WifiHeatmapNet(len(ap_cols), grid.n_cells)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(70):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            logp = torch.log_softmax(net(xb), 1)
            loss = torch.sum(yb * (torch.log(yb + 1e-9) - logp), 1).mean()
            loss.backward(); opt.step()
    net.eval()
    print(f"  env model trained on {len(sub)} non-{TEST_PHONE} visits")
    return net


def build_wifi_map(df_db, ap_cols):
    """Per-node averaged real WiFi scan -> (node_xy, scan_matrix, kdtree)."""
    sub = df_db[df_db["has_wifi"] & df_db["mode"].isin(KEEP_MODES)]
    g = sub.groupby([sub["x"].round(1), sub["y"].round(1)])
    xy, scans = [], []
    for (nx, ny), idx in g.groups.items():
        rows = sub.loc[idx, ap_cols].to_numpy(float)
        # average only over APs actually seen (-100 = absent)
        present = rows > -100
        avg = np.where(present.any(0), np.where(present, rows, np.nan), -100.0)
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(np.where(present, rows, np.nan), axis=0)
        avg = np.where(np.isnan(avg), -100.0, avg)
        xy.append([nx, ny]); scans.append(avg)
    xy = np.array(xy); scans = np.array(scans)
    return xy, scans, cKDTree(xy)


def heatmap_fix(net, scan_vec, grid, coords_t):
    """One WiFi scan -> (position fix (2,), covariance R (2,2)) from heatmap spread."""
    x = torch.tensor(encode_wifi(scan_vec[None, :]), dtype=torch.float32)
    with torch.no_grad():
        p = torch.softmax(net(x), 1).numpy()[0]            # (n_cells,)
    c = grid.coords                                        # (n_cells, 2)
    mu = (p[:, None] * c).sum(0)
    d = c - mu
    R = (p[:, None, None] * np.einsum("ni,nj->nij", d, d)).sum(0)
    return mu.astype(float), R.astype(float)


# --------------------------------------------------------------------------- #
# Causal IMU pedestrian dead-reckoning
# --------------------------------------------------------------------------- #
class StepDetector:
    def __init__(self, fs=FS, thresh=0.6, refractory_s=0.3):
        self.refr = int(refractory_s * fs); self.thresh = thresh
        self.mean = 9.81; self.i = 0; self.last = -999

    def update(self, accmag):
        self.i += 1
        self.mean = 0.98 * self.mean + 0.02 * accmag
        hp = accmag - self.mean
        if hp > self.thresh and (self.i - self.last) > self.refr:
            self.last = self.i
            return True
        return False


def pdr_controls(df, heading_offset, step_len):
    """Per-frame displacement control inputs u_t (T,2) from causal step detection."""
    acc = df[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(float)
    accmag = np.linalg.norm(acc, axis=1)
    head = df["Orn_z"].to_numpy(float) + heading_offset
    det = StepDetector()
    u = np.zeros((len(df), 2))
    for t in range(len(df)):
        if det.update(accmag[t]):
            u[t] = [step_len * np.cos(head[t]), step_len * np.sin(head[t])]
    return u


# --------------------------------------------------------------------------- #
# EKF
# --------------------------------------------------------------------------- #
def run_ekf(df, net, wmap, grid, coords_t, heading_offset, step_len,
            q_step, r_scale, wifi_period=WIFI_PERIOD_S):
    xy, scans, tree = wmap
    u = pdr_controls(df, heading_offset, step_len)
    pos_true = df[["True_X", "True_Y"]].to_numpy(float)
    n = len(df)
    fix_stride = max(1, int(wifi_period * FS))

    x = pos_true[0].copy()           # initialise at start (or first WiFi fix)
    P = np.eye(2) * 4.0
    Q_frame = np.eye(2) * 0.01
    track = np.zeros((n, 2))
    for t in range(n):
        # --- predict ---
        x = x + u[t]
        P = P + Q_frame + (q_step * np.eye(2) if np.any(u[t]) else 0.0)
        # --- measurement update at WiFi cadence ---
        if t % fix_stride == 0:
            _, idx = tree.query(pos_true[t])          # nearest surveyed node
            z, R = heatmap_fix(net, scans[idx], grid, coords_t)
            R = R * r_scale + np.eye(2) * 0.5
            S = P + R
            K = P @ np.linalg.inv(S)
            x = x + K @ (z - x)
            P = (np.eye(2) - K) @ P
        track[t] = x
    err = np.linalg.norm(track - pos_true, axis=1)
    return track, err


# --------------------------------------------------------------------------- #
# Main: calibrate on train walks, freeze, evaluate on S9+ (fwd + reversed)
# --------------------------------------------------------------------------- #
def load_walk(phone):
    return pd.read_csv(f"../Datasets/Continuous_Fused_{phone}.csv")


def fit_heading_offset(df):
    dx = np.gradient(df["True_X"].values); dy = np.gradient(df["True_Y"].values)
    th = np.arctan2(dy, dx); o = df["Orn_z"].to_numpy(float)
    return float(np.arctan2(np.mean(np.sin(th - o)), np.mean(np.cos(th - o))))


def main():
    print("=" * 64); print("Stage 3a: Causal EKF (WiFi heatmap + IMU PDR)"); print("=" * 64)
    df_db = pd.read_csv(f"{DB_DIR}/nodes.csv")
    ap_cols = json.load(open(f"{DB_DIR}/bssid_vocab.json"))["ap_columns"]
    grid = Grid(df_db["x"].values, df_db["y"].values)
    coords_t = torch.tensor(grid.coords, dtype=torch.float32)

    net = build_env(df_db, ap_cols, grid)
    wmap = build_wifi_map(df_db, ap_cols)

    # --- calibrate on train walks ---
    train = {p: load_walk(p) for p in TRAIN_PHONES}
    head_off = float(np.mean([fit_heading_offset(d) for d in train.values()]))
    # step length: match integrated steps to true path length, averaged over train
    Ls = []
    for d in train.values():
        u0 = pdr_controls(d, head_off, 1.0)
        nsteps = np.count_nonzero(np.any(u0 != 0, 1))
        plen = np.sum(np.linalg.norm(np.diff(d[["True_X", "True_Y"]].to_numpy(), axis=0), axis=1))
        if nsteps > 0:
            Ls.append(plen / nsteps)
    step_len = float(np.mean(Ls))
    print(f"  calibrated heading_offset={head_off:.3f} rad, step_len={step_len:.2f} m")

    # small grid search over filter noise on train
    best = (1e9, None)
    for q_step in [0.2, 0.5, 1.0]:
        for r_scale in [0.5, 1.0, 2.0]:
            errs = []
            for d in train.values():
                _, e = run_ekf(d, net, wmap, grid, coords_t, head_off, step_len, q_step, r_scale)
                errs.append(e.mean())
            m = np.mean(errs)
            if m < best[0]:
                best = (m, (q_step, r_scale))
    q_step, r_scale = best[1]
    print(f"  best filter params: q_step={q_step}, r_scale={r_scale} (train MAE={best[0]:.2f}m)")

    # --- evaluate on S9+ forward + reversed ---
    print("\n--- TEST: S9+ ---")
    dft = load_walk(TEST_PHONE)
    track_f, err_f = run_ekf(dft, net, wmap, grid, coords_t, head_off, step_len, q_step, r_scale)
    print(f"FORWARD : MAE={err_f.mean():.2f}m  median={np.median(err_f):.2f}m  p90={np.percentile(err_f,90):.2f}m  max={err_f.max():.2f}m")

    dfr = dft.iloc[::-1].reset_index(drop=True)
    track_r, err_r = run_ekf(dfr, net, wmap, grid, coords_t, head_off, step_len, q_step, r_scale)
    print(f"REVERSED: MAE={err_r.mean():.2f}m  median={np.median(err_r):.2f}m  p90={np.percentile(err_r,90):.2f}m  max={err_r.max():.2f}m")

    # plot
    pt = dft[["True_X", "True_Y"]].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for a, (trk, e, lab, pth) in zip(ax, [(track_f, err_f, "FORWARD", pt),
                                          (track_r, err_r, "REVERSED", dfr[["True_X","True_Y"]].to_numpy())]):
        a.plot(pth[:, 0], pth[:, 1], "b-", alpha=0.5, label="truth")
        a.plot(trk[:, 0], trk[:, 1], "g-", lw=1.5, label=f"EKF (MAE={e.mean():.2f}m)")
        a.scatter([pth[0,0]], [pth[0,1]], c="gold", s=120, ec="k", zorder=5, label="start")
        a.set_title(f"S9+ {lab}"); a.set_xlabel("X"); a.set_ylabel("Y")
        a.legend(); a.grid(alpha=0.3); a.set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig("../Datasets/stage3_ekf_fusion.png", dpi=180, bbox_inches="tight")
    print("\nSaved -> Datasets/stage3_ekf_fusion.png")


if __name__ == "__main__":
    main()
