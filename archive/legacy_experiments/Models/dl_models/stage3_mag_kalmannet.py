"""
Magnetic-augmented KalmanNet: add the magnetic field as a SECOND measurement channel.

Magnetic gives a scalar field value per location, not a position. We turn it into a
position correction the KalmanNet way:
  - build a DEVICE-INVARIANT magnetic anomaly map A(x,y) from the static DB. Hard-iron
    bias is an additive per-device constant, so it cancels in the spatial anomaly
    (feature minus that device's building mean) -> the anomaly field is device-neutral
    and matchable by any phone after online calibration.
  - at runtime, innovation_mag = obs_anomaly - A(x_pred); the field gradient grad A(x_pred)
    tells which direction reduces the mismatch. Both are fed to the GRU, which emits a
    learned vector gain k_mag so the correction is  k_mag * innovation_mag  (the network
    learns to align it with the gradient). Over a sequence the recurrent state accumulates
    these constraints and disambiguates the (individually weak) magnetic signal.

State update per step (causal):
  x_pred = x + u_t
  x = x_pred + mask_wifi * (K_wifi @ innov_wifi)        # WiFi (sparse, ~1 Hz)
            + mask_mag  * (k_mag  *  innov_mag)          # magnetic (dense, every frame)

We compare WiFi+IMU vs WiFi+IMU+Mag under FULL and DEGRADED WiFi, since magnetic should
help most where WiFi is sparse/dropped.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from stage3_synthetic_eval import (setup_env, corridor_graph, sample_path, synth_walk,
                                   FS, STEP_LEN)
from stage3_ekf_fusion import StepDetector, heatmap_fix

DB = "../Datasets/fingerprint_db/it_engineering"
KEEP = {"Navigation", "Call listening", "Swinging"}
T_BINS = 160


# --------------------------------------------------------------------------- #
# Device-invariant magnetic anomaly map
# --------------------------------------------------------------------------- #
def build_mag_map(grid):
    df = pd.read_csv(f"{DB}/nodes.csv")
    df = df[df["mode"].isin(KEEP)].dropna(subset=["magN_mean"]).reset_index(drop=True)
    # de-bias per device: subtract each phone's building-wide mean (removes hard-iron DC)
    df["anom"] = df["magN_mean"] - df.groupby("phone")["magN_mean"].transform("mean")
    node = df.groupby([df["x"].round(1), df["y"].round(1)])["anom"].mean()
    nstd = df.groupby([df["x"].round(1), df["y"].round(1)])["anom"].std().median()
    nxy = np.array([list(k) for k in node.index]); nval = node.values

    # interpolate onto the env grid (linear, nearest-fill outside hull)
    lin = griddata(nxy, nval, grid.coords, method="linear")
    nn = griddata(nxy, nval, grid.coords, method="nearest")
    vals = np.where(np.isnan(lin), nn, lin).reshape(grid.nx, grid.ny)
    gx, gy = np.gradient(vals, grid.cell)  # field gradient per axis

    spatial_std = np.nanstd(nval)
    print(f"  magnetic anomaly map: spatial std={spatial_std:.2f} uT, "
          f"within-node noise std={nstd:.2f} uT, SNR={spatial_std/max(nstd,1e-6):.2f}")
    return (torch.tensor(vals, dtype=torch.float32),
            torch.tensor(gx, dtype=torch.float32),
            torch.tensor(gy, dtype=torch.float32),
            float(nstd), grid)


def bilinear(gridT, x, grid):
    """Sample a (nx,ny) grid at world points x=(B,2). Returns (B,)."""
    ix = (x[:, 0] - grid.x0) / grid.cell
    iy = (x[:, 1] - grid.y0) / grid.cell
    ix = ix.clamp(0, grid.nx - 1.001); iy = iy.clamp(0, grid.ny - 1.001)
    x0 = ix.floor().long(); y0 = iy.floor().long()
    x1 = (x0 + 1).clamp(max=grid.nx - 1); y1 = (y0 + 1).clamp(max=grid.ny - 1)
    fx = ix - x0.float(); fy = iy - y0.float()
    v00 = gridT[x0, y0]; v10 = gridT[x1, y0]; v01 = gridT[x0, y1]; v11 = gridT[x1, y1]
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy + v11 * fx * fy)


# --------------------------------------------------------------------------- #
# Sequence builder (adds a dense magnetic-anomaly observation)
# --------------------------------------------------------------------------- #
def build_sequence(walk, env, fix_tree, magmap, rng, wifi_period=1.0, ap_dropout=0.0, T=T_BINS):
    net, grid, coords_t, all_nodes, pool_nodes, pool, ap = env
    valsT, gxT, gyT, nstd, _ = magmap
    true_xy, accmag, head = walk
    n = len(true_xy); stride = max(1, int(wifi_period * FS))
    node_keys = [tuple(k) for k in pool_nodes]

    det = StepDetector(); u = np.zeros((n, 2))
    for t in range(n):
        if det.update(accmag[t]):
            u[t] = STEP_LEN * np.array([np.cos(head[t]), np.sin(head[t])])

    fixes = {}
    for t in range(0, n, stride):
        _, idx = fix_tree.query(true_xy[t])
        scan = pool[node_keys[idx]][rng.integers(len(pool[node_keys[idx]]))].copy()
        if ap_dropout > 0:
            scan[rng.random(len(scan)) < ap_dropout] = -100.0
        z, _ = heatmap_fix(net, scan, grid, coords_t)
        fixes[t] = z

    # dense magnetic observation = map value at true position + noise
    tx = torch.tensor(true_xy, dtype=torch.float32)
    mag_true = bilinear(valsT, tx, grid).numpy()
    mag_obs_frames = mag_true + rng.normal(0, nstd, n)

    edges = np.linspace(0, n, T + 1).astype(int)
    M = np.zeros((T, 2)); S = np.zeros((T, 2)); mask = np.zeros((T, 1))
    Yb = np.zeros((T, 2)); magobs = np.zeros((T, 1))
    start = true_xy[0].copy(); last = fixes[0]
    for i in range(T):
        a, b = edges[i], edges[i + 1]
        M[i] = u[a:b].sum(0)
        bf = [fixes[t] for t in fixes if a <= t < b]
        if bf:
            last = np.mean(bf, axis=0); mask[i] = 1.0
        S[i] = last
        Yb[i] = true_xy[min(b, n) - 1]
        magobs[i] = mag_obs_frames[min(b, n) - 1]
    return M, S, mask, magobs, Yb, start


# --------------------------------------------------------------------------- #
# Magnetic-augmented KalmanNet
# --------------------------------------------------------------------------- #
class MagKalmanNet(nn.Module):
    def __init__(self, magmap, use_mag=True, hidden=64):
        super().__init__()
        self.use_mag = use_mag
        self.vals, self.gx, self.gy, _, self.grid = magmap
        feat = 2 + 2 + 2 + 1 + (1 + 2 + 1 if use_mag else 0)  # innovW,dz,u,maskW [,innovM,grad,maskM]
        self.cell = nn.GRUCell(feat, hidden)
        out_dim = 4 + (2 if use_mag else 0)                    # K_wifi(2x2) [+ k_mag(2)]
        self.head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.head.weight)
        b = [0.5, 0., 0., 0.5] + ([0., 0.] if use_mag else [])
        self.head.bias.data = torch.tensor(b)
        self.hidden = hidden

    def forward(self, M, S, mask, magobs, start_zero=True):
        B, T, _ = M.shape
        h = torch.zeros(B, self.hidden); x = torch.zeros(B, 2)
        z_prev = S[:, 0, :]; dx_prev = torch.zeros(B, 2)
        outs = []
        for t in range(T):
            mw = mask[:, t, :]
            x_pred = x + M[:, t, :]
            innovW = (S[:, t, :] - x_pred) * mw
            dz = (S[:, t, :] - z_prev) * mw
            if self.use_mag:
                xp = (x_pred + self._origin).detach()    # absolute coords for map lookup
                mval = bilinear(self.vals, xp, self.grid).unsqueeze(-1)
                g = torch.stack([bilinear(self.gx, xp, self.grid),
                                 bilinear(self.gy, xp, self.grid)], dim=1)
                innovM = magobs[:, t, :] - mval
                feat = torch.cat([innovW, dz, M[:, t, :], mw, innovM, g, torch.ones(B, 1)], 1)
            else:
                feat = torch.cat([innovW, dz, M[:, t, :], mw], 1)
            h = self.cell(feat, h)
            o = self.head(h)
            K = o[:, :4].view(B, 2, 2)
            corr = mw * torch.bmm(K, innovW.unsqueeze(-1)).squeeze(-1)
            if self.use_mag:
                kmag = o[:, 4:6]
                corr = corr + kmag * innovM
            x_new = x_pred + corr
            dx_prev = x_new - x; z_prev = torch.where(mw.bool(), S[:, t, :], z_prev); x = x_new
            outs.append(x)
        return torch.stack(outs, 1)


# --------------------------------------------------------------------------- #
def make_set(nwalk, seed, env, graph, fix_tree, magmap, wifi_period, ap_dropout):
    A, main_cc = graph
    rng = np.random.default_rng(seed)
    data = []
    for k in range(nwalk * 2):
        if len(data) >= nwalk:
            break
        path = sample_path(A, main_cc, env[3], rng, min_len=30.0)
        if path is None:
            continue
        w = synth_walk(path, rng)
        if w is None:
            continue
        data.append(build_sequence(w, env, fix_tree, magmap,
                                   np.random.default_rng(seed*9991 + k),
                                   wifi_period=wifi_period, ap_dropout=ap_dropout))
    M, S, mask, magobs, Y, start = (np.stack([d[i] for d in data]) for i in range(6))
    return M, S, mask, magobs, Y, start


def train(model, data, epochs=150):
    M, S, mask, magobs, Y, start = data
    origin = torch.tensor(start, dtype=torch.float32)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Mt, St, Kt, MOt = t(M), t(S - start[:, None, :]), t(mask), t(magobs)
    Yt = t(Y - start[:, None, :])
    opt = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5); mse = nn.MSELoss()
    B = len(Mt); idx = np.arange(B)
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for s in range(0, B, 32):
            b = idx[s:s+32]
            model._origin = origin[b]
            opt.zero_grad()
            out = model(Mt[b], St[b], Kt[b], MOt[b])
            mse(out, Yt[b]).backward(); opt.step()
    return model


def evaluate(model, data):
    M, S, mask, magobs, Y, start = data
    origin = torch.tensor(start, dtype=torch.float32)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    model.eval(); model._origin = origin
    with torch.no_grad():
        out = model(t(M), t(S - start[:, None, :]), t(mask), t(magobs))
    err = np.linalg.norm(out.numpy() - (Y - start[:, None, :]), axis=2)
    return err.mean(1)


def main():
    print("=" * 68); print("Magnetic-augmented KalmanNet (extra measurement channel)"); print("=" * 68)
    env = setup_env()
    graph = corridor_graph(env[3])
    fix_tree = cKDTree(env[4])
    magmap = build_mag_map(env[1])

    regimes = {"full WiFi (1 Hz)": dict(wifi_period=1.0, ap_dropout=0.0),
               "degraded WiFi (5 s, 40% AP drop)": dict(wifi_period=5.0, ap_dropout=0.4)}

    results = {}
    for rname, rcfg in regimes.items():
        print(f"\n### Regime: {rname} ###")
        tr = make_set(250, 1, env, graph, fix_tree, magmap, **rcfg)
        te = make_set(60, 2, env, graph, fix_tree, magmap, **rcfg)
        row = {}
        for use_mag, label in [(False, "WiFi+IMU"), (True, "WiFi+IMU+Mag")]:
            torch.manual_seed(0)
            m = MagKalmanNet(magmap, use_mag=use_mag)
            m = train(m, tr)
            err = evaluate(m, te)
            mean, h = err.mean(), 1.96*err.std(ddof=1)/np.sqrt(len(err))
            row[label] = err
            print(f"  {label:14s} MAE={mean:.2f} +/- {h:.2f} m  (median {np.median(err):.2f})")
        gain = 100*(row['WiFi+IMU'].mean()-row['WiFi+IMU+Mag'].mean())/row['WiFi+IMU'].mean()
        print(f"  -> magnetic adds {gain:+.1f}%")
        results[rname] = row

    # plot CDFs
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, (rname, row) in zip(ax, results.items()):
        for lab, c in [("WiFi+IMU", "tab:blue"), ("WiFi+IMU+Mag", "tab:green")]:
            xs = np.sort(row[lab]); a.plot(xs, np.linspace(0,1,len(xs)), color=c,
                                           label=f"{lab} ({row[lab].mean():.2f}m)")
        a.set_title(rname); a.set_xlabel("per-walk MAE (m)"); a.set_ylabel("CDF")
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("../Datasets/stage3_mag_kalmannet.png", dpi=170)
    print("\nSaved -> Datasets/stage3_mag_kalmannet.png")


if __name__ == "__main__":
    main()
