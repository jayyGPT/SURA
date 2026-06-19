"""
Stage 3: Dual-Update KalmanNet

A Kalman filter whose gain is learned by a GRU, extended with TWO independent
measurement channels: Wi-Fi (sparse, ~1 Hz) and Magnetic sequence (dense, every frame).

Architecture:
    The GRU receives innovations from both sensors and outputs two separate 2x2
    matrix gains. The state update is:

        x_pred = x_{t-1} + u_t                                     (PDR predict)
        x_t = x_pred + m_wifi * (K_wifi @ y_wifi)                  (Wi-Fi correct)
                      + m_mag  * (K_mag  @ y_mag)                  (Magnetic correct)

    When Wi-Fi is absent (15 out of 16 frames), K_wifi is masked to zero and the
    filter corrects using K_mag alone. When magnetic is unreliable (flat corridor),
    the GRU learns to drive K_mag toward zero and wait for the next Wi-Fi scan.
    This dynamic "sliding" between sensors is the core advantage.

Evaluation regimes:
    1. Full Wi-Fi (1 Hz)              -- magnetic should still add marginal value
    2. Degraded Wi-Fi (5s, 40% drop)  -- magnetic should provide large improvement

Compared models:
    - WiFi-only KalmanNet      (original stage3_kalmannet.py baseline)
    - Dual KalmanNet (WiFi+Mag) (this file)
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
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
from tqdm import tqdm

from stage3_synthetic_eval import (setup_env, corridor_graph, sample_path, synth_walk,
                                   FS, STEP_LEN)
from stage3_ekf_fusion import StepDetector, heatmap_fix
from stage2_mag_sequence import build_multichannel_mag_map, bilinear_mc, MagSequenceMatcher

DB = "../Datasets/fingerprint_db/it_engineering"
KEEP = {"Navigation", "Call listening", "Swinging"}
T_BINS = 160


# --------------------------------------------------------------------------- #
# Scalar magnetic anomaly map (for the simple gradient-based innovation)
# --------------------------------------------------------------------------- #
def build_scalar_mag_map(grid):
    """Single-channel anomaly map + gradients (for the GRU innovation features)."""
    df = pd.read_csv(f"{DB}/nodes.csv")
    df = df[df["mode"].isin(KEEP)].dropna(subset=["magN_mean"]).reset_index(drop=True)
    df["anom"] = df["magN_mean"] - df.groupby("phone")["magN_mean"].transform("mean")
    node = df.groupby([df["x"].round(1), df["y"].round(1)])["anom"].mean()
    nstd = df.groupby([df["x"].round(1), df["y"].round(1)])["anom"].std().median()
    nxy = np.array([list(k) for k in node.index]); nval = node.values

    lin = griddata(nxy, nval, grid.coords, method="linear")
    nn_fill = griddata(nxy, nval, grid.coords, method="nearest")
    vals = np.where(np.isnan(lin), nn_fill, lin).reshape(grid.nx, grid.ny)
    gx, gy = np.gradient(vals, grid.cell)

    return (torch.tensor(vals, dtype=torch.float32),
            torch.tensor(gx, dtype=torch.float32),
            torch.tensor(gy, dtype=torch.float32),
            float(nstd), grid)


def bilinear_scalar(gridT, x, grid):
    """Sample a [nx, ny] grid at world coords x [B, 2]. Returns [B]."""
    ix = (x[:, 0] - grid.x0) / grid.cell
    iy = (x[:, 1] - grid.y0) / grid.cell
    ix = ix.clamp(0, grid.nx - 1.001); iy = iy.clamp(0, grid.ny - 1.001)
    x0 = ix.floor().long(); y0 = iy.floor().long()
    x1 = (x0 + 1).clamp(max=grid.nx - 1); y1 = (y0 + 1).clamp(max=grid.ny - 1)
    fx = ix - x0.float(); fy = iy - y0.float()
    v00 = gridT[x0, y0]; v10 = gridT[x1, y0]; v01 = gridT[x0, y1]; v11 = gridT[x1, y1]
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
            v01 * (1 - fx) * fy + v11 * fx * fy)


# --------------------------------------------------------------------------- #
# Sequence builder: produces binned inputs with BOTH Wi-Fi and magnetic channels
# --------------------------------------------------------------------------- #
def build_sequence(walk, env, fix_tree, magmap, rng,
                   wifi_period=1.0, ap_dropout=0.0, T=T_BINS):
    net, grid, coords_t, all_nodes, pool_nodes, pool, ap = env
    valsT, gxT, gyT, nstd, _ = magmap
    true_xy, accmag, head = walk
    n = len(true_xy); stride = max(1, int(wifi_period * FS))
    node_keys = [tuple(k) for k in pool_nodes]

    # PDR displacement
    det = StepDetector(); u = np.zeros((n, 2))
    for t in range(n):
        if det.update(accmag[t]):
            u[t] = STEP_LEN * np.array([np.cos(head[t]), np.sin(head[t])])

    # Wi-Fi fixes at the specified cadence
    fixes = {}
    for t in range(0, n, stride):
        _, idx = fix_tree.query(true_xy[t])
        scan = pool[node_keys[idx]][rng.integers(len(pool[node_keys[idx]]))].copy()
        if ap_dropout > 0:
            scan[rng.random(len(scan)) < ap_dropout] = -100.0
        z, _ = heatmap_fix(net, scan, grid, coords_t)
        fixes[t] = z

    # Dense magnetic observation = map value at true position + noise
    tx = torch.tensor(true_xy, dtype=torch.float32)
    mag_true = bilinear_scalar(valsT, tx, grid).numpy()
    mag_obs_frames = mag_true + rng.normal(0, nstd, n)

    # Bin into T_BINS
    edges = np.linspace(0, n, T + 1).astype(int)
    M = np.zeros((T, 2))           # PDR motion
    S_wifi = np.zeros((T, 2))      # Wi-Fi spatial fix (carried forward)
    mask_wifi = np.zeros((T, 1))   # 1 when Wi-Fi is present in this bin
    mag_obs = np.zeros((T, 1))     # magnetic observation (every bin)
    Y = np.zeros((T, 2))           # ground truth position
    start = true_xy[0].copy()
    last_wifi = fixes[0]

    for i in range(T):
        a, b = edges[i], edges[i + 1]
        M[i] = u[a:b].sum(0)
        bf = [fixes[t] for t in fixes if a <= t < b]
        if bf:
            last_wifi = np.mean(bf, axis=0)
            mask_wifi[i] = 1.0
        S_wifi[i] = last_wifi
        Y[i] = true_xy[min(b, n) - 1]
        mag_obs[i] = mag_obs_frames[min(b, n) - 1]

    return M, S_wifi, mask_wifi, mag_obs, Y, start


# --------------------------------------------------------------------------- #
# DualKalmanNet: two independent 2x2 matrix gains
# --------------------------------------------------------------------------- #
class DualKalmanNet(nn.Module):
    """
    GRU-learned dual matrix Kalman gain.

    Input features per step:
        y_wifi  (2)   : Wi-Fi innovation (z_wifi - x_pred), masked
        y_mag   (1)   : magnetic innovation (obs - map(x_pred))
        grad    (2)   : magnetic field gradient at x_pred
        dz_wifi (2)   : temporal diff of Wi-Fi fix
        u       (2)   : PDR motion
        dx_prev (2)   : previous state update
        m_wifi  (1)   : Wi-Fi availability mask
        m_mag   (1)   : magnetic availability mask (always 1)
        ---
        Total: 13

    Output:
        K_wifi  (2x2) : Wi-Fi gain matrix
        K_mag   (2x2) : Magnetic gain matrix
        ---
        Total: 8 values from the linear head
    """
    def __init__(self, magmap, hidden=64):
        super().__init__()
        self.vals, self.gx, self.gy, _, self.grid = magmap

        feat_dim = 2 + 1 + 2 + 2 + 2 + 2 + 1 + 1  # = 13
        self.cell = nn.GRUCell(feat_dim, hidden)
        self.head = nn.Linear(hidden, 8)  # 4 for K_wifi + 4 for K_mag
        self.hidden = hidden

        # Initialize: K_wifi near diagonal 0.5, K_mag near zero
        nn.init.zeros_(self.head.weight)
        self.head.bias.data = torch.tensor([0.5, 0.0, 0.0, 0.5,   # K_wifi
                                            0.0, 0.0, 0.0, 0.0])  # K_mag starts conservative

    def forward(self, M, S_wifi, mask_wifi, mag_obs, start_abs):
        """
        M          : [B, T, 2]  PDR motion
        S_wifi     : [B, T, 2]  Wi-Fi spatial fix (centred)
        mask_wifi  : [B, T, 1]  Wi-Fi mask
        mag_obs    : [B, T, 1]  magnetic observation
        start_abs  : [B, 2]     absolute start position (for map lookup)
        """
        B, T, _ = M.shape
        h = torch.zeros(B, self.hidden, device=M.device)
        x = torch.zeros(B, 2, device=M.device)         # centred state
        z_prev = S_wifi[:, 0, :]
        dx_prev = torch.zeros(B, 2, device=M.device)
        outs = []

        for t in range(T):
            mw = mask_wifi[:, t, :]                      # [B, 1]
            x_pred = x + M[:, t, :]                      # PDR predict

            # Wi-Fi innovation
            y_wifi = (S_wifi[:, t, :] - x_pred) * mw     # [B, 2]
            dz_wifi = (S_wifi[:, t, :] - z_prev) * mw    # [B, 2]

            # Magnetic innovation: requires absolute coords for map lookup
            xp_abs = (x_pred + start_abs).detach()
            map_val = bilinear_scalar(self.vals, xp_abs, self.grid).unsqueeze(-1)  # [B, 1]
            grad = torch.stack([
                bilinear_scalar(self.gx, xp_abs, self.grid),
                bilinear_scalar(self.gy, xp_abs, self.grid),
            ], dim=1)  # [B, 2]
            y_mag = mag_obs[:, t, :] - map_val            # [B, 1]
            m_mag = torch.ones(B, 1, device=M.device)     # always available

            # Assemble GRU features
            feat = torch.cat([y_wifi, y_mag, grad, dz_wifi, M[:, t, :],
                              dx_prev, mw, m_mag], dim=1)  # [B, 13]
            h = self.cell(feat, h)
            o = self.head(h)                               # [B, 8]

            # Decode dual gains
            K_wifi = o[:, :4].view(B, 2, 2)
            K_mag = o[:, 4:8].view(B, 2, 2)

            # Dual correction
            corr_wifi = mw * torch.bmm(K_wifi, y_wifi.unsqueeze(-1)).squeeze(-1)
            corr_mag = torch.bmm(K_mag, (y_mag * grad).unsqueeze(-1)).squeeze(-1)

            x_new = x_pred + corr_wifi + corr_mag

            # Bookkeeping
            dx_prev = x_new - x
            z_prev = torch.where(mw.bool(), S_wifi[:, t, :], z_prev)
            x = x_new
            outs.append(x)

        return torch.stack(outs, 1)


# --------------------------------------------------------------------------- #
# WiFi-only KalmanNet baseline (for comparison)
# --------------------------------------------------------------------------- #
class WiFiOnlyKalmanNet(nn.Module):
    """Original single-measurement KalmanNet (WiFi only, no magnetic)."""
    def __init__(self, hidden=64):
        super().__init__()
        feat_dim = 2 + 2 + 2 + 2 + 1  # y_wifi, dz, u, dx_prev, mask
        self.cell = nn.GRUCell(feat_dim, hidden)
        self.head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.head.weight)
        self.head.bias.data = torch.tensor([0.5, 0.0, 0.0, 0.5])
        self.hidden = hidden

    def forward(self, M, S, mask):
        B, T, _ = M.shape
        h = torch.zeros(B, self.hidden); x = torch.zeros(B, 2)
        z_prev = S[:, 0, :]; dx_prev = torch.zeros(B, 2)
        outs = []
        for t in range(T):
            mw = mask[:, t, :]
            x_pred = x + M[:, t, :]
            innov = (S[:, t, :] - x_pred) * mw
            dz = (S[:, t, :] - z_prev) * mw
            feat = torch.cat([innov, dz, M[:, t, :], dx_prev, mw], dim=1)
            h = self.cell(feat, h)
            K = self.head(h).view(B, 2, 2)
            corr = torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1) * mw
            x_new = x_pred + corr
            dx_prev = x_new - x
            z_prev = torch.where(mw.bool(), S[:, t, :], z_prev)
            x = x_new
            outs.append(x)
        return torch.stack(outs, 1)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def make_set(nwalk, seed, env, graph, fix_tree, magmap, wifi_period, ap_dropout):
    A, main_cc = graph
    rng = np.random.default_rng(seed)
    data = []
    pbar = tqdm(total=nwalk, desc=f"    Generating {nwalk} walks", unit="walk", leave=False)
    for k in range(nwalk * 3):
        if len(data) >= nwalk:
            break
        path = sample_path(A, main_cc, env[3], rng, min_len=30.0)
        if path is None: continue
        w = synth_walk(path, rng)
        if w is None: continue
        data.append(build_sequence(w, env, fix_tree, magmap,
                                   np.random.default_rng(seed * 9991 + k),
                                   wifi_period=wifi_period, ap_dropout=ap_dropout))
        pbar.update(1)
    pbar.close()
    M, S, mask, mag, Y, start = (np.stack([d[i] for d in data]) for i in range(6))
    return M, S, mask, mag, Y, start


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_dual(model, data, epochs=150, lr=2e-3):
    M, S, mask, mag, Y, start = data
    origin = torch.tensor(start, dtype=torch.float32)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Mt, St, Kt, MOt = t(M), t(S - start[:, None, :]), t(mask), t(mag)
    Yt = t(Y - start[:, None, :])

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    B = len(Mt); idx = np.arange(B)

    pbar = tqdm(range(epochs), desc="    Training", unit="ep", leave=False)
    for ep in pbar:
        model.train(); np.random.shuffle(idx)
        for s in range(0, B, 32):
            b = idx[s:s+32]
            model._origin = origin[b] if hasattr(model, '_origin') else None
            opt.zero_grad()
            if isinstance(model, DualKalmanNet):
                out = model(Mt[b], St[b], Kt[b], MOt[b], origin[b])
            else:
                out = model(Mt[b], St[b], Kt[b])
            mse(out, Yt[b]).backward(); opt.step()
        if (ep + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                if isinstance(model, DualKalmanNet):
                    out = model(Mt, St, Kt, MOt, origin)
                else:
                    out = model(Mt, St, Kt)
                e = torch.norm(out - Yt, dim=2).mean().item()
            pbar.set_postfix(mae=f"{e:.3f}m")
            tqdm.write(f"    ep{ep+1:03d}  test MAE = {e:.3f}m")
    pbar.close()
    return model


def evaluate(model, data):
    M, S, mask, mag, Y, start = data
    origin = torch.tensor(start, dtype=torch.float32)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if isinstance(model, DualKalmanNet):
            out = model(t(M), t(S - start[:, None, :]), t(mask), t(mag), origin)
        else:
            out = model(t(M), t(S - start[:, None, :]), t(mask))
    err = np.linalg.norm(out.numpy() - (Y - start[:, None, :]), axis=2)
    return err.mean(1)


# --------------------------------------------------------------------------- #
# Main: train and compare across regimes
# --------------------------------------------------------------------------- #
def main():
    print("=" * 68)
    print("Dual-Update KalmanNet: WiFi+IMU vs WiFi+IMU+Magnetic")
    print("=" * 68)

    env = setup_env()
    graph = corridor_graph(env[3])
    fix_tree = cKDTree(env[4])
    magmap = build_scalar_mag_map(env[1])

    regimes = {
        "Full WiFi (1 Hz, 0% AP drop)": dict(wifi_period=1.0, ap_dropout=0.0),
        "Degraded WiFi (5s, 40% AP drop)": dict(wifi_period=5.0, ap_dropout=0.4),
    }

    all_results = {}
    for rname, rcfg in regimes.items():
        print(f"\n{'='*68}")
        print(f"Regime: {rname}")
        print(f"{'='*68}")
        tr = make_set(250, 1, env, graph, fix_tree, magmap, **rcfg)
        te = make_set(60,  2, env, graph, fix_tree, magmap, **rcfg)
        print(f"  Train: {len(tr[0])} walks | Test: {len(te[0])} walks | Bins: {T_BINS}")

        row = {}
        for label, ModelClass, use_mag in [
            ("WiFi+IMU (KalmanNet)", WiFiOnlyKalmanNet, False),
            ("WiFi+IMU+Mag (DualKalmanNet)", DualKalmanNet, True),
        ]:
            print(f"\n  Training: {label}")
            torch.manual_seed(0)
            if use_mag:
                model = ModelClass(magmap)
            else:
                model = ModelClass()
            model = train_dual(model, tr if use_mag else tr, epochs=150)
            err = evaluate(model, te)
            mean_e = err.mean()
            ci = 1.96 * err.std(ddof=1) / np.sqrt(len(err))
            row[label] = err
            print(f"  {label:40s}  MAE = {mean_e:.2f} +/- {ci:.2f} m  "
                  f"(median {np.median(err):.2f})")

        keys = list(row.keys())
        baseline = row[keys[0]].mean()
        dual = row[keys[1]].mean()
        gain = 100 * (baseline - dual) / baseline
        print(f"\n  Magnetic adds: {gain:+.1f}% improvement")
        all_results[rname] = row

    # CDF plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (rname, row) in zip(axes, all_results.items()):
        for (lab, errs), c in zip(row.items(), ["tab:blue", "tab:green"]):
            xs = np.sort(errs)
            ax.plot(xs, np.linspace(0, 1, len(xs)), color=c, lw=2,
                    label=f"{lab} ({errs.mean():.2f}m)")
        ax.set_title(rname, fontsize=10)
        ax.set_xlabel("Per-walk MAE (m)")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../Datasets/stage3_dual_kalmannet.png", dpi=170)
    print("\nSaved -> Datasets/stage3_dual_kalmannet.png")


if __name__ == "__main__":
    main()
