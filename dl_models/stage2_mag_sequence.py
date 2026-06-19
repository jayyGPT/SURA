"""
Stage 2.5: Magnetic Sequence Matcher

Train a 1D-CNN to map a sliding window of rotation-invariant magnetic features
to a 2D spatial coordinate (z_mag) and an uncertainty estimate (sigma_mag).

Unlike Wi-Fi (strong per-point, direction-independent), magnetic data is weak
per-frame but discriminative over a temporal sequence. The CNN captures the
spatial anomaly profile that accumulates along a path.

The model is trained on synthetic trajectories that sample the real magnetic
anomaly map (built from device-invariant residuals) with realistic noise. It
is later plugged into the Dual-Update KalmanNet as a second measurement channel.

Multiple window sizes are swept to find the optimal temporal context.
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"  # force unbuffered output for live progress
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

from stage3_synthetic_eval import (setup_env, corridor_graph, sample_path, synth_walk,
                                   FS, STEP_LEN)

DB = "../Datasets/fingerprint_db/it_engineering"
KEEP = {"Navigation", "Call listening", "Swinging"}
MAG_FEATS = ["magN", "magV", "magH", "dip"]   # 4 rotation-invariant channels


# --------------------------------------------------------------------------- #
# Multi-channel magnetic anomaly map on the environment grid
# --------------------------------------------------------------------------- #
def build_multichannel_mag_map(grid):
    """
    Returns:
        maps_t  : Tensor [C, nx, ny]  -- interpolated anomaly map per feature
        stds    : list[float]         -- per-feature within-node noise std
        grid    : Grid object
    """
    df = pd.read_csv(f"{DB}/nodes.csv")
    df = df[df["mode"].isin(KEEP)].dropna(subset=["magN_mean"]).reset_index(drop=True)

    maps, stds = [], []
    for f in MAG_FEATS:
        col = f"{f}_mean"
        if col not in df.columns:
            continue
        # device-invariant anomaly: subtract per-phone building-wide mean
        df[f"anom_{f}"] = df[col] - df.groupby("phone")[col].transform("mean")
        node_agg = df.groupby([df["x"].round(1), df["y"].round(1)])[f"anom_{f}"].mean()
        nxy = np.array([list(k) for k in node_agg.index])
        nval = node_agg.values

        lin = griddata(nxy, nval, grid.coords, method="linear")
        nn_fill = griddata(nxy, nval, grid.coords, method="nearest")
        vals = np.where(np.isnan(lin), nn_fill, lin).reshape(grid.nx, grid.ny)
        maps.append(torch.tensor(vals, dtype=torch.float32))

        node_std = df.groupby([df["x"].round(1), df["y"].round(1)])[f"anom_{f}"].std().median()
        stds.append(float(node_std) if np.isfinite(node_std) else 1.0)

    maps_t = torch.stack(maps, dim=0)  # [C, nx, ny]
    return maps_t, stds, grid


def bilinear_mc(maps_t, x, grid):
    """Bilinear sample from [C, nx, ny] at world coords x [B, 2]. Returns [B, C]."""
    ix = (x[:, 0] - grid.x0) / grid.cell
    iy = (x[:, 1] - grid.y0) / grid.cell
    ix = ix.clamp(0, grid.nx - 1.001)
    iy = iy.clamp(0, grid.ny - 1.001)
    x0 = ix.floor().long(); y0 = iy.floor().long()
    x1 = (x0 + 1).clamp(max=grid.nx - 1)
    y1 = (y0 + 1).clamp(max=grid.ny - 1)
    fx = ix - x0.float(); fy = iy - y0.float()
    v00 = maps_t[:, x0, y0]  # [C, B]
    v10 = maps_t[:, x1, y0]
    v01 = maps_t[:, x0, y1]
    v11 = maps_t[:, x1, y1]
    out = (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy + v11 * fx * fy)
    return out.t()  # [B, C]


# --------------------------------------------------------------------------- #
# Dataset: sliding windows of magnetic sequences from synthetic walks
# --------------------------------------------------------------------------- #
def generate_windows(nwalk, seed, env, graph, maps_t, stds, grid, window_size, stride=5):
    """Generate (window, target_xy) pairs from synthetic walks."""
    A, main_cc = graph
    rng = np.random.default_rng(seed)
    C = maps_t.shape[0]

    windows, targets = [], []
    max_attempts = nwalk * 4
    pbar = tqdm(total=nwalk * 40, desc=f"  Generating windows (w={window_size})", unit="win", leave=False)
    attempts = 0
    while len(windows) < nwalk * 40 and attempts < max_attempts:
        attempts += 1
        path = sample_path(A, main_cc, env[3], rng, min_len=30.0)
        if path is None:
            continue
        w = synth_walk(path, rng)
        if w is None:
            continue
        true_xy, _, _ = w
        n = len(true_xy)
        if n <= window_size:
            continue

        # sample the magnetic map at each true position + realistic noise
        tx = torch.tensor(true_xy, dtype=torch.float32)
        mag_clean = bilinear_mc(maps_t, tx, grid).numpy()  # [n, C]
        noise = rng.normal(0, 1, size=(n, C)) * np.array(stds)
        mag_obs = mag_clean + noise

        prev_count = len(windows)
        for t in range(window_size, n, stride):
            windows.append(mag_obs[t - window_size: t])
            targets.append(true_xy[t - 1])
        pbar.update(len(windows) - prev_count)

    pbar.close()
    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


# --------------------------------------------------------------------------- #
# MagSequenceMatcher: 1D-CNN with uncertainty output
# --------------------------------------------------------------------------- #
class MagSequenceMatcher(nn.Module):
    """
    Maps a [B, W, C] magnetic window to a position fix [B, 2] and
    a log-variance [B, 1] (used downstream as measurement uncertainty).
    """
    def __init__(self, in_channels=4, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
        self.var_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1),  # outputs log(sigma^2)
        )

    def forward(self, x):
        # x: [B, W, C] -> conv expects [B, C, W]
        feat = self.encoder(x.transpose(1, 2)).squeeze(-1)  # [B, hidden]
        pos = self.pos_head(feat)        # [B, 2]
        logvar = self.var_head(feat)      # [B, 1]
        return pos, logvar


def nll_loss(pred_pos, logvar, true_pos):
    """Heteroscedastic Gaussian NLL: encourages calibrated uncertainty."""
    var = torch.exp(logvar).clamp(min=0.01)  # [B, 1]
    sq_err = ((pred_pos - true_pos) ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    return (0.5 * sq_err / var + 0.5 * logvar).mean()


# --------------------------------------------------------------------------- #
# Training loop with window-size sweep
# --------------------------------------------------------------------------- #
def train_one(window_size, env, graph, maps_t, stds, grid, epochs=60, verbose=True):
    """Train a MagSequenceMatcher for a given window_size. Returns (model, test_mae)."""
    if verbose:
        print(f"\n--- Window = {window_size} frames ({window_size / FS:.1f}s) ---")

    X_tr, Y_tr = generate_windows(300, 42, env, graph, maps_t, stds, grid, window_size, stride=5)
    X_te, Y_te = generate_windows(60, 200, env, graph, maps_t, stds, grid, window_size, stride=10)
    if verbose:
        print(f"  Train windows: {len(X_tr)}  |  Test windows: {len(X_te)}")
    if len(X_tr) < 100 or len(X_te) < 20:
        print("  Too few windows, skipping.")
        return None, float('inf')

    X_tr_t = torch.tensor(X_tr); Y_tr_t = torch.tensor(Y_tr)
    X_te_t = torch.tensor(X_te); Y_te_t = torch.tensor(Y_te)

    model = MagSequenceMatcher(in_channels=len(MAG_FEATS))
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    B = len(X_tr_t); idx = np.arange(B)
    best_mae = float('inf')

    epoch_pbar = tqdm(range(epochs), desc=f"  Training (w={window_size})", unit="ep", leave=False)
    for ep in epoch_pbar:
        model.train(); np.random.shuffle(idx)
        epoch_loss = 0; n_batches = 0
        for s in range(0, B, 128):
            b = idx[s:s + 128]; opt.zero_grad()
            pred, logvar = model(X_tr_t[b])
            loss = nll_loss(pred, logvar, Y_tr_t[b])
            loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1

        model.eval()
        with torch.no_grad():
            p_te, lv_te = model(X_te_t)
            mae = torch.norm(p_te - Y_te_t, dim=1).mean().item()
        sched.step(mae)

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        epoch_pbar.set_postfix(loss=f"{epoch_loss/n_batches:.2f}", mae=f"{mae:.2f}m", best=f"{best_mae:.2f}m")
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            tqdm.write(f"  Epoch {ep+1:02d} | Loss: {epoch_loss/n_batches:.3f} | Test MAE: {mae:.2f}m")
    epoch_pbar.close()

    model.load_state_dict(best_state)
    if verbose:
        print(f"  Best Test MAE: {best_mae:.2f}m")
    return model, best_mae


def main():
    print("=" * 68)
    print("Stage 2.5: Magnetic Sequence Matcher (window-size sweep)")
    print("=" * 68)

    env = setup_env()
    graph = corridor_graph(env[3])
    maps_t, stds, grid = build_multichannel_mag_map(env[1])
    print(f"Magnetic anomaly map: {maps_t.shape[0]} channels, grid {grid.nx}x{grid.ny}")
    print(f"Per-channel noise std: {[f'{s:.3f}' for s in stds]}")

    # Sweep window sizes: 50 (~3s), 84 (~5s), 134 (~8s), 167 (~10s)
    candidates = [50, 84, 134, 167]
    results = {}
    best_overall = (float('inf'), None, None)
    for ws in candidates:
        model, mae = train_one(ws, env, graph, maps_t, stds, grid)
        results[ws] = mae
        if mae < best_overall[0]:
            best_overall = (mae, ws, model)

    print("\n" + "=" * 68)
    print("Window-size sweep results:")
    for ws, mae in results.items():
        tag = " <-- BEST" if ws == best_overall[1] else ""
        print(f"  {ws:4d} frames ({ws/FS:5.1f}s)  MAE = {mae:.2f}m{tag}")

    # Save the best model
    best_model = best_overall[2]
    best_ws = best_overall[1]
    if best_model is not None:
        torch.save({
            "state_dict": best_model.state_dict(),
            "window_size": best_ws,
            "in_channels": len(MAG_FEATS),
        }, "best_mag_sequence.pth")
        print(f"\nSaved best model (window={best_ws}) -> best_mag_sequence.pth")

    # CDF plot of test errors for the best model
    _, Y_te = generate_windows(60, 200, env, graph, maps_t, stds, grid, best_ws, stride=10)
    X_te, _ = generate_windows(60, 200, env, graph, maps_t, stds, grid, best_ws, stride=10)
    best_model.eval()
    with torch.no_grad():
        p, _ = best_model(torch.tensor(X_te))
        errs = torch.norm(p - torch.tensor(Y_te), dim=1).numpy()
    plt.figure(figsize=(8, 5))
    xs = np.sort(errs)
    plt.plot(xs, np.linspace(0, 1, len(xs)), 'g-', lw=2, label=f"MagSeq (MAE={errs.mean():.2f}m)")
    plt.xlabel("Position error (m)"); plt.ylabel("CDF")
    plt.title(f"Magnetic Sequence Matcher (window={best_ws} frames)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("../Datasets/stage2_mag_sequence_cdf.png", dpi=170)
    print("Saved -> Datasets/stage2_mag_sequence_cdf.png")


if __name__ == "__main__":
    main()
