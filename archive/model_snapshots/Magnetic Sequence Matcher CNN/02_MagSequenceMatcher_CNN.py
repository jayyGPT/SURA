"""
MagSequenceMatcher — 1D-CNN Magnetic Sequence Matcher
======================================================
Standalone, self-contained reproduction script.

Source: stage2_mag_sequence.py
Paper:  Section II.C, Figure 3

This script defines the CNN model, data pipeline, loss function,
window-size sweep, and complete train/eval loop.
Run from the dl_models/ directory.
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
from scipy.interpolate import griddata
from tqdm import tqdm

# These imports come from the existing codebase
from stage3_synthetic_eval import (setup_env, corridor_graph, sample_path,
                                   synth_walk, FS, STEP_LEN)

# ========================= CONSTANTS ========================================
DB = "../Datasets/fingerprint_db/it_engineering"
KEEP = {"Navigation", "Call listening", "Swinging"}
MAG_FEATS = ["magN", "magV", "magH", "dip"]  # 4 rotation-invariant channels


# ========================= MAGNETIC ANOMALY MAP =============================
def build_multichannel_mag_map(grid):
    """
    Build a 4-channel device-invariant magnetic anomaly map.

    Returns:
        maps_t: Tensor [C, nx, ny] — interpolated anomaly per feature
        stds:   list[float] — per-feature within-node noise std
        grid:   Grid object
    """
    df = pd.read_csv(f"{DB}/nodes.csv")
    df = df[df["mode"].isin(KEEP)].dropna(subset=["magN_mean"]).reset_index(drop=True)

    maps, stds = [], []
    for f in MAG_FEATS:
        col = f"{f}_mean"
        if col not in df.columns:
            continue
        # Device-invariant anomaly: subtract per-phone building-wide mean
        df[f"anom_{f}"] = df[col] - df.groupby("phone")[col].transform("mean")
        node_agg = df.groupby([df["x"].round(1), df["y"].round(1)])[f"anom_{f}"].mean()
        nxy = np.array([list(k) for k in node_agg.index])
        nval = node_agg.values

        lin = griddata(nxy, nval, grid.coords, method="linear")
        nn_fill = griddata(nxy, nval, grid.coords, method="nearest")
        vals = np.where(np.isnan(lin), nn_fill, lin).reshape(grid.nx, grid.ny)
        maps.append(torch.tensor(vals, dtype=torch.float32))

        node_std = df.groupby(
            [df["x"].round(1), df["y"].round(1)]
        )[f"anom_{f}"].std().median()
        stds.append(float(node_std) if np.isfinite(node_std) else 1.0)

    maps_t = torch.stack(maps, dim=0)  # [C, nx, ny]
    return maps_t, stds, grid


def bilinear_mc(maps_t, x, grid):
    """Bilinear sample [C, nx, ny] at world coords x [B, 2]. Returns [B, C]."""
    ix = (x[:, 0] - grid.x0) / grid.cell
    iy = (x[:, 1] - grid.y0) / grid.cell
    ix = ix.clamp(0, grid.nx - 1.001)
    iy = iy.clamp(0, grid.ny - 1.001)
    x0 = ix.floor().long(); y0 = iy.floor().long()
    x1 = (x0 + 1).clamp(max=grid.nx - 1)
    y1 = (y0 + 1).clamp(max=grid.ny - 1)
    fx = ix - x0.float(); fy = iy - y0.float()
    v00 = maps_t[:, x0, y0]; v10 = maps_t[:, x1, y0]
    v01 = maps_t[:, x0, y1]; v11 = maps_t[:, x1, y1]
    out = (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy + v11 * fx * fy)
    return out.t()  # [B, C]


# ========================= DATA GENERATION ==================================
def generate_windows(nwalk, seed, env, graph, maps_t, stds, grid,
                     window_size, stride=5):
    """
    Generate (magnetic_window, target_xy) pairs from synthetic walks.

    Each window is [T, 4] (T frames × 4 magnetic features).
    Target is the true 2D position at the last frame of the window.
    """
    A, main_cc = graph
    rng = np.random.default_rng(seed)
    C = maps_t.shape[0]
    windows, targets = [], []
    max_attempts = nwalk * 4
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

        # Sample magnetic map at each true position + realistic noise
        tx = torch.tensor(true_xy, dtype=torch.float32)
        mag_clean = bilinear_mc(maps_t, tx, grid).numpy()
        noise = rng.normal(0, 1, size=(n, C)) * np.array(stds)
        mag_obs = mag_clean + noise

        for t in range(window_size, n, stride):
            windows.append(mag_obs[t - window_size: t])
            targets.append(true_xy[t - 1])

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


# ========================= MODEL ===========================================
class MagSequenceMatcher(nn.Module):
    """
    1D-CNN: magnetic window [B, T, 4] -> position [B, 2] + log-variance [B, 1].

    Encoder:
        Conv1D(4->32, k=7) -> BN -> ReLU -> MaxPool(2)
        Conv1D(32->64, k=5) -> BN -> ReLU -> MaxPool(2)
        Conv1D(64->128, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)

    Position head: FC(128->64) -> ReLU -> Dropout(0.2) -> FC(64->2)
    Variance head: FC(128->32) -> ReLU -> FC(32->1)   [outputs log(sigma^2)]
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
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [B, W, C] -> conv expects [B, C, W]
        feat = self.encoder(x.transpose(1, 2)).squeeze(-1)  # [B, hidden]
        pos = self.pos_head(feat)        # [B, 2]
        logvar = self.var_head(feat)      # [B, 1]
        return pos, logvar


# ========================= LOSS =============================================
def nll_loss(pred_pos, logvar, true_pos):
    """
    Heteroscedastic Gaussian NLL loss.

    L = (1/2) * ||z - z_true||^2 / sigma^2 + (1/2) * log(sigma^2)

    Encourages accurate predictions AND calibrated uncertainty.
    """
    var = torch.exp(logvar).clamp(min=0.01)
    sq_err = ((pred_pos - true_pos) ** 2).sum(dim=1, keepdim=True)
    return (0.5 * sq_err / var + 0.5 * logvar).mean()


# ========================= TRAINING =========================================
def train_one(window_size, env, graph, maps_t, stds, grid, epochs=60):
    """Train a MagSequenceMatcher for a given window_size."""
    print(f"\n--- Window = {window_size} frames ({window_size / FS:.1f}s) ---")

    X_tr, Y_tr = generate_windows(300, 42, env, graph, maps_t, stds, grid,
                                  window_size, stride=5)
    X_te, Y_te = generate_windows(60, 200, env, graph, maps_t, stds, grid,
                                  window_size, stride=10)
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
    best_state = None

    for ep in range(epochs):
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
            p_te, _ = model(X_te_t)
            mae = torch.norm(p_te - Y_te_t, dim=1).mean().item()
        sched.step(mae)

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:02d} | Loss: {epoch_loss/n_batches:.3f} | "
                  f"Test MAE: {mae:.2f}m | Best: {best_mae:.2f}m")

    model.load_state_dict(best_state)
    print(f"  Best Test MAE: {best_mae:.2f}m")
    return model, best_mae


# ========================= MAIN =============================================
def main():
    print("=" * 68)
    print("MagSequenceMatcher — 1D-CNN Magnetic Sequence Matcher")
    print("=" * 68)

    env = setup_env()
    graph = corridor_graph(env[3])
    maps_t, stds, grid = build_multichannel_mag_map(env[1])
    print(f"Magnetic anomaly map: {maps_t.shape[0]} channels, "
          f"grid {grid.nx}x{grid.ny}")
    print(f"Per-channel noise std: {[f'{s:.3f}' for s in stds]}")

    # Window size sweep
    candidates = [50, 84, 134, 167]
    results = {}
    best_overall = (float('inf'), None, None)
    for ws in candidates:
        model, mae = train_one(ws, env, graph, maps_t, stds, grid)
        results[ws] = mae
        if mae < best_overall[0]:
            best_overall = (mae, ws, model)

    print(f"\n{'=' * 68}")
    print("Window-size sweep results:")
    for ws, mae in results.items():
        tag = " <-- BEST" if ws == best_overall[1] else ""
        print(f"  {ws:4d} frames ({ws/FS:5.1f}s)  MAE = {mae:.2f}m{tag}")

    # Save best model
    if best_overall[2] is not None:
        torch.save({
            "state_dict": best_overall[2].state_dict(),
            "window_size": best_overall[1],
            "in_channels": len(MAG_FEATS),
        }, "best_mag_sequence.pth")
        print(f"\nSaved best model (window={best_overall[1]}) -> "
              f"best_mag_sequence.pth")


if __name__ == "__main__":
    main()
