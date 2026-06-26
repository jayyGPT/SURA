"""
WifiHeatmapNet — Wi-Fi Probability Heatmap MLP
================================================
Standalone, self-contained reproduction script.

Source: stage2_wifi_heatmap.py
Paper:  Section II.B, Figure 2

This script defines the model, preprocessing, grid, loss, inference,
and a complete train/eval loop. Run it from the dl_models/ directory.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ========================= CONSTANTS ========================================
DB_DIR = "../Datasets/fingerprint_db/it_engineering"
CELL = 1.0           # grid resolution (m)
SIGMA = 2.0          # Gaussian target blob width (m)
WIFI_FLOOR = -100.0  # dBm value meaning "AP not detected"
RSS_CLIP = -90.0     # clip weak APs below this
KEEP_MODES = {"Navigation", "Call listening", "Swinging"}


# ========================= SPATIAL GRID =====================================
class Grid:
    """Regular 2D grid over the building's bounding box."""
    def __init__(self, xs, ys, cell=CELL):
        self.x0 = float(np.floor(xs.min()))
        self.x1 = float(np.ceil(xs.max()))
        self.y0 = float(np.floor(ys.min()))
        self.y1 = float(np.ceil(ys.max()))
        self.cell = cell
        self.nx = int(round((self.x1 - self.x0) / cell)) + 1
        self.ny = int(round((self.y1 - self.y0) / cell)) + 1
        gx = self.x0 + np.arange(self.nx) * cell
        gy = self.y0 + np.arange(self.ny) * cell
        self.gxx, self.gyy = np.meshgrid(gx, gy, indexing="ij")
        self.coords = np.stack([self.gxx.ravel(), self.gyy.ravel()], axis=1)
        self.n_cells = self.nx * self.ny  # = M in the paper

    def gaussian_target(self, x, y):
        """2D Gaussian soft label centred on (x, y) with std = SIGMA."""
        d2 = (self.gxx - x) ** 2 + (self.gyy - y) ** 2
        t = np.exp(-d2 / (2 * SIGMA ** 2)).ravel()
        s = t.sum()
        return (t / s).astype(np.float32) if s > 0 else t.astype(np.float32)


# ========================= PREPROCESSING ====================================
def encode_wifi(rss_mat):
    """
    Normalize raw RSS matrix to [0, 1].

    Steps:
        1. Clip to [-90, -30] dBm
        2. Rescale linearly to [0, 1]
        3. Set absent APs (value <= -100) to 0.0
    """
    x = np.clip(rss_mat, RSS_CLIP, -30.0)
    x = (x - RSS_CLIP) / (-30.0 - RSS_CLIP)
    x[rss_mat <= WIFI_FLOOR] = 0.0
    return x.astype(np.float32)


# ========================= MODEL ===========================================
class WifiHeatmapNet(nn.Module):
    """
    Multi-Layer Perceptron: RSSI vector -> probability heatmap over grid cells.

    Architecture:
        Input(N) -> FC(256) -> ReLU -> Dropout(0.3)
                 -> FC(256) -> ReLU -> Dropout(0.3)
                 -> FC(M)   [raw logits]

    Forward returns LOGITS (not probabilities).
    Apply softmax externally for inference.
    """
    def __init__(self, n_ap, n_cells):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ap, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_cells),
        )

    def forward(self, x):
        return self.net(x)


# ========================= INFERENCE ========================================
def soft_argmax(logits, coords_t):
    """
    Probability-weighted centroid (soft argmax).

    Args:
        logits:   [B, M] raw model output
        coords_t: [M, 2] physical coordinates of each grid cell

    Returns:
        [B, 2] continuous 2D position estimate
    """
    p = torch.softmax(logits, dim=1)
    return p @ coords_t


# ========================= LOSS =============================================
def kl_divergence_loss(logits, target):
    """
    KL divergence: D_KL(target || predicted).

    Args:
        logits: [B, M] raw model output
        target: [B, M] Gaussian soft label

    Returns:
        scalar loss
    """
    logp = torch.log_softmax(logits, dim=1)
    return torch.sum(target * (torch.log(target + 1e-9) - logp), dim=1).mean()


# ========================= TRAIN / EVAL =====================================
def run(split="random"):
    """Complete training and evaluation pipeline."""
    # --- Load data ---
    df = pd.read_csv(os.path.join(DB_DIR, "nodes.csv"))
    vocab = json.load(open(os.path.join(DB_DIR, "bssid_vocab.json")))
    ap_cols = vocab["ap_columns"]
    df = df[df["has_wifi"] & df["mode"].isin(KEEP_MODES)].reset_index(drop=True)
    print(f"[{split}] usable WiFi visits: {len(df)}")

    # --- Build grid ---
    grid = Grid(df["x"].values, df["y"].values)
    print(f"Grid: {grid.nx} x {grid.ny} = {grid.n_cells} cells")

    # --- Encode inputs and targets ---
    X = encode_wifi(df[ap_cols].to_numpy(float))
    Y = np.stack([grid.gaussian_target(x, y) for x, y in df[["x", "y"]].values])
    pos = df[["x", "y"]].to_numpy(np.float32)

    # --- Train/test split ---
    if split == "phone":
        te = (df["phone"] == "S9+").to_numpy()
    else:
        rng = np.random.default_rng(0)
        te = rng.random(len(df)) < 0.2
    tr = ~te
    print(f"Train: {tr.sum()}  Test: {te.sum()}")

    # --- DataLoader ---
    coords_t = torch.tensor(grid.coords, dtype=torch.float32)
    dl = DataLoader(
        TensorDataset(torch.tensor(X[tr]), torch.tensor(Y[tr]), torch.tensor(pos[tr])),
        batch_size=64, shuffle=True,
    )

    # --- Model + optimizer ---
    model = WifiHeatmapNet(len(ap_cols), grid.n_cells)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # --- Training loop ---
    Xte = torch.tensor(X[te])
    poste = torch.tensor(pos[te])
    best = 1e9
    for ep in range(80):
        model.train()
        for xb, yb, _ in dl:
            opt.zero_grad()
            loss = kl_divergence_loss(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = soft_argmax(model(Xte), coords_t).numpy()
            err = np.linalg.norm(pred - poste.numpy(), axis=1)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep{ep+1:02d}  test MAE={err.mean():.2f}m  median={np.median(err):.2f}m")
        best = min(best, err.mean())

    # --- Final evaluation ---
    with torch.no_grad():
        logits = model(Xte)
        pred = soft_argmax(logits, coords_t).numpy()
    err = np.linalg.norm(pred - poste.numpy(), axis=1)
    print(f"[{split}] FINAL  MAE={err.mean():.2f}m  median={np.median(err):.2f}m  "
          f"p90={np.percentile(err,90):.2f}m  max={err.max():.2f}m")

    # --- Save model ---
    if split == "random":
        torch.save({
            "state": model.state_dict(),
            "grid": {"x0": grid.x0, "y0": grid.y0, "nx": grid.nx,
                     "ny": grid.ny, "cell": grid.cell},
            "ap_cols": ap_cols,
        }, "best_wifi_heatmap.pth")
    return err


if __name__ == "__main__":
    print("=" * 64)
    print("WifiHeatmapNet — Wi-Fi Probability Heatmap MLP")
    print("=" * 64)
    run("random")
    print()
    run("phone")
