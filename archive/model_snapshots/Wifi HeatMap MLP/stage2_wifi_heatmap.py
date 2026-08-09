"""
Stage 2: WiFi-anchored environment branch (pretrained on the static fingerprint DB).

Goal: learn the ENVIRONMENT as a measurement model -- given a single WiFi scan,
output a probability HEATMAP over floor cells (not a single point). A heatmap
honestly represents fingerprint ambiguity (one scan can match several places);
the downstream causal filter resolves it over time.

Design:
  - Per-frame, WiFi-only MLP -> softmax over a 2D grid of corridor cells.
    Per-frame + WiFi-only => trivially direction-invariant (the environment).
  - Target: Gaussian blob centred on the true node (sigma in metres), normalised
    to a distribution. Loss: KL divergence (soft labels).
  - Prediction: soft-argmax (probability-weighted centroid) -> (x, y) in metres.
  - Single floor: Stairs/Room modes are excluded (they reuse corridor X,Y at a
    different height and would alias the 2D map). Multi-floor is Phase 2.

Eval: random visit split AND a held-out-phone split (S9+) to test device invariance.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_DIR = "../Datasets/fingerprint_db/it_engineering"
CELL = 1.0          # grid resolution (m)
SIGMA = 2.0         # target blob width (m)
WIFI_FLOOR = -100.0
RSS_CLIP = -90.0    # clip weak APs
KEEP_MODES = {"Navigation", "Call listening", "Swinging"}  # single corridor floor


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
class Grid:
    def __init__(self, xs, ys, cell=CELL):
        self.x0, self.x1 = float(np.floor(xs.min())), float(np.ceil(xs.max()))
        self.y0, self.y1 = float(np.floor(ys.min())), float(np.ceil(ys.max()))
        self.cell = cell
        self.nx = int(round((self.x1 - self.x0) / cell)) + 1
        self.ny = int(round((self.y1 - self.y0) / cell)) + 1
        gx = self.x0 + np.arange(self.nx) * cell
        gy = self.y0 + np.arange(self.ny) * cell
        self.gxx, self.gyy = np.meshgrid(gx, gy, indexing="ij")  # (nx, ny)
        self.coords = np.stack([self.gxx.ravel(), self.gyy.ravel()], axis=1)  # (nx*ny, 2)
        self.n_cells = self.nx * self.ny

    def gaussian_target(self, x, y):
        d2 = (self.gxx - x) ** 2 + (self.gyy - y) ** 2
        t = np.exp(-d2 / (2 * SIGMA ** 2)).ravel()
        s = t.sum()
        return (t / s).astype(np.float32) if s > 0 else t.astype(np.float32)


def encode_wifi(rss_mat):
    """RSS (N, A) with -100 floor -> [0,1] strength, missing -> 0."""
    x = np.clip(rss_mat, RSS_CLIP, -30.0)
    x = (x - RSS_CLIP) / (-30.0 - RSS_CLIP)   # 0..1
    x[rss_mat <= WIFI_FLOOR] = 0.0            # absent AP
    return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class WifiHeatmapNet(nn.Module):
    def __init__(self, n_ap, n_cells):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ap, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_cells),
        )

    def forward(self, x):
        return self.net(x)  # logits over cells


def soft_argmax(logits, coords_t):
    p = torch.softmax(logits, dim=1)            # (B, n_cells)
    return p @ coords_t                          # (B, 2)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run(split="random"):
    df = pd.read_csv(os.path.join(DB_DIR, "nodes.csv"))
    vocab = json.load(open(os.path.join(DB_DIR, "bssid_vocab.json")))
    ap_cols = vocab["ap_columns"]

    df = df[df["has_wifi"] & df["mode"].isin(KEEP_MODES)].reset_index(drop=True)
    print(f"[{split}] usable WiFi visits: {len(df)} | modes={sorted(df['mode'].unique())}")

    grid = Grid(df["x"].values, df["y"].values)
    print(f"Grid: {grid.nx} x {grid.ny} = {grid.n_cells} cells "
          f"(X[{grid.x0:.0f},{grid.x1:.0f}] Y[{grid.y0:.0f},{grid.y1:.0f}])")

    X = encode_wifi(df[ap_cols].to_numpy(float))
    Y = np.stack([grid.gaussian_target(x, y) for x, y in df[["x", "y"]].values])
    pos = df[["x", "y"]].to_numpy(np.float32)

    if split == "phone":  # held-out device
        te = (df["phone"] == "S9+").to_numpy()
    else:                 # random visit split
        rng = np.random.default_rng(0)
        te = rng.random(len(df)) < 0.2
    tr = ~te
    print(f"Train: {tr.sum()}  Test: {te.sum()}")

    coords_t = torch.tensor(grid.coords, dtype=torch.float32)
    dl = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(Y[tr]),
                                  torch.tensor(pos[tr])), batch_size=64, shuffle=True)

    model = WifiHeatmapNet(len(ap_cols), grid.n_cells)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def kl(logits, target):
        logp = torch.log_softmax(logits, dim=1)
        return torch.sum(target * (torch.log(target + 1e-9) - logp), dim=1).mean()

    Xte = torch.tensor(X[te]); poste = torch.tensor(pos[te])
    best = 1e9
    for ep in range(80):
        model.train()
        for xb, yb, _ in dl:
            opt.zero_grad()
            loss = kl(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = soft_argmax(model(Xte), coords_t).numpy()
            err = np.linalg.norm(pred - poste.numpy(), axis=1)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep{ep+1:02d}  test MAE={err.mean():.2f}m  median={np.median(err):.2f}m")
        best = min(best, err.mean())

    # final eval + nearest-node top-1
    with torch.no_grad():
        logits = model(Xte)
        pred = soft_argmax(logits, coords_t).numpy()
    err = np.linalg.norm(pred - poste.numpy(), axis=1)
    print(f"[{split}] FINAL  MAE={err.mean():.2f}m  median={np.median(err):.2f}m  "
          f"p90={np.percentile(err,90):.2f}m  max={err.max():.2f}m")

    # save a few example heatmaps
    if split == "random":
        torch.save({"state": model.state_dict(),
                    "grid": {"x0": grid.x0, "y0": grid.y0, "nx": grid.nx,
                             "ny": grid.ny, "cell": grid.cell},
                    "ap_cols": ap_cols}, "best_wifi_heatmap.pth")
        _plot_examples(model, Xte, poste.numpy(), grid, coords_t)
    return err


def _plot_examples(model, Xte, poste, grid, coords_t, k=4):
    model.eval()
    idx = np.linspace(0, len(Xte) - 1, k).astype(int)
    with torch.no_grad():
        p = torch.softmax(model(Xte[idx]), dim=1).numpy()
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    for j, ax in enumerate(axes):
        hm = p[j].reshape(grid.nx, grid.ny).T
        ax.imshow(hm, origin="lower", extent=[grid.x0, grid.x1, grid.y0, grid.y1],
                  aspect="equal", cmap="hot")
        ax.scatter([poste[idx[j], 0]], [poste[idx[j], 1]], marker="*", s=140,
                   edgecolors="cyan", facecolors="none", linewidths=1.5, label="true")
        ax.set_title(f"WiFi heatmap #{idx[j]}"); ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig("../Datasets/stage2_wifi_heatmaps.png", dpi=180, bbox_inches="tight")
    print("Saved -> Datasets/stage2_wifi_heatmaps.png")


if __name__ == "__main__":
    print("=" * 64)
    print("Stage 2: WiFi-anchored heatmap (static DB pretrain)")
    print("=" * 64)
    run("random")
    print()
    run("phone")
