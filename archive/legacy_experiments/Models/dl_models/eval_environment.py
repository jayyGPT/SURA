"""
Paper evaluation, Route A: generalization + robustness of the WiFi environment model.

Pillars:
  1. Cross-building   : per-building random 80/20 accuracy (works campus-wide?)
  2. Device general.  : leave-one-phone-out (train on N-1 phones, test on the held-out)
  3. Robustness       : WiFi AP-dropout degradation curve (IT)
  4. Env-not-traj     : cross-scenario (train Scenario-1, test Scenario-2) on shared nodes

Each building uses its OWN AP vocabulary and its OWN grid (per-environment model).
Single floor: Stairs/Room excluded.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage2_wifi_heatmap import Grid, WifiHeatmapNet, encode_wifi, soft_argmax

FP_ROOT = "../Datasets/fingerprint_db"
BUILDINGS = ["it_engineering", "cs_engineering", "electrical_engineering", "iact", "be_building"]
DROP_MODES = {"Stairs", "Room"}
EPOCHS = 70


def load_db(slug):
    df = pd.read_csv(f"{FP_ROOT}/{slug}/nodes.csv")
    ap_cols = json.load(open(f"{FP_ROOT}/{slug}/bssid_vocab.json"))["ap_columns"]
    df = df[df["has_wifi"] & ~df["mode"].isin(DROP_MODES)].reset_index(drop=True)
    return df, ap_cols


def train_eval(df, ap_cols, tr, te, grid=None, epochs=EPOCHS, ap_dropout=0.0, seed=0):
    """Train heatmap on tr rows, eval on te rows. Returns (err array, model, grid)."""
    torch.manual_seed(seed)
    if grid is None:
        grid = Grid(df["x"].values, df["y"].values)
    coords_t = torch.tensor(grid.coords, dtype=torch.float32)
    X = encode_wifi(df[ap_cols].to_numpy(float))
    Y = np.stack([grid.gaussian_target(x, y) for x, y in df[["x", "y"]].values])
    pos = df[["x", "y"]].to_numpy(np.float32)

    Xtr = torch.tensor(X[tr]); Ytr = torch.tensor(Y[tr])
    net = WifiHeatmapNet(len(ap_cols), grid.n_cells)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 64
    idx = np.arange(len(Xtr))
    for _ in range(epochs):
        net.train(); np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            logp = torch.log_softmax(net(Xtr[b]), 1)
            loss = torch.sum(Ytr[b] * (torch.log(Ytr[b] + 1e-9) - logp), 1).mean()
            loss.backward(); opt.step()

    Xte = X[te].copy()
    if ap_dropout > 0:                       # simulate APs not seen at inference
        rng = np.random.default_rng(seed)
        mask = rng.random(Xte.shape) < ap_dropout
        Xte[mask] = 0.0
    net.eval()
    with torch.no_grad():
        pred = soft_argmax(net(torch.tensor(Xte)), coords_t).numpy()
    err = np.linalg.norm(pred - pos[te], axis=1)
    return err, net, grid


def pillar_cross_building():
    print("\n" + "=" * 64); print("PILLAR 1+2: cross-building + leave-one-phone-out"); print("=" * 64)
    rows = []
    for slug in BUILDINGS:
        df, ap_cols = load_db(slug)
        if len(df) < 40:
            print(f"  {slug}: too few visits ({len(df)}), skipped"); continue
        grid = Grid(df["x"].values, df["y"].values)
        rng = np.random.default_rng(0)
        te = rng.random(len(df)) < 0.2
        err_rand, _, _ = train_eval(df, ap_cols, ~te, te, grid)

        # leave-one-phone-out
        lopo = []
        for ph in sorted(df["phone"].unique()):
            te_p = (df["phone"] == ph).to_numpy()
            if te_p.sum() < 20 or (~te_p).sum() < 40:
                continue
            e, _, _ = train_eval(df, ap_cols, ~te_p, te_p, grid)
            lopo.append((ph, e.mean()))
        lopo_mean = np.mean([m for _, m in lopo]) if lopo else np.nan

        rows.append({"building": slug, "nodes": df[["x","y"]].round(1).drop_duplicates().shape[0],
                     "visits": len(df), "n_ap": len(ap_cols),
                     "rand_MAE": err_rand.mean(), "rand_median": np.median(err_rand),
                     "LOPO_MAE": lopo_mean,
                     "LOPO_detail": ", ".join(f"{p}:{m:.1f}" for p, m in lopo)})
        print(f"  {slug:24s} nodes={rows[-1]['nodes']:3d} visits={len(df):4d}  "
              f"random MAE={err_rand.mean():.2f}m (med {np.median(err_rand):.2f})  "
              f"LOPO MAE={lopo_mean:.2f}m  [{rows[-1]['LOPO_detail']}]")
    out = pd.DataFrame(rows)
    out.to_csv("../Datasets/env_eval_buildings.csv", index=False)
    print("\nSaved -> Datasets/env_eval_buildings.csv")
    return out


def pillar_ap_dropout(slug="it_engineering"):
    print("\n" + "=" * 64); print(f"PILLAR 3: WiFi AP-dropout robustness ({slug})"); print("=" * 64)
    df, ap_cols = load_db(slug)
    grid = Grid(df["x"].values, df["y"].values)
    rng = np.random.default_rng(0)
    te = rng.random(len(df)) < 0.2
    res = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        err, _, _ = train_eval(df, ap_cols, ~te, te, grid, ap_dropout=p)
        res.append((p, err.mean(), np.median(err)))
        print(f"  drop {int(p*100):2d}% APs:  MAE={err.mean():.2f}m  median={np.median(err):.2f}m")
    res = np.array(res)
    plt.figure(figsize=(7, 4))
    plt.plot(res[:, 0]*100, res[:, 1], "o-", label="MAE")
    plt.plot(res[:, 0]*100, res[:, 2], "s--", label="median")
    plt.xlabel("% of APs dropped at inference"); plt.ylabel("error (m)")
    plt.title(f"WiFi AP-dropout robustness ({slug})"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("../Datasets/env_eval_ap_dropout.png", dpi=170)
    print("Saved -> Datasets/env_eval_ap_dropout.png")
    return res


def pillar_cross_scenario(slug="it_engineering"):
    print("\n" + "=" * 64); print(f"PILLAR 4: cross-scenario / direction (env-not-trajectory) ({slug})"); print("=" * 64)
    df, ap_cols = load_db(slug)
    scen = sorted(df["scenario"].unique())
    print(f"  scenarios present: {scen}")
    s1 = df["scenario"] == "Scenario-1"
    s2 = df["scenario"] == "Scenario-2"
    if s1.sum() < 40 or s2.sum() < 20:
        print("  insufficient scenario overlap; skipped"); return None
    # shared spatial support: only score S2 nodes that also exist in S1
    s1_nodes = set(map(tuple, df[s1][["x", "y"]].round(1).values))
    grid = Grid(df["x"].values, df["y"].values)
    err, _, _ = train_eval(df, ap_cols, s1.to_numpy(), s2.to_numpy(), grid)
    s2pos = df[s2][["x", "y"]].round(1).values
    shared = np.array([tuple(p) in s1_nodes for p in s2pos])
    print(f"  train Scenario-1 ({s1.sum()}), test Scenario-2 ({s2.sum()})")
    print(f"    all S2 nodes:    MAE={err.mean():.2f}m  median={np.median(err):.2f}m")
    if shared.any():
        print(f"    S2 on shared nodes ({shared.sum()}): MAE={err[shared].mean():.2f}m  median={np.median(err[shared]):.2f}m")
    return err


if __name__ == "__main__":
    pillar_cross_building()
    pillar_ap_dropout()
    pillar_cross_scenario()
