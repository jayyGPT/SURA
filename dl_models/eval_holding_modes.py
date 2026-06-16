"""
Robustness of the WiFi environment model to phone HOLDING MODE (posture) and to
combined held-out (posture + device), evaluated at SHARED nodes (no spatial
extrapolation confound).

Data reality (IT Engineering, paired WiFi, KEEP modes):
  - Navigation : phones A8/G7/S8/S9+, Scenario-1 + Scenario-2
  - Call listening : S8 only, Scenario-1
  - Swinging   : S8 only, Scenario-1
  Swinging/Call nodes are a 100% subset of Navigation nodes.

Because Swinging/Call exist ONLY on S8 and ONLY in Scenario-1, a fully-combined
"held-out phone AND held-out orientation AND held-out trajectory" is NOT available in
the real data. The closest honest combined test: train on Navigation EXCLUDING S8,
then test on S8-Swinging -> both the device (S8) and the posture (swing) are unseen.
"""
import numpy as np
import pandas as pd
from eval_environment import load_db, train_eval
from stage2_wifi_heatmap import Grid


def shared_eval(df, ap, grid, tr_mask, te_mask, label, seed=0):
    tr = tr_mask.to_numpy(); te = te_mask.to_numpy()
    tr_nodes = set(map(tuple, df[tr][["x", "y"]].round(1).values))
    err, _, _ = train_eval(df, ap, tr, te, grid, seed=seed)
    te_pos = df[te][["x", "y"]].round(1).values
    shared = np.array([tuple(p) in tr_nodes for p in te_pos])
    e = err[shared]
    print(f"  {label:46s} train={tr.sum():3d} test={te.sum():3d} (shared {shared.sum():3d})"
          f"  MAE={e.mean():.2f}m  median={np.median(e):.2f}m")
    return e.mean()


def main():
    df, ap = load_db("it_engineering")
    grid = Grid(df["x"].values, df["y"].values)
    m = df["mode"]; p = df["phone"]; sc = df["scenario"]
    nav = m == "Navigation"; sw = m == "Swinging"; call = m == "Call listening"

    print("=" * 72)
    print("Holding-mode & combined held-out robustness (env model, shared nodes)")
    print("=" * 72)

    print("\n[Reference] same posture (Navigation)")
    rng = np.random.default_rng(0)
    rand = pd.Series(rng.random(len(df)) < 0.2, index=df.index)
    shared_eval(df, ap, grid, nav & ~rand, nav & rand, "Navigation random split")
    shared_eval(df, ap, grid, nav & (sc == "Scenario-1"), nav & (sc == "Scenario-2"),
                "trajectory held-out (train S1 -> test S2)")

    print("\n[Held-out POSTURE] train Navigation (all phones) -> test new posture")
    shared_eval(df, ap, grid, nav, call, "posture held-out: Call listening (S8)")
    shared_eval(df, ap, grid, nav, sw,   "posture held-out: Swinging (S8)")

    print("\n[Held-out POSTURE + DEVICE] train Navigation excluding S8 -> test S8 posture")
    shared_eval(df, ap, grid, nav & (p != "S8"), call, "posture+device held-out: Call (S8)")
    shared_eval(df, ap, grid, nav & (p != "S8"), sw,   "posture+device held-out: Swinging (S8)")

    print("\nNote: Swinging/Call are S8-only & Scenario-1-only in the corpus, so a fully")
    print("combined posture+device+trajectory held-out test needs synthesis or new data.")


if __name__ == "__main__":
    main()
