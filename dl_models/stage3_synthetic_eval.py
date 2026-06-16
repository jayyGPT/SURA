"""
Stage 3 quantitative evaluation: causal EKF fusion vs baselines, on physically &
statistically faithful synthetic trajectories with KNOWN ground truth.

We cannot fabricate *measured* trajectory GT, but we can synthesise walks that are
faithful to the real environment and sensors, then score the filter against them:

  PHYSICAL FAITHFULNESS
    - Paths walk the REAL surveyed corridor graph: an epsilon-graph over the IT static
      nodes (edges <=1.6 m), shortest path between random endpoints in the largest
      connected component -> trajectories that stay on real corridors with real turns.
    - Constant-ish gait: per-walk speed 1.0-1.35 m/s, step cadence 1.7-2.0 Hz.

  STATISTICAL FAITHFULNESS (calibrated to the data)
    - Heading = true path tangent + slow gyro drift + 8.8 deg white noise
      (8.8 deg = the residual we measured for real Orn_z vs path direction).
    - WiFi fixes (~1 Hz) are drawn from REAL HELD-OUT static scans at the nearest node,
      so measurement noise (~3 dBm/AP across repeat visits) and device variation are real.
      The environment heatmap model is trained on a disjoint split of visits (no leakage).

  EVALUATION
    - N independent walks -> error distribution + 95% CI and a CDF.
    - Baselines isolate the fusion's value:  PDR-only (drifts) | WiFi-only (jumpy)
      | EKF fusion (drift-free + smooth).
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from stage2_wifi_heatmap import Grid, WifiHeatmapNet, encode_wifi, soft_argmax
from stage3_ekf_fusion import StepDetector, heatmap_fix

DB = "../Datasets/fingerprint_db/it_engineering"
KEEP = {"Navigation", "Call listening", "Swinging"}
FS = 16.7
HEAD_NOISE = np.deg2rad(8.8)     # measured real Orn_z residual
STEP_LEN = 0.65                  # fixed PDR step length (deployment uses an average)


# --------------------------------------------------------------------------- #
# Environment model (trained on a disjoint visit split) + held-out WiFi scan pool
# --------------------------------------------------------------------------- #
def setup_env(seed=0, epochs=70):
    df = pd.read_csv(f"{DB}/nodes.csv")
    ap = json.load(open(f"{DB}/bssid_vocab.json"))["ap_columns"]
    df = df[df["has_wifi"] & df["mode"].isin(KEEP)].reset_index(drop=True)
    grid = Grid(df["x"].values, df["y"].values)
    coords_t = torch.tensor(grid.coords, dtype=torch.float32)

    rng = np.random.default_rng(seed)
    holdout = rng.random(len(df)) < 0.30          # 30% scans reserved for WiFi fixes
    tr = ~holdout

    X = encode_wifi(df[ap].to_numpy(float))
    Y = np.stack([grid.gaussian_target(x, y) for x, y in df[["x", "y"]].values])
    Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
    net = WifiHeatmapNet(len(ap), grid.n_cells)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    idx = np.arange(len(Xtr))
    torch.manual_seed(seed)
    for _ in range(epochs):
        net.train(); np.random.shuffle(idx)
        for s in range(0, len(idx), 64):
            b = idx[s:s + 64]; opt.zero_grad()
            logp = torch.log_softmax(net(Xtr[b]), 1)
            (torch.sum(Ytr[b] * (torch.log(Ytr[b] + 1e-9) - logp), 1).mean()).backward()
            opt.step()
    net.eval()

    # per-node pool of held-out REAL scan vectors (for measurement noise realism)
    hd = df[holdout]
    pool = {}
    for (nx, ny), gi in hd.groupby([hd["x"].round(1), hd["y"].round(1)]).groups.items():
        pool[(round(nx, 1), round(ny, 1))] = df.loc[gi, ap].to_numpy(np.float32)
    pool_nodes = np.array(list(pool.keys()), float)
    # FULL node topology (for path generation) -- all surveyed corridor nodes
    all_nodes = df[["x", "y"]].round(1).drop_duplicates().to_numpy(float)
    return net, grid, coords_t, all_nodes, pool_nodes, pool, ap


# --------------------------------------------------------------------------- #
# Corridor graph over real nodes (largest connected component)
# --------------------------------------------------------------------------- #
def corridor_graph(nodes, r=1.6):
    tree = cKDTree(nodes)
    pairs = list(tree.query_pairs(r=r))
    N = len(nodes); ri, ci, w = [], [], []
    for i, j in pairs:
        d = np.linalg.norm(nodes[i] - nodes[j])
        ri += [i, j]; ci += [j, i]; w += [d, d]
    A = csr_matrix((w, (ri, ci)), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    main = np.flatnonzero(lab == np.bincount(lab).argmax())
    return A, main


def sample_path(A, main, nodes, rng, min_len=20.0):
    for _ in range(50):
        src, dst = rng.choice(main, 2, replace=False)
        dist, pred = dijkstra(A, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]) or dist[dst] < min_len:
            continue
        path = [dst]
        while path[-1] != src:
            path.append(pred[path[-1]])
            if pred[path[-1]] < 0:
                break
        return nodes[path[::-1]]
    return None


# --------------------------------------------------------------------------- #
# Synthesise one walk (faithful kinematics + sensors)
# --------------------------------------------------------------------------- #
def synth_walk(path_xy, rng):
    seg = np.diff(path_xy, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(L)]); total = cum[-1]
    speed = rng.uniform(1.0, 1.35); step_freq = rng.uniform(1.7, 2.0)
    n = int((total / speed) * FS)
    if n < 60:
        return None
    s = np.linspace(0, total, n)
    x = np.interp(s, cum, path_xy[:, 0]); y = np.interp(s, cum, path_xy[:, 1])
    true_xy = np.stack([x, y], 1)
    th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    th = np.convolve(th, np.ones(7) / 7, mode="same")           # mild smoothing
    drift = np.cumsum(rng.normal(0, np.deg2rad(0.5) / np.sqrt(FS), n))
    head = th + drift + rng.normal(0, HEAD_NOISE, n)
    tt = np.arange(n) / FS
    accmag = 9.81 + rng.uniform(0.8, 1.3) * np.sin(2 * np.pi * step_freq * tt) + rng.normal(0, 0.3, n)
    return true_xy, accmag, head


# --------------------------------------------------------------------------- #
# Causal filters: EKF fusion + PDR-only + WiFi-only
# --------------------------------------------------------------------------- #
def run_filters(true_xy, accmag, head, net, grid, coords_t, fix_tree, pool_nodes, pool, ap_cols,
                rng, step_len=STEP_LEN, q_step=0.5, r_scale=1.0, wifi_period=1.0):
    n = len(true_xy); stride = max(1, int(wifi_period * FS))
    det = StepDetector()
    x_e = true_xy[0].copy(); P = np.eye(2) * 4.0
    x_p = true_xy[0].copy(); last = true_xy[0].copy()
    Qf = np.eye(2) * 0.01
    te, tp, tw = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
    node_keys = [tuple(k) for k in pool_nodes]
    for t in range(n):
        step = det.update(accmag[t])
        u = step_len * np.array([np.cos(head[t]), np.sin(head[t])]) if step else np.zeros(2)
        x_p = x_p + u
        x_e = x_e + u; P = P + Qf + (q_step * np.eye(2) if step else 0.0)
        if t % stride == 0:
            _, idx = fix_tree.query(true_xy[t])
            scans = pool[node_keys[idx]]
            scan = scans[rng.integers(len(scans))]          # a real held-out scan
            z, R = heatmap_fix(net, scan, grid, coords_t)
            R = R * r_scale + np.eye(2) * 0.5
            K = P @ np.linalg.inv(P + R)
            x_e = x_e + K @ (z - x_e); P = (np.eye(2) - K) @ P
            last = z
        te[t], tp[t], tw[t] = x_e, x_p, last
    e = lambda trk: np.linalg.norm(trk - true_xy, axis=1)
    return te, tp, tw, e(te), e(tp), e(tw)


# --------------------------------------------------------------------------- #
def main():
    print("=" * 64); print("Stage 3: EKF vs baselines on faithful synthetic trajectories"); print("=" * 64)
    net, grid, coords_t, all_nodes, pool_nodes, pool, ap = setup_env()
    fix_tree = cKDTree(pool_nodes)               # WiFi fixes come from held-out-scan nodes
    A, main_cc = corridor_graph(all_nodes)       # paths span the FULL corridor topology
    print(f"env model ready | full corridor graph: {len(main_cc)}/{len(all_nodes)} nodes "
          f"in main component | {len(pool_nodes)} WiFi-fix nodes\n")

    # calibrate q_step, r_scale on a disjoint batch
    crng = np.random.default_rng(100)
    cal = [w for w in (synth_walk(sample_path(A, main_cc, all_nodes, crng), crng)
                       for _ in range(15)) if w]
    best = (1e9, (0.5, 1.0))
    for q in [0.2, 0.5, 1.0]:
        for r in [0.5, 1.0, 2.0]:
            ms = [run_filters(*w, net, grid, coords_t, fix_tree, pool_nodes, pool, ap,
                              np.random.default_rng(7), q_step=q, r_scale=r)[3].mean() for w in cal]
            if np.mean(ms) < best[0]:
                best = (np.mean(ms), (q, r))
    q_step, r_scale = best[1]
    print(f"calibrated filter: q_step={q_step}, r_scale={r_scale}\n")

    # evaluation batch
    erng = np.random.default_rng(0)
    walks, ekf, pdr, wifi, examples = 0, [], [], [], []
    N = 60
    for k in range(N):
        path = sample_path(A, main_cc, all_nodes, erng, min_len=30.0)
        if path is None:
            continue
        w = synth_walk(path, erng)
        if w is None:
            continue
        te, tp, tw, ee, ep, ewi = run_filters(*w, net, grid, coords_t, fix_tree, pool_nodes, pool, ap,
                                              np.random.default_rng(1000 + k),
                                              q_step=q_step, r_scale=r_scale)
        ekf.append(ee.mean()); pdr.append(ep.mean()); wifi.append(ewi.mean()); walks += 1
        if len(examples) < 3:
            examples.append((w[0], te, tp, tw, ee.mean()))

    def ci(a):
        a = np.array(a); m = a.mean(); h = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        return m, h
    me, he = ci(ekf); mp, hp = ci(pdr); mw, hw = ci(wifi)
    print(f"Walks evaluated: {walks}")
    print(f"  PDR-only  MAE = {mp:.2f} +/- {hp:.2f} m   (median {np.median(pdr):.2f})")
    print(f"  WiFi-only MAE = {mw:.2f} +/- {hw:.2f} m   (median {np.median(wifi):.2f})")
    print(f"  EKF fusion MAE= {me:.2f} +/- {he:.2f} m   (median {np.median(ekf):.2f})")
    print(f"  fusion vs WiFi-only: {100*(mw-me)/mw:+.0f}%   vs PDR-only: {100*(mp-me)/mp:+.0f}%")

    pd.DataFrame({"ekf": ekf, "pdr": pdr, "wifi": wifi}).to_csv("../Datasets/stage3_synth_eval.csv", index=False)

    # CDF + example trajectories
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for arr, lab, c in [(pdr, "PDR-only", "tab:red"), (wifi, "WiFi-only", "tab:orange"),
                        (ekf, "EKF fusion", "tab:green")]:
        xs = np.sort(arr); ax[0].plot(xs, np.linspace(0, 1, len(xs)), label=f"{lab} (MAE {np.mean(arr):.2f}m)", color=c)
    ax[0].set_xlabel("per-walk mean error (m)"); ax[0].set_ylabel("CDF")
    ax[0].set_title(f"Error CDF over {walks} synthetic walks"); ax[0].legend(); ax[0].grid(alpha=0.3)

    tru, te, tp, tw, _ = examples[0]
    ax[1].scatter(all_nodes[:, 0], all_nodes[:, 1], s=6, c="lightgray")
    ax[1].plot(tru[:, 0], tru[:, 1], "b-", lw=2, label="ground truth")
    ax[1].plot(tp[:, 0], tp[:, 1], "r-", lw=1, alpha=0.7, label="PDR-only")
    ax[1].plot(te[:, 0], te[:, 1], "g-", lw=1.5, label="EKF fusion")
    ax[1].set_title("Example walk"); ax[1].legend(fontsize=8); ax[1].set_aspect("equal", "box"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("../Datasets/stage3_synth_eval.png", dpi=170)
    print("\nSaved -> Datasets/stage3_synth_eval.png + stage3_synth_eval.csv")


if __name__ == "__main__":
    main()
