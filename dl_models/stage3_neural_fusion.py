"""
Neural Kalman fusion vs classical EKF, on faithful synthetic trajectories.

We swap the classical EKF's fixed Kalman update for the learned fusion from
`train_ekf_fusion.py` (NeuralEKFFusion): a CausalConv+LSTM motion encoder and a
learned alpha-gate (neural Kalman gain) that auto-regressively blends a drift-free
spatial anchor with smooth motion:

    p[t] = alpha[t] * spatial[t] + (1 - alpha[t]) * (p[t-1] + delta[t])

To compare apples-to-apples, BOTH filters consume identical inputs on identical
sequences: the spatial anchor = our Stage-2 WiFi heatmap fix; the motion = per-bin
IMU pedestrian-dead-reckoning displacement. The only difference is fixed-gain KF math
vs the learned LSTM + alpha-gate. Now that the synthetic generator yields unlimited
GT trajectories, the learned fusion is trainable without the route-memorisation risk
that originally forced a classical filter.

Strictly causal: CausalConv (left pad), unidirectional LSTM, autoregressive update,
and alpha is masked to 0 when no WiFi observation is present in a bin.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from stage3_synthetic_eval import (setup_env, corridor_graph, sample_path, synth_walk,
                                   FS, STEP_LEN)
from stage3_ekf_fusion import StepDetector, heatmap_fix

T_BINS = 160
WIFI_PERIOD = 1.0


# --------------------------------------------------------------------------- #
# Turn one walk into a fixed-length binned sequence (shared by both filters)
# --------------------------------------------------------------------------- #
def build_sequence(walk, net, grid, coords_t, fix_tree, pool_nodes, pool, rng,
                   step_len=STEP_LEN, T=T_BINS):
    true_xy, accmag, head = walk
    n = len(true_xy); stride = max(1, int(WIFI_PERIOD * FS))
    node_keys = [tuple(k) for k in pool_nodes]

    # per-frame PDR displacement
    det = StepDetector(); u = np.zeros((n, 2))
    for t in range(n):
        if det.update(accmag[t]):
            u[t] = step_len * np.array([np.cos(head[t]), np.sin(head[t])])

    # WiFi fixes at ~1 Hz (from real held-out scans -> heatmap)
    fixes = {}
    for t in range(0, n, stride):
        _, idx = fix_tree.query(true_xy[t])
        scans = pool[node_keys[idx]]
        z, R = heatmap_fix(net, scans[rng.integers(len(scans))], grid, coords_t)
        fixes[t] = (z, float(np.trace(R)))

    edges = np.linspace(0, n, T + 1).astype(int)
    M = np.zeros((T, 2)); S = np.zeros((T, 2)); conf = np.zeros((T, 1)); mask = np.zeros((T, 1))
    Y = np.zeros((T, 2))
    start = true_xy[0].copy()
    last_fix = fixes[0][0]; last_conf = fixes[0][1]
    for i in range(T):
        a, b = edges[i], edges[i + 1]
        M[i] = u[a:b].sum(0)
        binfix = [fixes[t] for t in fixes if a <= t < b]
        if binfix:
            last_fix = np.mean([f[0] for f in binfix], axis=0)
            last_conf = np.mean([f[1] for f in binfix])
            mask[i] = 1.0
        S[i] = last_fix; conf[i] = np.log1p(last_conf)
        Y[i] = true_xy[min(b, n) - 1]
    Yd = np.diff(np.vstack([start, Y]), axis=0)
    return M, S, conf, mask, Y, Yd, start


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Module):
    def __init__(self, ci, co, k=3):
        super().__init__(); self.pad = k - 1
        self.conv = nn.Conv1d(ci, co, k)

    def forward(self, x):
        return self.conv(torch.nn.functional.pad(x, (self.pad, 0)))


class NeuralKalmanFusion(nn.Module):
    """Learned alpha-gate fusion of a spatial anchor (WiFi fix) and IMU motion."""
    def __init__(self, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(CausalConv1d(2, 32, 3), nn.ReLU())
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        self.motion_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2))
        self.alpha_gate = nn.Sequential(
            nn.Linear(hidden + 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, M, S, conf, mask, start):
        B, T, _ = M.shape
        h = self.conv(M.permute(0, 2, 1)).permute(0, 2, 1)
        h, _ = self.lstm(h)
        delta = self.motion_head(h)                                  # (B,T,2)
        alpha = self.alpha_gate(torch.cat([h, conf, mask], dim=2))   # (B,T,1)
        alpha = alpha * mask                                         # no obs -> pure motion
        p_prev = start
        out = []
        for t in range(T):
            a = alpha[:, t, :]
            p = a * S[:, t, :] + (1 - a) * (p_prev + delta[:, t, :])
            out.append(p); p_prev = p
        return torch.stack(out, 1), delta, alpha


def classical_ekf_bins(M, S, conf, mask, Y, start, q_step=0.5, r_scale=2.0):
    """Diagonal KF on the SAME binned inputs, for a fair comparison."""
    T = len(M); x = start.copy(); P = np.eye(2) * 4.0
    Qf = np.eye(2) * 0.05
    trk = np.zeros((T, 2))
    for t in range(T):
        x = x + M[t]
        P = P + Qf + (q_step * np.eye(2) if mask[t, 0] > 0 else 0.0)
        if mask[t, 0] > 0:
            R = np.eye(2) * (r_scale * np.expm1(conf[t, 0]) / 50.0 + 0.5)
            K = P @ np.linalg.inv(P + R)
            x = x + K @ (S[t] - x); P = (np.eye(2) - K) @ P
        trk[t] = x
    return np.linalg.norm(trk - Y, axis=1)


# --------------------------------------------------------------------------- #
def make_dataset(n, seed, env, graph):
    net, grid, coords_t, all_nodes, pool_nodes, pool, ap = env
    A, main_cc = graph
    fix_tree = cKDTree(pool_nodes)
    rng = np.random.default_rng(seed)
    data = []
    for k in range(n * 2):
        if len(data) >= n:
            break
        path = sample_path(A, main_cc, all_nodes, rng, min_len=30.0)
        if path is None:
            continue
        w = synth_walk(path, rng)
        if w is None:
            continue
        seq = build_sequence(w, net, grid, coords_t, fix_tree, pool_nodes, pool,
                             np.random.default_rng(seed * 9999 + k))
        data.append(seq)
    M, S, conf, mask, Y, Yd, start = (np.stack([d[i] for d in data]) for i in range(7))
    return data, (M, S, conf, mask, Y, Yd, start)


def main():
    print("=" * 64); print("Neural Kalman fusion vs classical EKF (synthetic GT)"); print("=" * 64)
    env = setup_env()
    graph = corridor_graph(env[3])  # all_nodes
    print("Generating datasets...")
    _, tr = make_dataset(250, 1, env, graph)
    test_raw, te = make_dataset(60, 2, env, graph)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Mtr, Str, Ctr, Ktr, Ytr, Ydtr, Pstr = map(to_t, tr)
    Mte, Ste, Cte, Kte, Yte, Ydte, Pste = map(to_t, te)
    print(f"train walks={len(Mtr)}  test walks={len(Mte)}  bins={T_BINS}\n")

    torch.manual_seed(0)
    model = NeuralKalmanFusion()
    opt = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    B = len(Mtr); idx = np.arange(B)
    for ep in range(150):
        model.train(); np.random.shuffle(idx)
        for s in range(0, B, 32):
            b = idx[s:s + 32]
            opt.zero_grad()
            p, delta, _ = model(Mtr[b], Str[b], Ctr[b], Ktr[b], Pstr[b])
            loss = mse(p, Ytr[b]) + 2.0 * mse(delta, Ydtr[b])
            loss.backward(); opt.step()
        if (ep + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                p, _, _ = model(Mte, Ste, Cte, Kte, Pste)
                e = torch.norm(p - Yte, dim=2).mean().item()
            print(f"  ep{ep+1:03d}  test MAE={e:.3f}m")

    # final per-walk MAE
    model.eval()
    with torch.no_grad():
        p, _, alpha = model(Mte, Ste, Cte, Kte, Pste)
    neural_err = torch.norm(p - Yte, dim=2).numpy()             # (B,T)
    neural_perwalk = neural_err.mean(1)

    classical_perwalk = np.array([
        classical_ekf_bins(te[0][i], te[1][i], te[2][i], te[3][i], te[4][i], te[6][i]).mean()
        for i in range(len(Mte))])
    # WiFi-only on bins: error of the (carried) fix vs truth
    wifi_perwalk = np.array([np.linalg.norm(te[1][i] - te[4][i], axis=1).mean() for i in range(len(Mte))])

    def ci(a):
        a = np.array(a); return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    print("\n--- per-walk MAE over 60 held-out synthetic walks (identical inputs) ---")
    for name, arr in [("WiFi-only (binned)", wifi_perwalk),
                      ("Classical EKF", classical_perwalk),
                      ("Neural Kalman fusion", neural_perwalk)]:
        m, h = ci(arr); print(f"  {name:22s} MAE = {m:.2f} +/- {h:.2f} m   (median {np.median(arr):.2f})")
    mc = classical_perwalk.mean(); mn = neural_perwalk.mean()
    print(f"\n  neural vs classical: {100*(mc-mn)/mc:+.1f}%   mean alpha (gated)={alpha[Kte.bool()].mean().item():.3f}")

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for arr, lab, c in [(wifi_perwalk, "WiFi-only", "tab:orange"),
                        (classical_perwalk, "Classical EKF", "tab:blue"),
                        (neural_perwalk, "Neural Kalman", "tab:green")]:
        xs = np.sort(arr); ax[0].plot(xs, np.linspace(0, 1, len(xs)), color=c,
                                      label=f"{lab} ({arr.mean():.2f}m)")
    ax[0].set_xlabel("per-walk mean error (m)"); ax[0].set_ylabel("CDF")
    ax[0].set_title("Neural Kalman vs Classical EKF (60 walks)"); ax[0].legend(); ax[0].grid(alpha=0.3)

    i = 0
    tr_xy = te[4][i]
    with torch.no_grad():
        pe, _, _ = model(Mte[i:i+1], Ste[i:i+1], Cte[i:i+1], Kte[i:i+1], Pste[i:i+1])
    pe = pe[0].numpy()
    ax[1].scatter(env[3][:, 0], env[3][:, 1], s=6, c="lightgray")
    ax[1].plot(tr_xy[:, 0], tr_xy[:, 1], "b-", lw=2, label="ground truth")
    ax[1].plot(pe[:, 0], pe[:, 1], "g-", lw=1.5, label="neural fusion")
    ax[1].set_title("Example walk"); ax[1].legend(fontsize=8); ax[1].set_aspect("equal", "box"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("../Datasets/stage3_neural_vs_ekf.png", dpi=170)
    print("\nSaved -> Datasets/stage3_neural_vs_ekf.png")


if __name__ == "__main__":
    main()
