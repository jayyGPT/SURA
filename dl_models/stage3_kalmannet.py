"""
KalmanNet: a Kalman filter whose gain is learned by an RNN -- no linear/Gaussian
assumption, but the KF predict->innovate->correct recursion is preserved.

Per step (state x = position, known motion f: x_pred = x_{t-1} + u_t, known H = I):
    x_pred  = x_{t-1} + u_t                         # predict (motion, known)
    innov   = z_t - x_pred                          # innovation (only when WiFi obs)
    K_t     = GRU(features)  -> 2x2 matrix           # LEARNED Kalman gain
    x_t     = x_pred + K_t @ innov                  # correct

The GRU consumes innovation, observation-difference, motion, and the previous state
update (KalmanNet's F-features) and implicitly tracks the uncertainty that the analytic
KF would carry in P -- so it needs no F/H/Q/R and no linear-Gaussian assumption, while
keeping the KF inductive bias (more robust & data-efficient than a blind LSTM).

Compared head-to-head, on identical binned inputs / identical held-out walks, against:
  WiFi-only | classical EKF | scalar-alpha neural fusion | KalmanNet (matrix gain).
Strictly causal (GRUCell stepping, autoregressive innovation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage3_neural_fusion import (setup_env, corridor_graph, make_dataset,
                                  classical_ekf_bins, NeuralKalmanFusion, T_BINS)


class KalmanNet(nn.Module):
    """GRU-learned matrix Kalman gain on a centred (start=0) position state."""
    def __init__(self, hidden=64):
        super().__init__()
        feat_dim = 2 + 2 + 2 + 2 + 1   # innov, dz, motion u, prev-update dx, mask
        self.cell = nn.GRUCell(feat_dim, hidden)
        self.kgain = nn.Linear(hidden, 4)
        # start near a diagonal gain of ~0.5 (sane KF-like behaviour)
        nn.init.zeros_(self.kgain.weight)
        self.kgain.bias.data = torch.tensor([0.5, 0.0, 0.0, 0.5])
        self.hidden = hidden

    def forward(self, M, S, mask):
        B, T, _ = M.shape
        h = torch.zeros(B, self.hidden, device=M.device)
        x = torch.zeros(B, 2, device=M.device)
        z_prev = S[:, 0, :]
        dx_prev = torch.zeros(B, 2, device=M.device)
        outs = []
        for t in range(T):
            m = mask[:, t, :]
            x_pred = x + M[:, t, :]
            innov = (S[:, t, :] - x_pred) * m
            dz = (S[:, t, :] - z_prev) * m
            feat = torch.cat([innov, dz, M[:, t, :], dx_prev, m], dim=1)
            h = self.cell(feat, h)
            K = self.kgain(h).view(B, 2, 2)
            corr = torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1) * m
            x_new = x_pred + corr
            dx_prev = x_new - x
            z_prev = torch.where(m.bool(), S[:, t, :], z_prev)
            x = x_new
            outs.append(x)
        return torch.stack(outs, 1)


def train_model(model, fwd, Mtr, args_tr, Ytr, Mte, args_te, Yte, epochs=150, lr=2e-3):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    B = len(Mtr); idx = np.arange(B)
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for s in range(0, B, 32):
            b = idx[s:s + 32]
            opt.zero_grad()
            out = fwd(model, b, Mtr, args_tr)
            mse(out, Ytr[b]).backward()
            opt.step()
        if (ep + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                e = torch.norm(fwd(model, slice(None), Mte, args_te) - Yte, dim=2).mean().item()
            print(f"    ep{ep+1:03d} test MAE={e:.3f}m")
    return model


def main():
    print("=" * 64); print("KalmanNet vs classical EKF vs scalar-alpha neural fusion"); print("=" * 64)
    env = setup_env()
    graph = corridor_graph(env[3])
    print("Generating datasets...")
    _, tr = make_dataset(250, 1, env, graph)
    test_raw, te = make_dataset(60, 2, env, graph)
    Mtr, Str, Ctr, Ktr, Ytr, Ydtr, Pstr = tr
    Mte, Ste, Cte, Kte, Yte, Ydte, Pste = te
    t = lambda a: torch.tensor(a, dtype=torch.float32)

    # centred frames (start -> 0) for the learned filters
    def center(M, S, Y, start):
        return t(M), t(S - start[:, None, :]), t(Y - start[:, None, :])
    Mtr_t, Str_c, Ytr_c = center(Mtr, Str, Ytr, Pstr)
    Mte_t, Ste_c, Yte_c = center(Mte, Ste, Yte, Pste)
    Ktr_t, Kte_t = t(Ktr), t(Kte)
    Ctr_t, Cte_t, Ydtr_t = t(Ctr), t(Cte), t(Ydtr)
    print(f"train={len(Mtr)} test={len(Mte)} bins={T_BINS}\n")

    # ---- KalmanNet ----
    print("Training KalmanNet (matrix gain)...")
    knet = KalmanNet()
    fwd_knet = lambda m, b, M, A: m(M[b], A[0][b], A[1][b])
    knet = train_model(knet, fwd_knet, Mtr_t, (Str_c, Ktr_t), Ytr_c,
                       Mte_t, (Ste_c, Kte_t), Yte_c)

    # ---- scalar-alpha neural fusion (same data; explicit loop for its delta loss) ----
    print("\nTraining scalar-alpha neural fusion...")
    nfus = NeuralKalmanFusion()
    opt = optim.Adam(nfus.parameters(), lr=2e-3, weight_decay=1e-5); mse = nn.MSELoss()
    idx = np.arange(len(Mtr))
    st_tr = torch.zeros(len(Mtr), 2); st_te = torch.zeros(len(Mte), 2)
    for ep in range(150):
        nfus.train(); np.random.shuffle(idx)
        for s in range(0, len(idx), 32):
            b = idx[s:s+32]; opt.zero_grad()
            p, d, _ = nfus(Mtr_t[b], Str_c[b], Ctr_t[b], Ktr_t[b], st_tr[b])
            (mse(p, Ytr_c[b]) + 2.0*mse(d, Ydtr_t[b])).backward(); opt.step()
        if (ep+1) % 30 == 0:
            nfus.eval()
            with torch.no_grad():
                p,_,_ = nfus(Mte_t, Ste_c, Cte_t, Kte_t, st_te)
                print(f"    ep{ep+1:03d} test MAE={torch.norm(p-Yte_c,dim=2).mean().item():.3f}m")

    # ---- evaluate all on identical test walks ----
    knet.eval(); nfus.eval()
    with torch.no_grad():
        knet_err = torch.norm(knet(Mte_t, Ste_c, Kte_t) - Yte_c, dim=2).numpy().mean(1)
        p,_,_ = nfus(Mte_t, Ste_c, Cte_t, Kte_t, st_te)
        nfus_err = torch.norm(p - Yte_c, dim=2).numpy().mean(1)
    ekf_err = np.array([classical_ekf_bins(Mte[i], Ste[i], Cte[i], Kte[i], Yte[i], Pste[i]).mean()
                        for i in range(len(Mte))])
    wifi_err = np.array([np.linalg.norm(Ste[i]-Yte[i], axis=1).mean() for i in range(len(Mte))])

    def ci(a):
        a = np.array(a); return a.mean(), 1.96*a.std(ddof=1)/np.sqrt(len(a))
    print("\n--- per-walk MAE over 60 held-out walks (identical inputs) ---")
    res = [("WiFi-only", wifi_err), ("Classical EKF", ekf_err),
           ("Neural fusion (scalar a)", nfus_err), ("KalmanNet (matrix gain)", knet_err)]
    for name, arr in res:
        m, h = ci(arr); print(f"  {name:26s} MAE = {m:.2f} +/- {h:.2f} m   (median {np.median(arr):.2f})")
    print(f"\n  KalmanNet vs classical EKF: {100*(ekf_err.mean()-knet_err.mean())/ekf_err.mean():+.1f}%")
    print(f"  KalmanNet vs neural fusion: {100*(nfus_err.mean()-knet_err.mean())/nfus_err.mean():+.1f}%")

    # CDF plot
    plt.figure(figsize=(8, 5))
    for (name, arr), c in zip(res, ["tab:orange", "tab:blue", "tab:green", "tab:red"]):
        xs = np.sort(arr); plt.plot(xs, np.linspace(0, 1, len(xs)), color=c, label=f"{name} ({arr.mean():.2f}m)")
    plt.xlabel("per-walk mean error (m)"); plt.ylabel("CDF")
    plt.title("Fusion mechanisms (60 synthetic walks, identical inputs)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("../Datasets/stage3_kalmannet.png", dpi=170)
    print("\nSaved -> Datasets/stage3_kalmannet.png")


if __name__ == "__main__":
    main()
