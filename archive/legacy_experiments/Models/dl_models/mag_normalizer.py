"""
Online, causal magnetic normalizer — train==deploy by construction.

Why: raw magnetometer readings are device-dependent (hard-iron offset + soft-iron
scale distortion). We must NOT normalize with dataset-global per-device statistics,
or an unseen phone has no calibration and breaks. Instead every phone calibrates
itself from its OWN live stream:

  1. Rolling hard/soft-iron calibration: as the user walks, the phone rotates and
     the magnetometer samples a sphere. We fit an ellipsoid to a trailing buffer
     of raw samples and map it back to a sphere (removes device bias + distortion).
     Falls back to a numerically-stable sphere fit (hard-iron only) when the point
     cloud lacks rotational diversity or the ellipsoid fit is ill-conditioned.

  2. Causal running normalization: a trailing EMA mean/std on the rotation-invariant
     features yields the *relative* magnetic anomaly map (kills slow device +
     environmental DC drift). This is the signal that is spatially unique.

Everything uses PAST samples only (strictly causal) so the exact same transform
runs in preprocessing and at inference. No look-ahead.

Rotation-invariant features (orientation-independent, gravity from accelerometer):
    magN   = |M|                          magnitude
    magV   = M . g_hat                     world-vertical component (signed)
    magH   = sqrt(|M|^2 - magV^2)          horizontal magnitude
    dip    = atan2(magV, magH)             inclination angle (rad)
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Calibration fits (operate on a buffer of PAST raw magnetometer samples)
# --------------------------------------------------------------------------- #
def sphere_fit(P):
    """Hard-iron + uniform scale. Solve |p - c|^2 = r^2. Returns (center(3,), radius)."""
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    b = np.sum(P ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    return c, float(np.sqrt(max(r2, 1e-6)))


def ellipsoid_fit(P):
    """
    Full hard+soft-iron. Fit general quadric, return (center(3,), W(3x3), radius)
    such that calibrated = W @ (p - center) lies on a sphere of `radius`.
    Returns None if ill-conditioned / not positive-definite.
    """
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    D = np.column_stack([x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z])
    try:
        v, *_ = np.linalg.lstsq(D, np.ones(len(P)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c, f, g, h, p, q, r = v
    A3 = np.array([[a, h, g], [h, b, f], [g, f, c]])
    # A3 must be positive-definite for a real ellipsoid
    eig = np.linalg.eigvalsh(A3)
    if np.any(eig <= 1e-9) or (eig.max() / eig.min() > 1e4):
        return None
    try:
        center = -np.linalg.solve(A3, np.array([p, q, r]))
    except np.linalg.LinAlgError:
        return None
    # constant term at center -> scale of the quadric
    k = 1.0 + center @ A3 @ center  # since quadric == 1 at surface
    if k <= 1e-9:
        return None
    # W maps ellipsoid -> unit sphere; rescale to physical radius later
    A3n = A3 / k
    evals, evecs = np.linalg.eigh(A3n)
    if np.any(evals <= 1e-12):
        return None
    W = evecs @ np.diag(np.sqrt(evals)) @ evecs.T  # symmetric soft-iron correction
    # radius after this W is 1; report 1.0 and let caller rescale
    return center, W, 1.0


# --------------------------------------------------------------------------- #
# Streaming normalizer
# --------------------------------------------------------------------------- #
class OnlineMagNormalizer:
    def __init__(self, buffer_size=600, refit_every=50, min_points=80,
                 ema_alpha=0.02, warmup=60, diversity_thresh=0.15):
        """
        buffer_size       trailing raw-sample buffer used for calibration
        refit_every       recompute calibration every N samples
        min_points        min buffer size before attempting a fit
        ema_alpha         EMA rate for running mean/std (smaller = slower)
        warmup            samples before running-norm output is considered stable
        diversity_thresh  min (min/max) eigenvalue ratio of the mag point cloud to
                          trust an ellipsoid fit (i.e. the phone actually rotated)
        """
        self.buffer_size = buffer_size
        self.refit_every = refit_every
        self.min_points = min_points
        self.ema_alpha = ema_alpha
        self.warmup = warmup
        self.diversity_thresh = diversity_thresh
        self.reset()

    def reset(self):
        self._buf = []                  # trailing raw mag samples
        self._n = 0
        self.center = np.zeros(3)       # current hard-iron estimate
        self.W = np.eye(3)              # current soft-iron correction
        self.scale = 1.0                # physical radius to preserve units
        self._calibrated = False
        # EMA stats for the 4 invariant features
        self._mean = None
        self._var = None

    # ----- calibration -----
    def _maybe_refit(self):
        if len(self._buf) < self.min_points:
            return
        P = np.asarray(self._buf)
        # rotational diversity: spread of the point cloud across axes
        cov = np.cov((P - P.mean(0)).T)
        ev = np.linalg.eigvalsh(cov)
        diverse = ev.min() / max(ev.max(), 1e-9) > self.diversity_thresh

        fit = ellipsoid_fit(P) if diverse else None
        if fit is not None:
            center, W, _ = fit
            # preserve physical scale: median calibrated magnitude -> median raw magnitude
            cal = (W @ (P - center).T).T
            med_cal = np.median(np.linalg.norm(cal, axis=1))
            med_raw = np.median(np.linalg.norm(P - center, axis=1))
            self.center, self.W = center, W
            self.scale = (med_raw / med_cal) if med_cal > 1e-6 else 1.0
            self._calibrated = True
        else:
            c, r = sphere_fit(P)
            self.center, self.W, self.scale = c, np.eye(3), 1.0
            self._calibrated = True

    def _calibrate(self, mag):
        return self.scale * (self.W @ (mag - self.center))

    # ----- one streaming step -----
    def update(self, mag, acc):
        """Ingest one frame (mag(3,), acc(3,)). Returns dict of causal features."""
        mag = np.asarray(mag, float)
        acc = np.asarray(acc, float)
        self._buf.append(mag)
        if len(self._buf) > self.buffer_size:
            self._buf.pop(0)
        self._n += 1
        if self._n % self.refit_every == 0 or (not self._calibrated and len(self._buf) >= self.min_points):
            self._maybe_refit()

        m = self._calibrate(mag)
        a_norm = np.linalg.norm(acc)
        g_hat = acc / a_norm if a_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        magN = float(np.linalg.norm(m))
        magV = float(m @ g_hat)
        magH = float(np.sqrt(max(magN ** 2 - magV ** 2, 0.0)))
        dip = float(np.arctan2(magV, magH))
        feat = np.array([magN, magV, magH, dip])

        # causal EMA running normalization -> relative anomaly
        if self._mean is None:
            self._mean = feat.copy()
            self._var = np.ones(4)
        else:
            a = self.ema_alpha
            self._mean = (1 - a) * self._mean + a * feat
            self._var = (1 - a) * self._var + a * (feat - self._mean) ** 2
        rel = (feat - self._mean) / np.sqrt(self._var + 1e-6)

        return {
            "calib_magN": magN, "calib_magV": magV, "calib_magH": magH, "dip": dip,
            "rel_magN": rel[0], "rel_magV": rel[1], "rel_magH": rel[2], "rel_dip": rel[3],
            "calibrated": self._calibrated, "stable": self._n >= self.warmup,
        }

    def process_stream(self, mag_arr, acc_arr, reset=True):
        """Run a whole session causally. Returns (T, 8) feature matrix:
        [calib_magN, calib_magV, calib_magH, dip, rel_magN, rel_magV, rel_magH, rel_dip]."""
        if reset:
            self.reset()
        out = np.empty((len(mag_arr), 8), dtype=np.float32)
        for t in range(len(mag_arr)):
            r = self.update(mag_arr[t], acc_arr[t])
            out[t] = [r["calib_magN"], r["calib_magV"], r["calib_magH"], r["dip"],
                      r["rel_magN"], r["rel_magV"], r["rel_magH"], r["rel_dip"]]
        return out


# --------------------------------------------------------------------------- #
# Self-test / validation on the continuous walks
# --------------------------------------------------------------------------- #
def _validate():
    import pandas as pd
    print("Validating OnlineMagNormalizer on continuous walks (cross-device |M|)\n")
    phones = ["A8", "G7", "S8", "S9+"]
    raw_means, cal_means = {}, {}
    for p in phones:
        path = f"../Datasets/Continuous_Fused_{p}.csv"
        df = pd.read_csv(path)
        mag = df[["Mag_x", "Mag_y", "Mag_z"]].to_numpy(float)
        acc = df[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(float)
        norm = OnlineMagNormalizer()
        feat = norm.process_stream(mag, acc)
        raw_means[p] = np.linalg.norm(mag, axis=1).mean()
        cal_means[p] = feat[:, 0].mean()  # calib_magN
        print(f"  {p:4s}  raw |M| mean={raw_means[p]:6.2f}   calibrated |M| mean={cal_means[p]:6.2f}")
    rv = np.std(list(raw_means.values()))
    cv = np.std(list(cal_means.values()))
    print(f"\n  cross-device spread of mean |M|:  raw std={rv:.2f}  ->  calibrated std={cv:.2f}")
    print(f"  {'IMPROVED' if cv < rv else 'NO IMPROVEMENT'} (calibration should shrink cross-device spread)")


if __name__ == "__main__":
    _validate()
