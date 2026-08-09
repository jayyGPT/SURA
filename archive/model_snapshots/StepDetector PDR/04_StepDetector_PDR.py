"""
StepDetector & PDR — Pedestrian Dead Reckoning
================================================
Standalone, self-contained reproduction script.

Source: stage3_ekf_fusion.py (lines 94-119, 162-189)
Paper:  Section II.D, Equation (5)

This script defines the step detector, heading calibration,
step length calibration, and PDR control vector generation.
Run from the dl_models/ directory.
"""
import numpy as np
import pandas as pd

# ========================= CONSTANTS ========================================
FS = 16.7          # IMU sampling rate (Hz)
TRAIN_PHONES = ["A8", "G7", "S8"]
TEST_PHONE = "S9+"


# ========================= STEP DETECTOR ===================================
class StepDetector:
    """
    Causal step detector using exponential high-pass filter.

    Parameters:
        fs:           float  — IMU sampling rate (Hz)
        thresh:       float  — high-pass residual threshold (m/s²)
        refractory_s: float  — minimum time between steps (s)

    Internal state:
        mean: EMA of acceleration magnitude (initialized to 9.81 m/s²)
        i:    frame counter
        last: frame index of last detected step

    Algorithm:
        1. Update EMA: mean = 0.98 * mean + 0.02 * accmag
        2. High-pass filter: hp = accmag - mean
        3. If hp > thresh AND (i - last) > refractory_frames: step detected
    """
    def __init__(self, fs=FS, thresh=0.6, refractory_s=0.3):
        self.refr = int(refractory_s * fs)  # = 5 frames at 16.7 Hz
        self.thresh = thresh
        self.mean = 9.81    # EMA initialized to gravity
        self.i = 0          # frame counter
        self.last = -999    # last step frame (far in the past)

    def update(self, accmag):
        """
        Process one acceleration magnitude sample.

        Args:
            accmag: float — ||a_t|| = sqrt(ax² + ay² + az²)

        Returns:
            True if a step is detected, False otherwise.
        """
        self.i += 1
        # Exponential moving average (low-pass) of acceleration magnitude
        self.mean = 0.98 * self.mean + 0.02 * accmag
        # High-pass residual: removes gravity baseline
        hp = accmag - self.mean
        # Threshold + refractory check
        if hp > self.thresh and (self.i - self.last) > self.refr:
            self.last = self.i
            return True
        return False


# ========================= PDR CONTROLS =====================================
def pdr_controls(df, heading_offset, step_len):
    """
    Generate per-frame displacement vectors u_t from IMU data.

    Args:
        df:             DataFrame with columns [Acc_x, Acc_y, Acc_z, Orn_z]
        heading_offset: float — calibrated φ_h (radians)
        step_len:       float — calibrated L_s (metres)

    Returns:
        u: ndarray [T, 2] — displacement vectors.
           Most frames are [0, 0]; only step-detected frames have nonzero u.
    """
    acc = df[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(float)
    accmag = np.linalg.norm(acc, axis=1)        # ||a_t||
    head = df["Orn_z"].to_numpy(float) + heading_offset  # θ_t + φ_h

    det = StepDetector()
    u = np.zeros((len(df), 2))
    for t in range(len(df)):
        if det.update(accmag[t]):
            u[t] = [step_len * np.cos(head[t]),
                     step_len * np.sin(head[t])]
    return u


# ========================= HEADING CALIBRATION ==============================
def fit_heading_offset(df):
    """
    Compute the angular offset φ_h between device heading and map frame.

    Uses circular mean of the angular difference between:
        - True trajectory heading (from ground truth positions)
        - Device-reported heading (Orn_z)

    Args:
        df: DataFrame with columns [True_X, True_Y, Orn_z]

    Returns:
        float — heading offset in radians
    """
    dx = np.gradient(df["True_X"].values)
    dy = np.gradient(df["True_Y"].values)
    th = np.arctan2(dy, dx)               # true trajectory heading
    o = df["Orn_z"].to_numpy(float)       # device heading
    # Circular mean of angular difference
    return float(np.arctan2(
        np.mean(np.sin(th - o)),
        np.mean(np.cos(th - o)),
    ))


# ========================= STEP LENGTH CALIBRATION ==========================
def calibrate_step_length(train_walks, heading_offset):
    """
    Compute average step length by matching step count to true path length.

    Args:
        train_walks: dict of {phone: DataFrame}
        heading_offset: float — previously calibrated φ_h

    Returns:
        float — average step length in metres
    """
    Ls = []
    for d in train_walks.values():
        # Count steps with unit step length
        u0 = pdr_controls(d, heading_offset, 1.0)
        nsteps = np.count_nonzero(np.any(u0 != 0, 1))
        # True path length
        plen = np.sum(np.linalg.norm(
            np.diff(d[["True_X", "True_Y"]].to_numpy(), axis=0), axis=1
        ))
        if nsteps > 0:
            Ls.append(plen / nsteps)
    return float(np.mean(Ls))


# ========================= MAIN =============================================
def main():
    print("=" * 64)
    print("StepDetector & PDR — Pedestrian Dead Reckoning")
    print("=" * 64)

    # Load continuous walk data
    def load_walk(phone):
        return pd.read_csv(f"../Datasets/Continuous_Fused_{phone}.csv")

    train = {p: load_walk(p) for p in TRAIN_PHONES}

    # Calibrate heading offset
    head_off = float(np.mean([fit_heading_offset(d) for d in train.values()]))
    print(f"Calibrated heading offset: {head_off:.4f} rad "
          f"({np.degrees(head_off):.1f}°)")

    # Calibrate step length
    step_len = calibrate_step_length(train, head_off)
    print(f"Calibrated step length: {step_len:.3f} m")

    # Demo: generate PDR trajectory for S9+
    print(f"\n--- PDR Demo: {TEST_PHONE} ---")
    dft = load_walk(TEST_PHONE)
    u = pdr_controls(dft, head_off, step_len)

    n_steps = np.count_nonzero(np.any(u != 0, 1))
    total_disp = np.linalg.norm(u.sum(0))
    true_len = np.sum(np.linalg.norm(
        np.diff(dft[["True_X", "True_Y"]].to_numpy(), axis=0), axis=1
    ))
    print(f"  Steps detected: {n_steps}")
    print(f"  PDR total displacement: {total_disp:.1f} m")
    print(f"  True path length: {true_len:.1f} m")

    # Compute PDR trajectory
    pos = dft[["True_X", "True_Y"]].to_numpy()[0].copy()
    track = np.zeros((len(dft), 2))
    for t in range(len(dft)):
        pos = pos + u[t]
        track[t] = pos

    # Error
    true_pos = dft[["True_X", "True_Y"]].to_numpy()
    err = np.linalg.norm(track - true_pos, axis=1)
    print(f"  PDR-only MAE: {err.mean():.2f} m")
    print(f"  PDR-only final error: {err[-1]:.2f} m (cumulative drift)")


if __name__ == "__main__":
    main()
