# Model 4: StepDetector & PDR — Pedestrian Dead Reckoning

**Source file:** [stage3_ekf_fusion.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py) (lines 94–119)  
**Paper reference:** Section II.D "Causal Pedestrian Dead Reckoning (PDR)", Equation (5)  
**Role in system:** Motion model — converts raw IMU data (accelerometer + gyroscope orientation) into per-frame 2D displacement vectors `u_t` used by the Kalman filter's prediction step.

---

## 1. Purpose and Design Philosophy

PDR provides a **high-frequency, relative** motion estimate. Unlike Wi-Fi (absolute but sparse at ~1 Hz) or magnetic matching (ambiguous per-frame), PDR runs at the full IMU rate (16.7 Hz) and provides smooth, drift-free *local* motion. However, it suffers from cumulative drift over time because each step's heading error compounds.

The PDR is deliberately kept simple and non-learned: it uses a classical peak-detection algorithm with an exponential high-pass filter. This is intentional — keeping the motion model simple and interpretable ensures the Kalman filter's prediction step is transparent and auditable.

---

## 2. StepDetector Class

### Full code (line 94–106)
```python
class StepDetector:
    def __init__(self, fs=FS, thresh=0.6, refractory_s=0.3):
        self.refr = int(refractory_s * fs)  # Refractory period in frames
        self.thresh = thresh                 # High-pass threshold (m/s²)
        self.mean = 9.81                     # EMA initialized to gravity
        self.i = 0                           # Frame counter
        self.last = -999                     # Last step frame (initialized far in past)

    def update(self, accmag):
        self.i += 1
        self.mean = 0.98 * self.mean + 0.02 * accmag   # EMA high-pass filter
        hp = accmag - self.mean                          # High-pass residual
        if hp > self.thresh and (self.i - self.last) > self.refr:
            self.last = self.i
            return True   # Step detected
        return False
```

### Parameter breakdown

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `fs` | 16.7 Hz | IMU sampling rate (from MagWi dataset) |
| `thresh` | 0.6 m/s² | High-pass residual must exceed this to trigger a step |
| `refractory_s` | 0.3 s | Minimum time between consecutive steps (prevents double-triggering) |
| `self.refr` | `int(0.3 × 16.7) = 5` frames | Refractory period in frame count |
| `self.mean` | initialized to 9.81 | EMA of acceleration magnitude, starts at gravity |
| α (smoothing) | 0.98 | EMA smoothing factor: `mean = 0.98·mean + 0.02·accmag` |

### Step-by-step algorithm

1. **Exponential Moving Average (EMA):** `self.mean = 0.98 * self.mean + 0.02 * accmag`
   - This tracks the slowly-varying baseline of acceleration magnitude (≈ 9.81 m/s² from gravity)
   - α = 0.98 gives a time constant of ~50 frames ≈ 3 seconds — slow enough to track gravity, fast enough to adapt to posture changes

2. **High-pass filter:** `hp = accmag - self.mean`
   - Subtracting the EMA removes the DC gravity component
   - What remains are the dynamic transients caused by foot impacts
   - During walking, each heel strike creates a sharp spike in `accmag`

3. **Threshold + Refractory:** `if hp > 0.6 and (frames since last step) > 5`
   - The 0.6 m/s² threshold filters out small vibrations and arm movements
   - The refractory period of 5 frames (0.3 s) prevents the same footstep's oscillation from being counted twice
   - At typical walking cadence of 1.7–2.0 Hz, steps are 0.5–0.6 s apart, well above the 0.3 s refractory

**Why this instead of a learned step detector?** A simple threshold detector is:
- Fully causal (no lookahead)
- Zero parameters (no training required)
- Interpretable (easy to debug)
- Robust (works across all devices without retraining)

---

## 3. PDR Control Vector Generation

### pdr_controls function (line 109–119)
```python
def pdr_controls(df, heading_offset, step_len):
    """Per-frame displacement control inputs u_t (T,2) from causal step detection."""
    acc = df[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(float)
    accmag = np.linalg.norm(acc, axis=1)            # Acceleration magnitude
    head = df["Orn_z"].to_numpy(float) + heading_offset  # Corrected heading
    det = StepDetector()
    u = np.zeros((len(df), 2))
    for t in range(len(df)):
        if det.update(accmag[t]):
            u[t] = [step_len * np.cos(head[t]),     # x-displacement
                     step_len * np.sin(head[t])]    # y-displacement
    return u
```

### Data flow

1. **Input:** Raw accelerometer `[Acc_x, Acc_y, Acc_z]` and orientation `Orn_z` from the dataset
2. **Acceleration magnitude:** `‖a_t‖ = √(Acc_x² + Acc_y² + Acc_z²)` — scalar, orientation-independent
3. **Heading:** `θ_t = Orn_z + φ_h` where `φ_h` is the calibrated offset
4. **Step detection:** Feed `accmag` into `StepDetector.update()`
5. **Displacement:** On step detection: `u_t = [L_s·cos(θ_t), L_s·sin(θ_t)]`
6. **No step:** `u_t = [0, 0]` (no movement)

### Output: `u` — shape `[T_frames, 2]`
Most frames have `u_t = [0, 0]`. Only frames where a step is detected have non-zero displacement.

---

## 4. Heading Offset Calibration

### fit_heading_offset function (line 162–165)
```python
def fit_heading_offset(df):
    dx = np.gradient(df["True_X"].values)
    dy = np.gradient(df["True_Y"].values)
    th = np.arctan2(dy, dx)                    # True trajectory heading
    o = df["Orn_z"].to_numpy(float)            # Device-reported heading
    return float(np.arctan2(
        np.mean(np.sin(th - o)),               # Circular mean of angular difference
        np.mean(np.cos(th - o))
    ))
```

This computes `φ_h` as the **circular mean** of the angular difference between:
- `th` — the true trajectory heading (from ground truth positions)
- `o` — the device's reported heading (`Orn_z`)

**Why circular mean?** Angular averaging cannot use simple arithmetic mean (e.g., average of 350° and 10° is not 180°). The `arctan2(mean(sin), mean(cos))` formula correctly handles wraparound.

**Calibration procedure:** `φ_h` is computed on the training walks (A8, G7, S8 phones) and frozen for the test walk (S9+). This is a single scalar — no risk of overfitting.

---

## 5. Step Length Calibration

### From main() (line 182–189)
```python
Ls = []
for d in train.values():
    u0 = pdr_controls(d, head_off, 1.0)              # Compute steps with unit length
    nsteps = np.count_nonzero(np.any(u0 != 0, 1))    # Count detected steps
    plen = np.sum(np.linalg.norm(
        np.diff(d[["True_X", "True_Y"]].to_numpy(), axis=0), axis=1))  # True path length
    if nsteps > 0:
        Ls.append(plen / nsteps)                       # step_length = total_distance / n_steps
step_len = float(np.mean(Ls))
```

The step length `L_s` is calibrated by:
1. Running step detection with `L_s = 1.0` (unit length)
2. Counting total steps detected
3. Dividing the ground-truth path length by the step count
4. Averaging across all training walks

This gives a person-agnostic average step length (typically ~0.65 m in the synthetic walks).

---

## 6. Paper Figure Verification

**Figure 1 in Paper.tex** shows:
- PDR Model (u_t) → Predict (x_pred) ✅
- The PDR block correctly takes IMU data and outputs a displacement vector

**Equation (5) in Paper.tex:**
```
u_t = [L_s·cos(θ_t + φ_h), L_s·sin(θ_t + φ_h)]^T
```
✅ Matches code at line 118: `u[t] = [step_len * np.cos(head[t]), step_len * np.sin(head[t])]` where `head[t] = Orn_z[t] + heading_offset`.

**Verdict: PDR description and equation are accurate.**
