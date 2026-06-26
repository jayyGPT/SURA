# Walkthrough: Comprehensive Paper Audit & Corrections

## Scope
A line-by-line review of [Paper.tex](file:///c:/Users/lenovo/Documents/GitHub/SURA/Publication/IEEE%20Globecoms/Paper.tex) was performed against every Python source file in the repository:
- [stage2_wifi_heatmap.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_wifi_heatmap.py) — Wi-Fi MLP
- [stage2_mag_sequence.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage2_mag_sequence.py) — Magnetic 1D-CNN
- [stage3_ekf_fusion.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_ekf_fusion.py) — PDR + EKF
- [stage3_synthetic_eval.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_synthetic_eval.py) — Trajectory generation
- [stage3_dual_kalmannet.py](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/stage3_dual_kalmannet.py) — Dual KalmanNet GRU

## Critical Fixes Applied

### 1. Magnetic Features: 3 → 4
**Before:** \"we extract 3 rotation-invariant features (magnitude, vertical projection, horizontal component)\"  
**After:** \"we extract 4 rotation-invariant features per frame: the field magnitude, the vertical projection against estimated gravity, the horizontal component magnitude, and the magnetic dip angle\"  
**Evidence:** `MAG_FEATS = [\"magN\", \"magV\", \"magH\", \"dip\"]` at stage2_mag_sequence.py:36

### 2. Magnetic Innovation Equation (Most Critical)
**Before:** `y_mag = z_mag − x_pred` (wrong — implies a 2D spatial residual)  
**After:** `y_mag = A_obs − A(x_pred) ∈ ℝ` (correct — scalar anomaly field residual)  
**Evidence:** stage3_dual_kalmannet.py:213: `y_mag = mag_obs[:, t, :] - map_val`

### 3. MLP Dropout
**Before:** No mention of regularization.  
**After:** \"MLP with Dropout regularization (p=0.3)\" + Figure 2 boxes now show \"FC(256), ReLU / Dropout(0.3)\"  
**Evidence:** stage2_wifi_heatmap.py:78-79

### 4. CNN Dropout in pos_head
**Before:** Figure 3 showed \"FC(64), ReLU / FC(2)\"  
**After:** \"FC(64), ReLU / Drop(0.2), FC(2)\"  
**Evidence:** stage2_mag_sequence.py:156

### 5. GRU Feature Vector
**Before:** Listed `u_t` and omitted `dz_wifi`  
**After:** Correctly lists all 13: `y_wifi(2) + y_mag(1) + ∇A(2) + Δz_wifi(2) + u_t(2) + Δx(2) + m_wifi(1) + m_mag(1)`  
**Evidence:** stage3_dual_kalmannet.py:217

### 6. φ_h Description
**Before:** \"constant initial angular offset between the device's local magnetic north and the map's coordinate frame\"  
**After:** \"constant angular offset between the device's local orientation frame and the map's coordinate frame, calibrated as the mean angular difference between the device heading and the true trajectory direction\"  
**Evidence:** stage3_ekf_fusion.py:162-165 (fit_heading_offset function)

### 7. Eq. 6 Notation
**Before:** `K_mag (y_mag ∇A)` — ambiguous  
**After:** `K_mag (y_mag · ∇A)` with explicit dot product notation  
**Evidence:** stage3_dual_kalmannet.py:228: `(y_mag * grad)` = element-wise broadcast

## IEEE Style Fixes

| Fix | Description |
|-----|-------------|
| Author marks | Added `\textsuperscript{*}` to Utkarsh and Jayendra |
| PLACEHOLDERs | Removed both (loss plot + posture table); replaced with substantive text |
| Figure 3 sizing | Converted from `figure*` to `figure` (saves ~1 column of space) |
| Equation labels | Added `\label{eq:kl_loss}`, `\label{eq:nll_loss}` |
| Equation reference | Added `~(\ref{eq:kn_correct})` in the mask explanation text |
| Training details | Added exact epoch counts (80/60/150), weight decay values, scheduler params, window sweep candidates |
| Robustness section | Replaced vague placeholder with specific dataset modes: Navigation, Call listening, Swinging |

## Build Verification

```
pdflatex Paper.tex; bibtex Paper; pdflatex Paper.tex; pdflatex Paper.tex
→ Output: 5 pages, 448775 bytes
→ Warnings: 2 minor Overfull hbox (cosmetic, acceptable)
→ Undefined references: 0
→ Missing citations: 0
```

## Output Files
- [Paper.pdf](file:///c:/Users/lenovo/Documents/GitHub/SURA/Publication/IEEE%20Globecoms/Paper.pdf)
- [Paper.tex](file:///c:/Users/lenovo/Documents/GitHub/SURA/Publication/IEEE%20Globecoms/Paper.tex)
- [Ref.bib](file:///c:/Users/lenovo/Documents/GitHub/SURA/Publication/IEEE%20Globecoms/Ref.bib)
