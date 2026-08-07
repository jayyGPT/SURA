# Professor's Feedback — All Comments

> Extracted from [FeedBack from Professor.pdf](file:///c:/Users/lenovo/Documents/GitHub/SURA/Publication/IEEE%20Globecoms/FeedBack%20from%20Professor.pdf)  
> All comments appear as **blue inline text** on Page 2 (Section II — Proposed System Architecture).

---

## Comment 1 — Notation Convention (Vectors & Matrices)

**Location:** Section II.A, near "magnetic field vector $\mathbf{M}_t \in \mathbb{R}^3$"

**Comment:**
> **(Notation: Use boldface small case letters for vectors and boldface capital letters for matrices)**

**Action required:** Audit all notation throughout the paper. Vectors like $\mathbf{M}_t$ should use lowercase boldface (e.g., $\mathbf{m}_t$), and matrices (like covariance $\mathbf{R}$, gain $\mathbf{K}$) should use uppercase boldface. Currently $\mathbf{M}_t$ (magnetic vector) is uppercase but should be lowercase since it's a vector.

---

## Comment 2 — Rename Normalized Wi-Fi Input Variable

**Location:** Section II.B, near "Wi-Fi scan $\mathbf{s}_t$ is normalized into $\mathbf{x}_{wifi}$"

**Comment:**
> **(Denote normalized $\mathbf{s}_t$ vector as $\tilde{\mathbf{s}}_t$ since $\mathbf{x}$ is used for the position vector)**

**Action required:** The normalized Wi-Fi input is currently called $\mathbf{x}_{wifi}$, but $\mathbf{x}$ is already the state/position vector. This creates ambiguity. Rename the normalized Wi-Fi input to $\tilde{\mathbf{s}}_t$ throughout.

---

## Comment 3 — Add Mathematical Expression for Clipping/Normalization

**Location:** Section II.B, near "by clipping to [-90, -30] dBm and rescaling"

**Comment:**
> **(Give the mathematical expression for the clipping and normalization operation)**

**Action required:** Add a formal equation showing:
$$\tilde{s}_{t,i} = \frac{\text{clip}(s_{t,i}, -90, -30) - (-90)}{-30 - (-90)} = \frac{\text{clip}(s_{t,i}, -90, -30) + 90}{60}$$
with absent APs ($s_{t,i} \leq -100$ dBm) mapped to 0.

---

## Comment 4 — Move Hyperparameters to Experimental Section

**Location:** Section II.B, near "Dropout regularization ($p=0.3$)" and "standard deviation $\sigma = 2.0$ m"

**Comment:**
> **[Mention the numerical values of p = 0.3 and σ = 2.0 m in the Experimental Results Section -III(B)]**

**Action required:** Remove the specific numerical hyperparameter values ($p=0.3$, $\sigma=2.0$ m) from the architecture description (Section II) and move them to the Training Details subsection in Section III(B).

---

## Comment 5 — Notation Convention (Vectors & Matrices) — Second Instance

**Location:** Section II.B, near "where $\mathbf{C}_c \in \mathbb{R}^2$"

**Comment:**
> **(Notation: Use boldface small case letters for vectors and boldface capital letters for matrices)**

**Action required:** $\mathbf{C}_c$ represents the physical coordinates of grid cell $c$ — this is a 2D vector, not a matrix. Rename to $\mathbf{c}_c$ (lowercase boldface).

---

## Comment 6 — Add Mathematical Expressions for Magnetic Feature Extraction

**Location:** Section II.C, near "the field magnitude $\|\mathbf{M}_t\|$, the vertical projection against estimated gravity, the horizontal component magnitude, and the magnetic dip angle"

**Comment:**
> **(Give mathematical expressions for extracting these magnetic features from $\mathbf{M}_t$)**

**Action required:** Add formal equations defining the 4 rotation-invariant features:
1. Field magnitude: $\|\mathbf{m}_t\|$
2. Vertical projection: $m_V = \frac{\mathbf{m}_t \cdot \hat{\mathbf{g}}_t}{\|\hat{\mathbf{g}}_t\|}$
3. Horizontal component: $m_H = \|\mathbf{m}_t - m_V \hat{\mathbf{g}}_t\|$
4. Dip angle: $\delta = \arctan\left(\frac{m_V}{m_H}\right)$

where $\hat{\mathbf{g}}_t$ is the estimated gravity direction from the accelerometer.

---

## Summary Table

| # | Issue | Section | Type | Priority |
|---|-------|---------|------|----------|
| 1 | Notation: vectors lowercase, matrices uppercase | Throughout | Formatting | High |
| 2 | Rename $\mathbf{x}_{wifi}$ to $\tilde{\mathbf{s}}_t$ | II.B | Notation | High |
| 3 | Add math for Wi-Fi clipping/normalization | II.B | Missing content | High |
| 4 | Move $p=0.3$, $\sigma=2.0$ to Section III(B) | II.B → III.B | Restructure | Medium |
| 5 | Rename $\mathbf{C}_c$ to $\mathbf{c}_c$ | II.B | Notation | High |
| 6 | Add math for magnetic feature extraction | II.C | Missing content | High |

---

## Status

- [x] Comment 1: Fix notation throughout — `M_t` → `m_t` (lowercase boldface for vectors)
- [x] Comment 2: Rename `x_{wifi}` → `\tilde{s}_t` (including figure label)
- [x] Comment 3: Add Wi-Fi normalization equation (Eq. 2 — piecewise clip+rescale)
- [x] Comment 4: Move `p=0.3` and `σ=2.0 m` to Section III(B) Training Details
- [x] Comment 5: Rename `C_c` → `c_c` (lowercase boldface for vector)
- [x] Comment 6: Add magnetic feature extraction equations (Eq. 4 — align environment, 2 lines)
