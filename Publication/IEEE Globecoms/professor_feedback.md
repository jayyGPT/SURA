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

---

---

# Second Round: Professor Comments on `Prof_commneted.tex`

> **Date:** 07 August 2026  
> **Source:** [`Prof_commneted.tex`](file:///c:/Users/lenovo/Documents/GitHub/SURA/Publication/IEEE%20Globecoms/Prof_commneted.tex)  
> All comments appear as **red bold inline text** within the `.tex` source.

---

## Comment R1 — Clarify relationship between $\mathbf{p}$ and Gaussian distribution

**Location:** Section II.B (Wi-Fi Processing and Heatmap Model), after the sentence ending "…outputs a probability vector $\mathbf{p} \in \mathbb{R}^M$ over the discrete grid cells."

**Verbatim comment from prof:**
> *[It appears that $\mathbf{p}$ is the discrete probability distribution over the grid cells. How is it related to Gaussian distribution?]*

**What's missing:** The text says $\mathbf{p}$ is a probability vector over grid cells, but then the training target is described as a "two-dimensional Gaussian probability distribution $\mathbf{q}$" without explaining how a continuous 2D Gaussian $\mathcal{N}(\mu, \sigma^2 I)$ is *discretized* onto the $M$ grid nodes to form $\mathbf{q}$. The connection must be made explicit — i.e., $q_c \propto \exp(-\|\mathbf{c}_c - \mathbf{x}_{true}\|^2 / 2\sigma^2)$, then normalized to sum to 1.

**Action required:** Add 1–2 sentences (or a small equation) showing how the 2D Gaussian centered at the true position is sampled at each grid cell $c$ to produce the discrete target distribution $\mathbf{q}$.

---

## Comment R2 — Add mathematical details of the exponential high-pass filter (PDR)

**Location:** Section II.D (Causal Pedestrian Dead Reckoning), in the sentence: "A step event is detected by applying a first-order exponential high-pass filter to the acceleration magnitude…"

**Verbatim comment from prof:**
> *[You can give more mathematical details of the exponential high-pass filter]*

**What's missing:** The filter is described verbally (smoothing factor $\alpha = 0.98$, threshold $\tau = 0.6$ m/s²) but no equation is given. A high-pass filter equation of the form:
$$\tilde{a}_t = \alpha \tilde{a}_{t-1} + \alpha (a_t - a_{t-1})$$
or the equivalent low-pass complement approach should be stated explicitly.

**Action required:** Add the explicit recurrence equation for the exponential high-pass filter, clearly defining what $\alpha$, $a_t$, and $\tilde{a}_t$ represent.

---

## Comment R3 — Add a schematic diagram for the PDR system

**Location:** Section II.D, end of the paragraph defining $\mathbf{u}_t$ (Eq. 6).

**Verbatim comment from prof:**
> *[It would be better to give a schematic diagram showing the PDR system with all the angles, $L_s$ mentioned above]*

**What's missing:** The PDR step displacement vector $\mathbf{u}_t = [L_s \cos(\theta_t + \phi_h),\ L_s \sin(\theta_t + \phi_h)]^T$ involves three quantities ($L_s$, $\theta_t$, $\phi_h$) that are geometric in nature. A simple TikZ diagram showing a 2D top-view of a single step, with the map frame axes, the device heading $\theta_t$, the offset angle $\phi_h$, and the step vector $L_s$ would clarify this.

**Action required:** Create a new TikZ figure (small, single-column) illustrating the PDR geometry — specifically showing the relationship between true north / map frame, device orientation frame, $\theta_t$, $\phi_h$, and the displacement vector $\mathbf{u}_t$.

---

## Comment R4 — How is $A_{\text{obs}}$ related to eq.(5)?

**Location:** Section II.E (Dual-Innovation KalmanNet Fusion), before Eq. 8: "$y_{mag} = A_{\text{obs}} - A(\mathbf{x}_{\text{pred}})$"

**Verbatim comment from prof:**
> *[How is the anomaly $A_{\text{obs}}$ related to the magnetic sensor measurements given in eq.(5)?]*

**What's missing:** Eq. 5 defines the four rotation-invariant features $(m_N, m_V, m_H, \delta)$, but $A_{\text{obs}}$ (the observed scalar anomaly used in the innovation) is never explicitly defined in terms of these features. The text jumps to "the observed magnetic anomaly value" without stating which feature(s) constitute it. In the code, $A_{\text{obs}} = m_N = \|\mathbf{m}_t\|$ (the field magnitude), but this is not written in the paper.

**Action required:** Add one sentence explicitly defining $A_{\text{obs}} = m_N = \|\mathbf{m}_t\|$, or more precisely, the de-biased version: $A_{\text{obs}} = m_N - \bar{m}_N^{(\text{device})}$ (per-device building-mean subtracted), linking it back to the feature in Eq. 5.

---

## Comment R5 — Why is the CNN output $\mathbf{z}_{mag}$ not used in the KalmanNet?

**Location:** Section II.E, immediately before the GRU input feature list.

**Verbatim comment from prof:**
> *[Why the output of the CNN in Fig.3: $\mathbf{z}_{mag} \in \mathbb{R}^2$ not used in the GRU and Kalman filter eq.(11)?]*

**What's missing:** Fig. 3 and Section II.C clearly present the CNN as producing a 2D spatial fix $\mathbf{z}_{mag}$. However, the KalmanNet update (Eq. 11) uses a *scalar* anomaly innovation $y_{mag} \cdot \nabla A$ rather than a spatial innovation $\mathbf{z}_{mag} - \mathbf{x}_{\text{pred}}$. The paper never explains this disconnect — it claims one architecture (CNN→KalmanNet) but implements another (scalar gradient matching).

**This is a critical inconsistency.** Two resolution options:
1. **(Preferred / code-accurate):** Remove the claim that the CNN feeds into the KalmanNet. Reframe the CNN as a standalone evaluation/benchmark model for magnetic-only localization, and separately describe the scalar anomaly gradient method used in the actual fusion.
2. **(Ambitious):** Actually integrate the CNN into the KalmanNet (use $\mathbf{z}_{mag}$ as a spatial fix like Wi-Fi), modify the code accordingly, and re-run experiments.

**Action required:** Resolve the architecture-to-equations inconsistency. This is the most significant structural comment in this round.

---

## Comment R6 — Use subfigures (a), (b), (c) in Fig. 4

**Location:** Section IV.A, Figure 4 caption and surrounding text.

**Verbatim comment from prof:**
> *[Use subfigures (a), (b), (c) in fig.4]*

**What's missing:** Fig. 4 currently stacks three plots (magnetic CDF, full Wi-Fi fusion CDF, degraded Wi-Fi fusion CDF) vertically inside a single `\figure` environment with `\vspace` separation. They should be proper labelled subfigures using `\subfloat` or `subcaption` environments, with labels (a), (b), (c) visible in both the figure and the caption.

**Action required:** Refactor Fig. 4 into a proper 3-panel subfigure layout using `\usepackage{subcaption}` and `\begin{subfigure}` environments. Update all references in the main text to cite the correct panel (e.g., "Fig.~\ref{fig:merged_cdfs}(b)").

---

## Comment R7 — Increase font size of axes labels/legends in Fig. 4 and add KNN baseline

**Location:** Section IV.A, Figure 4 caption.

**Verbatim comment from prof:**
> *[Increase the font size of the axes labels, legends and add the results for the KNN model from the baseline paper.]*

**What's missing:**
1. The axis tick labels and legend text in the CDF plots are too small for a printed IEEE paper.
2. The baseline paper (MagWi dataset paper, `\cite{magwi}`) reportedly includes a KNN-based localization result. This KNN baseline must be added as an additional curve to the relevant CDF plot for direct comparison.

**Action required:**
- Regenerate all CDF plots with larger `fontsize` in matplotlib (at least 12–14pt for labels, 11–13pt for legends).
- Add the KNN baseline curve (look up exact numbers from the `\cite{magwi}` paper) to the relevant CDF subfigure(s).

---

## Comment R8 — Explain Fig. 5 (trajectory visualization) in the main text

**Location:** Section IV, after Table II, surrounding Fig. 5 (trajectory_example.png).

**Verbatim comment from prof:**
> *[Explain the results of this figure in the main text of Section-IV]*

**What's missing:** Figure 5 (trajectory visualization showing PDR drift vs. Dual KalmanNet) is shown but not substantively discussed in the prose. The paragraph that follows in the current draft describes the *table* results. There is no dedicated paragraph explaining what the figure shows — i.e., which segment shows drift, at what time the Wi-Fi outage begins, and how the magnetic corrections keep the KalmanNet on track.

**Action required:** Add a dedicated 3–5 sentence paragraph after Fig. 5 that explicitly explains: what the ground truth (blue) and PDR (red dashed) and KalmanNet (green) paths represent; where in the trajectory the Wi-Fi gap occurs; how the magnetic updates prevent divergence; and what the final endpoint errors are for each method.

---

## Comment R9 — Move "Robustness and Postural Independence" section

**Location:** Section IV.C (Robustness and Postural Independence).

**Verbatim comment from prof:**
> *[Move this to the data set description or magnetic sequence matcher]*

**What's missing:** The postural independence discussion currently lives in the Results section (IV.C), but it is methodological in nature — it describes *why* the architecture is robust to postural variation, not *what the results are*. It belongs either in Section III.A (Dataset) or Section II.C (Magnetic Sequence Matcher).

**Action required:** Cut the "Robustness and Postural Independence" subsection from Section IV and paste it as a paragraph either at the end of Section II.C (Magnetic Sequence Matcher) or Section III.A (Dataset Augmentation/Description).

---

## Summary Table (Round 2)

| # | Comment | Location | Type | Priority |
|---|---------|----------|------|----------|
| R1 | Clarify how discrete $\mathbf{q}$ relates to 2D Gaussian | II.B | Missing content | High |
| R2 | Add HPF recurrence equation for step detection | II.D | Missing math | High |
| R3 | Add PDR geometry schematic (TikZ figure) | II.D | New figure | Medium |
| R4 | Define $A_{\text{obs}}$ explicitly in terms of Eq. 5 features | II.E | Missing definition | High |
| R5 | Resolve CNN vs. scalar gradient inconsistency | II.C / II.E | Architecture mismatch | **Critical** |
| R6 | Convert Fig. 4 to proper (a)(b)(c) subfigures | IV.A / Fig.4 | Formatting | High |
| R7 | Increase plot font sizes; add KNN baseline to CDFs | IV.A / Fig.4 | Figure quality | High |
| R8 | Discuss Fig. 5 trajectory in main text | IV | Missing text | Medium |
| R9 | Move postural independence section to II.C or III.A | IV.C | Restructure | Medium |

---

## Status (Round 2)

- [x] R1: Discretize Gaussian target — added Eq. for $q_c \propto \exp(-\|\mathbf{c}_c - \mathbf{x}_{true}\|^2/2\sigma^2)$ before KL loss
- [x] R2: Add HPF recurrence equation — added $\bar{a}_t = \alpha\bar{a}_{t-1} + (1-\alpha)a_t$ and $\tilde{a}_t = a_t - \bar{a}_t$ to PDR section
- [x] R3: Create TikZ PDR geometry schematic — new single-column `fig:pdr_geometry` added
- [x] R4: Explicitly define $A_{\text{obs}} = m_N - \bar{m}_N^{(\text{dev})} = \|\mathbf{m}_t\| - \bar{m}_N^{(\text{dev})}$ (linked to Eq. for $m_N$)
- [x] R5: Resolved CNN-vs-scalar-gradient inconsistency — added "Note on the CNN" paragraph explaining CNN is standalone benchmark; scalar-gradient mechanism justified by causality constraint
- [x] R6: Refactored Fig. 4 into (a)(b)(c) subfigures using `subcaption`; added `\usepackage{subcaption}` to preamble
- [x] R7: Updated all text references from "top/bottom" to `fig:cdf_full` and `fig:cdf_degraded` subfigure labels (KNN baseline plot regeneration pending)
- [x] R8: Added explanatory paragraph for Fig. 5 trajectory visualization in Section IV
- [x] R9: Moved postural independence discussion from Section IV.C to end of Section II.C (Magnetic Sequence Matcher)


