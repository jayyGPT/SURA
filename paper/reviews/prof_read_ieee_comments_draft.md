# Professor Read / IEEE Comments Draft

Working checklist collected during final paper proofreading. These are **questions and potential issues to investigate together before changing the manuscript**. A checked or confident-sounding statement here should not be treated as a final methodological conclusion until we resolve it against the active code and experiment protocol.

## Priority A — methodology / validity

### [ ] P1. PDR heading source, `Orn_z`, and heading-offset calibration (observations 1 + 10)

**Concern.** The manuscript currently describes device yaw `Orn_z` corrected by a fixed heading offset `phi_h`, calibrated from ground-truth training trajectories. We need to verify whether this is actually what the final experiment does, whether `Orn_z` is sufficiently available in MagWi, and whether any offset calibration ever touches test information.

**Preliminary code finding.** The active final fusion benchmark does **not** use dataset `Orn_z` or `fit_heading_offset()`. `synthesize_walk()` derives a heading from the synthetic ground-truth path tangent and adds random-walk drift plus white noise; `build_sequence()` then uses this synthetic heading directly for PDR. The manuscript therefore appears inconsistent with the active benchmark. `models/pdr.py` still contains a generic `fit_heading_offset()` helper, but that helper is not used by the current fusion benchmark.

**Dataset note.** The repository dataset audit reports substantial missingness in `Orn_z`, so the paper's current wording about `Orn_z` must be revisited even if we later design a real-data PDR evaluation.

**Decision needed.** Decide whether the paper should describe the actual synthetic-heading protocol, or whether we should implement and validate a truly causal heading-estimation/calibration method for real inference.

### [ ] P2. Step-length calibration and online realism (observation 2)

**Concern.** The paper says step length `L_s` is calibrated from training trajectories and frozen. We need to decide what is scientifically acceptable and what a deployable system should do.

**Preliminary code finding.** The active fusion benchmark does **not** estimate step length from the training walks; it uses a fixed constant `STEP_LENGTH_M = 0.65`. `models/pdr.py` contains a ground-truth-based `calibrate_step_length()` helper, but the current benchmark does not call it.

**Important distinction.** Estimating a hyperparameter from the **training set only** is normally legal and is not test leakage. However, ground-truth-based calibration is not automatically deployable to an unseen user. A rolling/online step-length estimate would be more practical, but it cannot use ground-truth distance at inference; it would need inertial features and/or trusted absolute Wi-Fi/magnetic corrections.

**Decision needed.** Align paper and code, then decide whether to keep a fixed nominal step length as a controlled simulation parameter or implement a causal online estimator.

### [ ] P3. Full data-augmentation / synthetic-evaluation audit (observation 11)

We need a line-by-line conceptual audit of how the synthetic fusion dataset is generated and whether any information unavailable at inference is used.

Current pipeline to inspect:
1. Construct an epsilon-neighbour corridor graph from surveyed node coordinates.
2. Randomly choose endpoints and use graph shortest paths.
3. Interpolate continuous ground-truth positions along the path.
4. Generate heading from the **ground-truth path tangent**, then add drift and white noise.
5. Generate synthetic acceleration magnitude using gait-frequency sinusoids plus noise; detect steps causally.
6. Use fixed 0.65 m step displacement along the synthetic heading.
7. For Wi-Fi observations, find the fingerprint node nearest the true position, sample a stored real RSSI scan there, optionally apply AP dropout, and pass it through the trained Wi-Fi heatmap.
8. Build a four-channel interpolated magnetic map (`magN`, `magV`, `magH`, `dip`), sample it along the true trajectory, add estimated map noise, and run the actual 84-frame magnetic CNN causally.
9. Bin the raw stream into 160 fusion time steps and train/test KalmanNet on separate random synthetic walks.

**Potential high-priority issue to verify.** `setup_environment()` currently loads the full processed fingerprint database; the Wi-Fi scan pool and magnetic map do not visibly filter out S9+ before synthetic measurement generation. Therefore the manuscript statement that S9+ is fully held out must be audited carefully. We must distinguish (a) an environment survey map that is legitimately available before deployment from (b) held-out-device fingerprints that would violate a device-generalization claim.

Also check whether per-phone magnetic centering is computed using the whole database, including any nominally held-out phone, and whether this creates a device-information leak.

## Priority B — architecture consistency / explanation

### [ ] P4. Remove obsolete magnetic-anomaly notation from active Section II-E (observation 3)

The current Section II-E still mentions `A_obs`, `A(x)`, and `nabla A` in a sentence explaining that they are *not* used. This is technically a disclaimer rather than an active equation, but it is confusing now that the final architecture is CNN-output-only. Remove or rewrite the sentence so the active method contains no legacy anomaly notation at all.

### [ ] P5. Explain why `Delta z_wifi` is a GRU input (observation 6)

Current code computes `wifi_delta = z_wifi,t - previous_wifi` when a new Wi-Fi fix is available. Its intended role is to tell the GRU whether successive absolute Wi-Fi fixes are spatially consistent or make a sudden jump. This gives the gain network information about short-term Wi-Fi stability/outlier behaviour beyond the instantaneous innovation `z_wifi - x_pred`.

We should decide whether this feature is genuinely necessary, testable by ablation, and described clearly enough in the paper. It is zeroed when Wi-Fi is unavailable; `previous_wifi` holds the last available fix.

### [ ] P6. Explain the CNN variance output precisely (observation 7)

The magnetic CNN has a shared Conv1D encoder followed by two heads: a 2D position head and a one-scalar variance head. The variance head outputs **log variance**, not variance directly. Training uses heteroscedastic Gaussian NLL:

`0.5 * squared_position_error / exp(log_variance) + 0.5 * log_variance`.

This makes the network learn larger uncertainty for examples whose location is harder to predict, while the `+ log_variance` term prevents it from making uncertainty arbitrarily large. The current head is a **single isotropic scalar variance for the 2D position**, not separate x/y covariance entries. Our calibration experiment showed that its absolute scale is conservative, but its ranking is useful, which is why the final fusion uses relative variance weighting.

### [ ] P7. Deep code-backed walkthrough of the complete magnetic CNN architecture (observation 9)

Prepare a full explanation directly from `models/magnetic_sequence_cnn.py` and `train/train_magnetic_sequence.py`, including tensor shapes and receptive/temporal processing:

- input `[batch, time, 4]` using `magN`, `magV`, `magH`, `dip`;
- transpose to Conv1D layout `[batch, 4, time]`;
- Conv1D 4->32, kernel 7 + BatchNorm + ReLU + MaxPool/2;
- Conv1D 32->64, kernel 5 + BatchNorm + ReLU + MaxPool/2;
- Conv1D 64->128, kernel 3 + BatchNorm + ReLU;
- AdaptiveAvgPool1D(1) -> one 128-dimensional sequence representation;
- position head: 128->64->2 with ReLU and Dropout(0.2);
- variance head: 128->32->1 with ReLU;
- joint heteroscedastic NLL training.

Also inspect how the magnetic fingerprint map and synthetic causal 84-frame windows are generated, because the network architecture cannot be evaluated independently of that data-generation process.

## Priority C — paper presentation / proofreading

### [ ] P8. Fix `ell_ref` equation overflow (observation 4)

The line defining the training-reference log variance is visually crossing/pressing the IEEE column boundary in the current PDF. Reformat using `aligned`, split the median definition and `sigma_ref^2 = exp(ell_ref)` across lines, or otherwise guarantee both fit inside one column. Re-render the PDF after the fix.

### [ ] P9. Present the 13 GRU inputs as a readable list (observation 5)

The current long prose sentence is hard to audit. Replace it with a compact aligned/list presentation, ideally one feature per line with its dimensionality, followed by the explicit total:

`2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 = 13`.

This should make it much easier for a reviewer to verify the GRU input dimension against the implementation.

### [ ] P10. Remove/standardize the em dash on page 2 (observation 8)

Locate the page-2 final-line em dash and replace it with IEEE-consistent punctuation/wording if it is stylistically awkward. Perform this only during the final copy-edit pass so pagination changes do not make the page reference stale.

## Discussion order proposed

1. P1 + P2 + P3 first: these may change the claimed experimental methodology and therefore dominate everything else.
2. P4 + P5: make the active fusion description internally consistent and justify the GRU features.
3. P6 + P7: fully understand/document the CNN and uncertainty head.
4. P8 + P9 + P10: final formatting/copy-editing after methodological text has stabilized.
