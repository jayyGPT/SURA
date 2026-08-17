# Professor Read / IEEE Comments Draft

Working checklist collected during final paper proofreading. These are **questions and potential issues to investigate together before changing the manuscript**. A confident-sounding statement here should not be treated as a final methodological conclusion until we resolve it against the active code and experiment protocol.

## Priority A - methodology / validity

### [x] P1. PDR heading source, `Orn_z`, and heading-offset calibration (observations 1 + 10)

**Concern.** The manuscript currently describes device yaw `Orn_z` corrected by a fixed heading offset `phi_h`, calibrated from ground-truth training trajectories. We need to verify whether this is actually what the final experiment does, whether `Orn_z` is sufficiently available in MagWi, and whether any offset calibration ever touches test information.

**Preliminary code finding.** The active final fusion benchmark does **not** use dataset `Orn_z` or `fit_heading_offset()`. `synthesize_walk()` derives a heading from the synthetic ground-truth path tangent and adds random-walk drift plus white noise; `build_sequence()` then uses this synthetic heading directly for PDR. The manuscript therefore appears inconsistent with the active benchmark. `models/pdr.py` still contains a generic `fit_heading_offset()` helper, but that helper is not used by the current fusion benchmark.

**Dataset note.** The repository dataset audit reports substantial missingness in `Orn_z`, so the paper's current wording about `Orn_z` must be revisited even if we later design a real-data PDR evaluation.

**Decision needed.** Decide whether the paper should describe the actual synthetic-heading protocol, or whether we should implement and validate a truly causal heading-estimation/calibration method for real inference.

### [x] P2. Step-length calibration and online realism (observation 2)

**Concern.** The paper says step length `L_s` is calibrated from training trajectories and frozen. We need to decide what is scientifically acceptable and what a deployable system should do.

**Preliminary code finding.** The active fusion benchmark does **not** estimate step length from the training walks; it uses a fixed constant `STEP_LENGTH_M = 0.65`. `models/pdr.py` contains a ground-truth-based `calibrate_step_length()` helper, but the current benchmark does not call it.

**Important distinction.** Estimating a hyperparameter from the **training set only** is normally legal and is not test leakage. However, ground-truth-based calibration is not automatically deployable to an unseen user. A rolling/online step-length estimate would be more practical, but it cannot use ground-truth distance at inference; it would need inertial features and/or trusted absolute Wi-Fi/magnetic corrections.

**Decision needed.** Align paper and code, then decide whether to keep a fixed nominal step length as a controlled simulation parameter or implement a causal online estimator.

### [x] P3. Full data-augmentation / synthetic-evaluation audit (observation 11)

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


### P1-P3 implementation note

Resolved in the methodology-consistency pass:

- **P1:** the evaluated fusion method no longer claims to consume MagWi `Orn_z` or a ground-truth-calibrated `phi_h`. The paper now defines PDR using a generic causal heading observation and documents the actual synthetic heading measurement used in the benchmark.
- **P2:** the paper no longer claims that stride length is estimated from the fusion training trajectories. It documents the actual fixed nominal `L_s = 0.65 m`; no rolling estimator is claimed or implemented.
- **P3:** the simulator/estimator boundary is now explicit, fusion results are labelled as held-out-trajectory rather than held-out-device evaluation, and the code saves fixed train/test seeds plus an exact binned-trajectory overlap check. The paper also explicitly states that survey-derived Wi-Fi/magnetic environment resources use the available processed survey fingerprints, so those fusion numbers are not presented as unseen-device generalization.

## Priority B - architecture consistency / explanation

### [x] P4. Remove obsolete magnetic-anomaly notation from active Section II-E (observation 3)

The current Section II-E still mentions `A_obs`, `A(x)`, and `nabla A` in a sentence explaining that they are *not* used. This is technically a disclaimer rather than an active equation, but it is confusing now that the final architecture is CNN-output-only. Remove or rewrite the sentence so the active method contains no legacy anomaly notation at all.

### [x] P5. Explain why `Delta z_wifi` is a GRU input (observation 6)

Current code computes `wifi_delta = z_wifi,t - previous_wifi` when a new Wi-Fi fix is available. Its intended role is to tell the GRU whether successive absolute Wi-Fi fixes are spatially consistent or make a sudden jump. This gives the gain network information about short-term Wi-Fi stability/outlier behaviour beyond the instantaneous innovation `z_wifi - x_pred`.

We should decide whether this feature is genuinely necessary, testable by ablation, and described clearly enough in the paper. It is zeroed when Wi-Fi is unavailable; `previous_wifi` holds the last available fix.

### P4-P5 implementation note

- **P4:** removed the legacy anomaly notation from the active KalmanNet subsection. The paper now states only that the magnetic CNN position output is consumed directly by the fusion network.
- **P5:** retained `Delta z_wifi` after a paired full-protocol ablation (`benchmarks/wifi_delta_ablation/`). For the CNN DualKalmanNet, removing the two-scalar feature worsened mean error by `+0.0280 m` in full Wi-Fi (paired 95% CI `[+0.0016, +0.0545] m`) and by `+0.0918 m` in degraded Wi-Fi (CI `[-0.0224, +0.2060] m`). Wi-Fi-only differences were inconclusive. The manuscript now defines the feature using the most recent available Wi-Fi fix and explains it only as a short-term consistency cue, without claiming that every large Wi-Fi delta is an outlier.

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

## Priority C - paper presentation / proofreading

### [ ] P8. Fix `ell_ref` equation overflow (observation 4)

The line defining the training-reference log variance is visually crossing/pressing the IEEE column boundary in the current PDF. Reformat using `aligned`, split the median definition and `sigma_ref^2 = exp(ell_ref)` across lines, or otherwise guarantee both fit inside one column. Re-render the PDF after the fix.

### [ ] P9. Present the 13 GRU inputs as a readable list (observation 5)

The current long prose sentence is hard to audit. Replace it with a compact aligned/list presentation, ideally one feature per line with its dimensionality, followed by the explicit total:

`2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 = 13`.

This should make it much easier for a reviewer to verify the GRU input dimension against the implementation.

### [ ] P10. Remove/standardize the em dash on page 2 (observation 8)

Locate the page-2 final-line em dash and replace it with IEEE-consistent punctuation/wording if it is stylistically awkward. Perform this only during the final copy-edit pass so pagination changes do not make the page reference stale.

---

# New Faculty Review - Prof. Nilesh Jha + General Supervisor Comments

Source: annotated `Hybrid-WiFi-Magnetometer-V3-comments.pdf` and the two follow-up emails. The PDF comments were made on an **older V3 draft** that still contained the scalar magnetic-anomaly architecture and old fusion metrics. We therefore preserve every comment below but distinguish comments that remain applicable from comments whose exact target has already been superseded.

## G. General comments from Prof. Neel Kanth Kundu

### [ ] G1. Make all statements and contribution claims more accurate, with appropriate citations

Audit the abstract, introduction, related-work discussion, dataset claims, generalization claims, causality claims, robustness claims, and numerical contribution statements. Every strong statement should either be supported by our experiment/code or by a specific citation. Avoid absolute words such as "entirely", "universally", "inherently", "rigorously", "optimal", or "robust" unless justified.

### [ ] G2. Improve mathematical definitions of variables

Perform a notation audit across the whole paper. Define each signal/function/state before first use; distinguish scalars, vectors, matrices, functions, sets, random variables, and constants consistently; include time indices where relevant; avoid symbols whose domain/codomain is unclear; reuse equation numbers instead of redefining quantities informally.

### [ ] G3. Separate measurements/preprocessing from methodology and loss-function design

Restructure the paper so that the reader first understands **what the phone/environment provides and how it is preprocessed**, then **what estimator/model consumes those measurements**, and finally **how each model is trained and what loss is used**. Do not mix sensor definitions, network architecture, target construction, and training loss in one subsection.

### [ ] G4. Improve flow between subsections/modules and perform a grammar pass

Add short motivation/transition sentences between Wi-Fi, magnetic, PDR, and KalmanNet modules; prefer top-down explanations; remove abrupt jumps; fix capitalization, punctuation, abbreviations, article usage, hyphenation, and awkward phrasing.

## J. Prof. Nilesh Jha - annotated V3 comments

### [ ] J1. Capitalization, abbreviations, and standard English usage throughout - especially the abstract

Use normal English capitalization: only proper nouns/acronyms should be capitalized. Standardize terms such as Wi-Fi, magnetic field, inertial measurement unit (IMU), KalmanNet/neural-Kalman wording, etc. Apply the correction to the abstract as well as body text.

**Status:** still applicable as a global copy-edit.

### [ ] J2. Rebuild the opening problem statement and verify what cited prior work actually uses

Prof. Jha asks whether the cited prior works already use the same three sensors. The first paragraph should end with the **precise technical problem targeted by this paper**, not merely list modalities. Depending on the literature, the gap should be framed as something like sparse/noisy Wi-Fi requiring additional measurements, or existing multi-sensor systems lacking an architecture that can exploit them practically/causally.

Actions:
- verify exactly which cited works use Wi-Fi, magnetic, and/or IMU;
- do not imply novelty of combining sensors if prior work already does so;
- end the opening with our actual unresolved problem;
- make the contribution distinction architectural/mathematical if that is the real novelty.

**Status:** strongly applicable.

### [ ] J3. Citation audit for the related-work paragraph and individual claims

Prof. Jha explicitly questioned the citation range `[6]-[8]`, individual references `[6]?`, `[8]?`, and marked at least one uncited claim with "cite". Verify each citation supports the sentence it is attached to. Avoid grouped citations when different papers support different claims.

**Status:** strongly applicable and overlaps G1.

### [ ] J4. Replace jargon-heavy limitation claims with a mathematical/causal argument

The V3 introduction described issues such as memorization/generalization in broad language. Prof. Jha asks us to start with the mathematical mechanism and then state its consequence. Example direction from his note: explain how temporal evolution can be sensitive to initial measurements/model structure, how lack of a particular correction produces accumulated error, and only then state the generalization consequence.

Do not write generic phrases like "pervasive limitations" or "critical limitations" without a precise mechanism.

**Status:** strongly applicable; current introduction still needs a contribution/limitation rewrite.

### [ ] J5. Expand the critique of prior architectures beyond a single initialization issue

His concern: if the only problem is bad initialization, then properly initializing previous methods could undermine our claimed contribution. We need to establish whether previous approaches fail because of architecture, sensor treatment, causality, missing-data handling, uncertainty handling, or some other concrete reason.

Actions:
- add/strengthen a related-work paragraph;
- cite a few directly relevant sensor-fusion methods;
- formulate the distinction mathematically where possible;
- be explicit about what our method changes versus simply tuning an existing model.

**Status:** strongly applicable.

### [ ] J6. Clarify terminology: "Bayesian", "state-space", "hybrid indoor", and "neural-Kalman"

Prof. Jha questioned the phrase "Bayesian filter" and suggested the structure may more accurately be described as a state-space model. He also asked us to define/correct the meaning of "hybrid indoor" and suggested wording such as "In this work, we propose a ...".

Actions:
- decide whether Bayesian terminology is mathematically justified;
- otherwise use state-space / learned state-space estimator wording;
- define "hybrid" as multi-modal/sensor-fusion if retained;
- standardize KalmanNet/neural-Kalman naming and hyphenation.

**Status:** strongly applicable.

### [ ] J7. Redesign/recheck Fig. 1 as a signal-flow / functional-block diagram

Specific requested improvements from the annotation:
- `u_t`, `z_wifi,t`, `z_mag,t` are inputs and should use a consistent visual convention/color;
- a signal should enter a functional block and emerge as a new signal, with signal labels above arrows where useful;
- replace vague "Predict" wording with the actual prediction function/block;
- define what "innovations" are;
- show where `x_{t-1}` enters;
- show the final `x_t` as an output arrow outside the block, analogous to Fig. 2.

**Status:** still applicable even though the figure has since changed; redraw/re-audit rather than assuming current Fig. 1 is final.

### [ ] J8. Simplify the Wi-Fi normalization equation

The V3 piecewise equation explicitly maps absent `-100 dBm` to zero, while the second branch's clipping/rescaling already maps `-100` to zero. Prof. Jha calls this redundant.

Action: either simplify the equation to one clipped affine mapping with a clear missing-AP convention, or explain why the explicit absent-AP branch is semantically necessary.

**Status:** still applicable to current Wi-Fi preprocessing text/equation.

### [ ] J9. General notation/spacing/capitalization cleanup in the problem formulation

Annotations include "notation and space is missing", "capitalization", wording such as "The objective in indoor localization", and a request to use enumeration rather than nested subsections where the material is conceptually one block.

Action: perform a sentence-by-sentence notation/grammar pass once the section structure is stable.

**Status:** applicable; exact page positions may have moved.

### [ ] J10. Separate signal definitions/preprocessing, proposed functions/method, and training/loss design

This is one of the most repeated comments in the PDF:
- "separate method into a new section";
- "separate training set into a new section";
- "Loss function is also part of separate sub section";
- briefly introduce signals and how they are obtained/preprocessed;
- similarly introduce phone signals and magnetic preprocessing;
- move model functions into the proposed approach section;
- move loss-function design into training.

**Status:** strongly applicable and directly matches G3. This should drive a structural rewrite rather than local edits.

### [ ] J11. Give PDR a short module-level motivation and explain it top-down

Before equations, add 1-2 lines explaining why PDR is included, what sensor measurements it uses, and whether it is standard or novel; cite an appropriate PDR reference. Then define the output/control variable first and decompose its ingredients, instead of narrating implementation steps one after another.

**Status:** strongly applicable, especially given P1/P2 methodology mismatch.

### [ ] J12. Do not oversell standard signal-processing operations such as the low-pass filter

Prof. Jha questioned wording around the EMA/low-pass filter because low-pass filtering is standard signal processing. Phrase it as a chosen preprocessing/detection component, not as if the filter itself were a methodological novelty.

**Status:** applicable.

### [ ] J13. Define a step-detection indicator and say explicitly what condition (11) detects

Instead of "Upon detection" without a mathematical object, define an indicator/event such as `d_t in {0,1}` for a detected step and state that condition (11) defines `d_t = 1`. Define any remaining variables at first use.

**Status:** applicable if the PDR equations remain after the methodology audit.

### [ ] J14. Introduce KalmanNet and dual innovation mathematically, and distinguish it from EKF

Prof. Jha asks for:
- the missing/expanded abbreviation where needed;
- a mathematical distinction between EKF and KalmanNet, preferably showing what analytical gain is replaced by the GRU-learned gain;
- a motivation that is logically consistent with the equations;
- an explicit definition/introduction of "dual innovation" in Kalman-filter language.

If the state-update equations are otherwise identical, do not claim the equations themselves solve variability; explain that the gain-generation mechanism is learned/context-dependent.

**Status:** strongly applicable.

### [ ] J15. Legacy anomaly-map mathematics - preserve as a historical warning, do not reintroduce it

On V3 page 4 Prof. Jha asked to:
- define the anomaly map mathematically and say how it is obtained;
- relate it explicitly to sensor measurements introduced earlier;
- define domain/codomain cleanly;
- avoid using an unbold capital letter ambiguously as a function/scalar;
- combine equations with clear references ("combining (..)-(..), we update ...").

**Current status:** the active architecture has removed `A_obs`, `A(x)`, and `nabla A`; therefore these exact comments are **superseded**. Their lasting lesson is that every new function/measurement we keep (`z_mag`, `ell_mag`, `w_mag`, heatmap covariance, etc.) must be defined with equal mathematical precision.

### [ ] J16. Audit broad generalization/building-agnostic claims

Prof. Jha questioned whether the statement that the framework is "entirely building-agnostic" / deployable to any building is accurate and suggested softening to a more defensible robustness statement.

Action: distinguish **architecture portability** from **trained-model portability**. The mathematics may be reusable across buildings, while Wi-Fi/magnetic environment models generally require a new survey/training/calibration for a new site.

**Status:** strongly applicable and overlaps G1.

### [ ] J17. Improve equation-to-equation prose flow

Use explicit transitions such as "Combining (15)-(17), the posterior update is..." and refer back to equation numbers rather than reintroducing undefined symbols in prose. This applies across Wi-Fi, magnetic, PDR, and KalmanNet sections.

**Status:** applicable and overlaps G2/G4.

### [ ] J18. Final grammar/punctuation micro-edits from the annotations

Preserve the small marked fixes for the final copy-edit pass: article usage (`the`), semicolon/punctuation corrections, spacing, capitalization, and awkward sentence fragments. Do these after the structural rewrite so page/line locations do not become stale.

**Status:** applicable as final pass.

---

## Consolidated order for the next session

1. **Validity first:** P1, P2, P3 - reconcile the experiment with the paper and audit leakage/generalization claims.
2. **Scientific framing:** G1, J2-J6, J16 - establish exactly what problem/contribution is defensible and verify the literature/citations.
3. **Paper architecture:** G3, J10 - separate measurements/preprocessing, methodology, and training/losses.
4. **Mathematical definitions:** G2, J11-J14, J17 plus P4-P7 - make every retained module/code path mathematically auditable.
5. **Figures/presentation:** J7, P8, P9, J8-J9.
6. **Copy-edit:** G4, J1, J12-J13 where relevant, J18, P10.

The goal is to finish the next revision as a **methodologically consistent new draft**, not to patch the old V3 sentence-by-sentence.