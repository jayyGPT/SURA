from pathlib import Path

PAPER = Path("paper/main.tex")
TRACKER = Path("paper/reviews/prof_read_ieee_comments_draft.md")


def split_once(text: str, marker: str) -> tuple[str, str]:
    if text.count(marker) != 1:
        raise RuntimeError(f"expected exactly one marker: {marker!r}; found {text.count(marker)}")
    return text.split(marker, 1)


paper = PAPER.read_text(encoding="utf-8")

old_remainder = (
    "The remainder of this paper is structured as follows. Section \\ref{sec:architecture} "
    "formalizes the proposed state-space architecture. Section \\ref{sec:setup} details the "
    "experimental setup, dataset processing, and augmentation. Section \\ref{sec:results} "
    "presents the performance evaluation, and Section \\ref{sec:conclusion} concludes the paper."
)
new_remainder = (
    "The remainder of this paper is structured as follows. Section \\ref{sec:measurements} "
    "defines the sensor observations and preprocessing used to construct model inputs. Section "
    "\\ref{sec:architecture} presents the learned spatial measurement models, PDR motion model, "
    "and DualKalmanNet fusion rule. Section \\ref{sec:setup} describes training objectives and the "
    "evaluation protocol, Section \\ref{sec:results} presents the results, and Section "
    "\\ref{sec:conclusion} concludes the paper."
)
if old_remainder not in paper:
    raise RuntimeError("introduction remainder paragraph did not match expected source")
paper = paper.replace(old_remainder, new_remainder, 1)

ARCH = r"\section{Proposed System Architecture}\label{sec:architecture}"
SETUP = r"\section{Experimental Setup and Dataset}\label{sec:setup}"
RESULTS = r"\section{Experimental Results}\label{sec:results}"
start = paper.index(ARCH)
end = paper.index(RESULTS)
old_middle = paper[start:end]
arch_text, setup_text = split_once(old_middle, SETUP)

GENERAL = r"\subsection{General Problem Formulation}"
WIFI = r"\subsection{Wi-Fi Processing and Heatmap Model}"
MAG = r"\subsection{Magnetic Survey Processing and Sequence Matcher}"
PDR = r"\subsection{Causal Pedestrian Dead Reckoning (PDR)}"
FUSION = r"\subsection{Dual-Innovation KalmanNet Fusion}\label{sec:kalmannet}"

pre_general, rest = split_once(arch_text, GENERAL)
general_body, rest = split_once(rest, WIFI)
wifi_body, rest = split_once(rest, MAG)
mag_body, rest = split_once(rest, PDR)
pdr_body, fusion_body = split_once(rest, FUSION)

fig_start = pre_general.index(r"\begin{figure}[htbp]")
fig_end = pre_general.index(r"\end{figure}", fig_start) + len(r"\end{figure}")
arch_figure = pre_general[fig_start:fig_end].strip()

# Wi-Fi: preprocessing stays with sensor definitions; model and inference move to the method;
# target construction and KL loss move to experimental training.
WIFI_MODEL_START = "To map this signal to the surveyed grid"
WIFI_TRAIN_START = "The training target"
WIFI_INFER_START = "At inference"
wifi_pre, wifi_after_pre = split_once(wifi_body, WIFI_MODEL_START)
wifi_before_train, wifi_after_train_marker = split_once(WIFI_MODEL_START + wifi_after_pre, WIFI_TRAIN_START)
wifi_train, wifi_infer = split_once(WIFI_TRAIN_START + wifi_after_train_marker, WIFI_INFER_START)
wifi_model = (wifi_before_train + WIFI_INFER_START + wifi_infer).strip()
wifi_train = wifi_train.strip()
wifi_pre = wifi_pre.strip()

# Magnetic: raw feature/map construction is preprocessing; CNN architecture is method;
# sequence synthesis and uncertainty-weighted regression are training/evaluation design.
MAG_TRAIN_START = "The CNN is trained on survey-derived"
MAG_FIG_START = r"\begin{figure*}[htbp]"
MAG_LOSS_START = r"Let $\ell_{mag}^{(i)}$"
MAG_SCOPE_START = "The current synthetic fusion evaluation samples"
mag_pre, mag_after_pre = split_once(mag_body, MAG_TRAIN_START)
mag_generation, mag_after_generation = split_once(MAG_TRAIN_START + mag_after_pre, MAG_FIG_START)
mag_arch, mag_after_arch = split_once(MAG_FIG_START + mag_after_generation, MAG_LOSS_START)
mag_loss, mag_scope = split_once(MAG_LOSS_START + mag_after_arch, MAG_SCOPE_START)
mag_pre = mag_pre.strip()
mag_arch = mag_arch.strip()
mag_loss = mag_loss.strip()
mag_scope = (MAG_SCOPE_START + mag_scope).strip()
mag_scope = mag_scope.replace(
    "The current synthetic fusion evaluation samples its magnetic inputs from the per-phone-centered "
    "survey map described above. It therefore evaluates temporal fusion within that surveyed magnetic "
    "domain; it does not establish that an uncalibrated unseen handset can be mapped into the same "
    "centered feature domain without an additional causal normalization procedure.",
    "The per-phone centering defines a survey-specific magnetic feature domain. The present fusion "
    "evaluation operates inside that domain and does not establish that an uncalibrated unseen handset "
    "can enter it without an additional causal normalization procedure.",
)

window_phrase = "A causal window ending at time $t$ is therefore"
if window_phrase not in mag_generation:
    raise RuntimeError("magnetic window definition not found")
window_pos = mag_generation.index(window_phrase)
eq_start = mag_generation.index(r"\begin{equation}", window_pos)
eq_end = mag_generation.index(r"\end{equation}", eq_start) + len(r"\end{equation}")
mag_window_eq = mag_generation[eq_start:eq_end].strip()
mag_training_generation = mag_generation[:window_pos].strip()

# PDR method: keep causal detector/control mathematics in the method, move simulator-specific
# heading generation to the evaluation section (where it is already documented in detail).
first_eq = pdr_body.index(r"\begin{equation}")
pdr_after_intro = pdr_body[first_eq:]
PDR_GEOM = r"where $\hat{\theta}_t$ denotes"
PDR_FIG = r"\begin{figure}[htbp]"
pdr_before_geom, pdr_geom_and_fig = split_once(pdr_after_intro, PDR_GEOM)
_, pdr_fig_rest = split_once(PDR_GEOM + pdr_geom_and_fig, PDR_FIG)
new_pdr_intro = (
    "Pedestrian Dead Reckoning (PDR) propagates short-term relative motion between absolute position "
    "updates and is a standard component of hybrid indoor navigation~\\cite{nnwifipdr,axesmapping}. "
    "Using the causal acceleration magnitude $a_t$ and heading observation $\\hat{\\theta}_t$ defined "
    "in Section~\\ref{sec:measurements}, the module detects steps and converts them into planar controls. "
    "We use a first-order exponential moving average (EMA) to track the slowly varying gravitational baseline,\n"
)
new_pdr_geom = (
    "where $\\hat{\\theta}_t$ is the causal heading observation and $L_s$ is the assumed stride length. "
    "The evaluated estimator uses the fixed nominal value $L_s=0.65$~m and does not adapt it from "
    "ground-truth path length. The estimator treats $\\hat{\\theta}_t$ as an external causal input; "
    "the synthetic procedure used to generate that observation, including drift and white noise, is "
    "specified in Section~\\ref{sec:setup}.\n\n"
)
pdr_method = (new_pdr_intro + pdr_before_geom + new_pdr_geom + PDR_FIG + pdr_fig_rest).strip()

# Experimental setup: retain the verified simulator protocol, but put training objectives before it.
TRAJ = r"\subsection{Survey-Derived Synthetic Trajectory Evaluation}"
TRAIN = r"\subsection{Training Details}"
setup_intro, setup_after_intro = split_once(setup_text, TRAJ)
traj_body, training_details = split_once(setup_after_intro, TRAIN)
traj_body = traj_body.replace("Section~II-D", r"Section~\ref{sec:pdr}")
setup_intro = setup_intro.strip()
traj_body = traj_body.strip()
training_details = training_details.strip()

problem_setting = r"""\subsection{Problem Setting and Sensor Observations}
Assume a targeted indoor environment with $N$ identifiable Wi-Fi Access Points (APs) and a Cartesian reference grid containing $M$ spatial cells. At time $t$, the phone-side observations relevant to this work are a Wi-Fi RSSI vector $\mathbf{s}_t\in\mathbb{R}^N$, a magnetometer vector $\mathbf{m}_t\in\mathbb{R}^3$, and an accelerometer vector $\mathbf{a}_t\in\mathbb{R}^3$. The PDR module also receives a causal heading observation $\hat{\theta}_t$. The objective is to estimate the planar Cartesian state $\mathbf{x}_t=[x_t,y_t]^T\in\mathbb{R}^2$. Missing Wi-Fi or magnetic measurements are represented explicitly by modality-availability masks during fusion."""

inertial_inputs = r"""\subsection{Inertial Inputs for PDR}
For PDR, the accelerometer stream is reduced to its magnitude $a_t=\|\mathbf{a}_t\|$. The heading observation $\hat{\theta}_t$ is treated as a causal external input to the estimator rather than as a quantity inferred from future trajectory information. The present synthetic evaluation generates this heading observation from the latent path plus a specified noise process in Section~\ref{sec:setup}; a deployable raw-sensor heading estimator is outside the scope of the current experiment."""

wifi_pre_section = (
    r"\subsection{Wi-Fi RSSI Preprocessing}" + "\n\n" + wifi_pre +
    "\n\nThe resulting normalized vector $\\tilde{\\mathbf{s}}_t$ is the complete Wi-Fi input to the learned spatial measurement model in Section~\\ref{sec:wifi_model}."
)

mag_pre_section = (
    r"\subsection{Magnetic Survey Preprocessing}" + "\n\n" + mag_pre + "\n\n" + mag_scope +
    "\n\nThe four preprocessed channels are the inputs to the magnetic sequence measurement model in Section~\\ref{sec:mag_model}."
)

wifi_model_section = (
    r"\subsection{Wi-Fi Heatmap Measurement Model}\label{sec:wifi_model}" + "\n\n" + wifi_model
)

mag_model_section = (
    r"\subsection{Magnetic Sequence Measurement Model}\label{sec:mag_model}" + "\n\n" +
    "The magnetic measurement model consumes the four preprocessed channels over a causal window ending at the current time,\n" +
    mag_window_eq +
    "\nso no future magnetic frame is included in $\\mathbf{M}_t$. The network architecture below maps this window to a Cartesian magnetic fix $\\mathbf{z}_{mag}$ and a scalar log-uncertainty score $\\ell_{mag}$.\n\n" +
    mag_arch +
    "\n\nThe score $\\ell_{mag}$ is treated only as a learned relative uncertainty indicator; its training objective and calibration interpretation are described in Section~\\ref{sec:setup}."
)

pdr_section = r"\subsection{Causal PDR Motion Model}\label{sec:pdr}" + "\n\n" + pdr_method

fusion_section = (
    r"\subsection{Dual-Innovation KalmanNet Fusion}\label{sec:kalmannet}" + "\n\n" +
    "The Wi-Fi and magnetic models provide absolute Cartesian measurements, while PDR supplies relative motion. DualKalmanNet combines these signals through a causal prediction/correction update with separate learned gains for the two absolute modalities.\n\n" +
    fusion_body.strip()
)

wifi_training_section = r"""\paragraph{Wi-Fi heatmap.}
""" + wifi_train

mag_training_section = (
    r"\paragraph{Magnetic sequence CNN.}" + "\n" + mag_training_generation +
    "\n\nEach generated 84-frame window uses the Cartesian position at its final frame as the regression target.\n\n" +
    mag_loss
)

new_middle = (
    r"\section{Sensor Measurements and Preprocessing}\label{sec:measurements}" + "\n\n" +
    "This section defines the observable signals and survey-derived preprocessing used to construct model inputs. Network architectures, state updates, and training objectives are deliberately separated into the following sections.\n\n" +
    problem_setting + "\n\n" +
    wifi_pre_section + "\n\n" +
    mag_pre_section + "\n\n" +
    inertial_inputs + "\n\n" +
    r"\section{Proposed State-Space Estimator}\label{sec:architecture}" + "\n\n" +
    "Given the preprocessed observations above, the estimator constructs two learned Cartesian spatial measurements and a causal PDR motion control, then combines them with DualKalmanNet. The complete signal flow is illustrated in Fig.~\\ref{fig:arch_diagram}.\n\n" +
    arch_figure + "\n\n" +
    wifi_model_section + "\n\n" +
    mag_model_section + "\n\n" +
    pdr_section + "\n\n" +
    fusion_section + "\n\n" +
    r"\section{Experimental Setup and Training}\label{sec:setup}" + "\n\n" +
    setup_intro + "\n\n" +
    r"\subsection{Measurement-Model Training Objectives}" + "\n\n" +
    "The learned Wi-Fi and magnetic measurement models are trained separately from the temporal fusion network. Their target construction and loss functions are defined here rather than in the measurement/model descriptions.\n\n" +
    wifi_training_section + "\n\n" +
    mag_training_section + "\n\n" +
    TRAJ + "\n\n" +
    traj_body + "\n\n" +
    TRAIN + "\n\n" +
    training_details + "\n\n"
)

paper = paper[:start] + new_middle + paper[end:]

# Structural invariants: losses must no longer live in the estimator section, and the simulator
# should use the PDR label rather than a hard-coded section number.
method_slice = paper[paper.index(r"\section{Proposed State-Space Estimator}"):paper.index(r"\section{Experimental Setup and Training}")]
if r"\mathcal{L}_{MLP}" in method_slice or r"\mathcal{L}_{CNN}" in method_slice:
    raise RuntimeError("training losses remain inside the proposed-method section")
if "Section~II-D" in paper:
    raise RuntimeError("hard-coded PDR section reference remains after restructuring")
for required in [r"\label{sec:measurements}", r"\label{sec:wifi_model}", r"\label{sec:mag_model}", r"\label{sec:pdr}"]:
    if required not in paper:
        raise RuntimeError(f"missing required label {required}")

PAPER.write_text(paper, encoding="utf-8")

tracker = TRACKER.read_text(encoding="utf-8")
old_g3 = """### [ ] G3. Separate measurements/preprocessing from methodology and loss-function design

Restructure the paper so that the reader first understands **what the phone/environment provides and how it is preprocessed**, then **what estimator/model consumes those measurements**, and finally **how each model is trained and what loss is used**. Do not mix sensor definitions, network architecture, target construction, and training loss in one subsection.
"""
new_g3 = """### [x] G3. Separate measurements/preprocessing from methodology and loss-function design

Restructure the paper so that the reader first understands **what the phone/environment provides and how it is preprocessed**, then **what estimator/model consumes those measurements**, and finally **how each model is trained and what loss is used**. Do not mix sensor definitions, network architecture, target construction, and training loss in one subsection.

**Resolved in measurement/method separation pass:** the manuscript now has separate top-level sections for sensor measurements and preprocessing, the proposed state-space estimator, and experimental setup/training. Wi-Fi and magnetic target/loss definitions were moved out of the model descriptions into a dedicated training-objectives subsection; simulator-specific heading generation remains in the evaluation protocol rather than the PDR method definition.
"""
if old_g3 not in tracker:
    raise RuntimeError("G3 tracker block did not match expected source")
tracker = tracker.replace(old_g3, new_g3, 1)

old_j10 = """### [ ] J10. Separate signal definitions/preprocessing, proposed functions/method, and training/loss design

This is one of the most repeated comments in the PDF:
- \"separate method into a new section\";
- \"separate training set into a new section\";
- \"Loss function is also part of separate sub section\";
- briefly introduce signals and how they are obtained/preprocessed;
- similarly introduce phone signals and magnetic preprocessing;
- move model functions into the proposed approach section;
- move loss-function design into training.

**Status:** strongly applicable and directly matches G3. This should drive a structural rewrite rather than local edits.
"""
new_j10 = """### [x] J10. Separate signal definitions/preprocessing, proposed functions/method, and training/loss design

This is one of the most repeated comments in the PDF:
- \"separate method into a new section\";
- \"separate training set into a new section\";
- \"Loss function is also part of separate sub section\";
- briefly introduce signals and how they are obtained/preprocessed;
- similarly introduce phone signals and magnetic preprocessing;
- move model functions into the proposed approach section;
- move loss-function design into training.

**Resolved:** Section II now defines sensor observations and preprocessing only; Section III contains the Wi-Fi/magnetic measurement functions, causal PDR model, and DualKalmanNet state update; Section IV contains model-training targets/losses plus the synthetic evaluation protocol and optimizer settings. The previous Wi-Fi and magnetic subsections no longer mix preprocessing, architecture, target construction, and loss design.
"""
if old_j10 not in tracker:
    raise RuntimeError("J10 tracker block did not match expected source")
tracker = tracker.replace(old_j10, new_j10, 1)
TRACKER.write_text(tracker, encoding="utf-8")

print("Restructured paper and marked G3/J10 resolved.")
