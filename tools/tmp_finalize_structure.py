from pathlib import Path

paper_path = Path('paper/main.tex')
text = paper_path.read_text(encoding='utf-8')

replacements = [
    (
        "The remainder of this paper is structured as follows. Section \\ref{sec:measurements} defines the sensor observations and preprocessing used to construct model inputs. Section \\ref{sec:architecture} presents the learned spatial measurement models, PDR motion model, and DualKalmanNet fusion rule. Section \\ref{sec:setup} describes training objectives and the evaluation protocol, Section \\ref{sec:results} presents the results, and Section \\ref{sec:conclusion} concludes the paper.",
        "Section \\ref{sec:measurements} defines measurements and preprocessing; Section \\ref{sec:architecture} presents the estimator; Section \\ref{sec:setup} gives training and evaluation; Section \\ref{sec:results} reports results; and Section \\ref{sec:conclusion} concludes the paper.",
    ),
    (
        "This section defines the observable signals and survey-derived preprocessing used to construct model inputs. Network architectures, state updates, and training objectives are deliberately separated into the following sections.\n\n\\subsection{Problem Setting and Sensor Observations}\n",
        "",
    ),
    ("\\subsection{Wi-Fi RSSI Preprocessing}\n\n", "\\paragraph{Wi-Fi preprocessing.}\n"),
    ("\\subsection{Magnetic Survey Preprocessing}\n\n", "\\paragraph{Magnetic preprocessing.}\n"),
    ("\\subsection{Inertial Inputs for PDR}\n", "\\paragraph{Inertial inputs.}\n"),
    (
        "\nThe resulting normalized vector $\\tilde{\\mathbf{s}}_t$ is the complete Wi-Fi input to the learned spatial measurement model in Section~\\ref{sec:wifi_model}.\n",
        "",
    ),
    (
        "\nThe four preprocessed channels are the inputs to the magnetic sequence measurement model in Section~\\ref{sec:mag_model}.\n",
        "",
    ),
    (
        "Given the preprocessed observations above, the estimator constructs two learned Cartesian spatial measurements and a causal PDR motion control, then combines them with DualKalmanNet. The complete signal flow is illustrated in Fig.~\\ref{fig:arch_diagram}.",
        "The estimator forms Wi-Fi and magnetic Cartesian measurements, propagates PDR motion, and combines them with DualKalmanNet (Fig.~\\ref{fig:arch_diagram}).",
    ),
    (
        "The magnetic measurement model consumes the four preprocessed channels over a causal window ending at the current time,",
        "The magnetic model consumes the four preprocessed channels over a causal window ending at time $t$:",
    ),
    (
        "so no future magnetic frame is included in $\\mathbf{M}_t$. The network architecture below maps this window to a Cartesian magnetic fix $\\mathbf{z}_{mag}$ and a scalar log-uncertainty score $\\ell_{mag}$.",
        "No future frame is included. The CNN maps $\\mathbf{M}_t$ to $\\mathbf{z}_{mag}$ and $\\ell_{mag}$.",
    ),
    (
        "The score $\\ell_{mag}$ is treated only as a learned relative uncertainty indicator; its training objective and calibration interpretation are described in Section~\\ref{sec:setup}.",
        "$\\ell_{mag}$ is used only as a relative uncertainty indicator; its training objective is given in Section~\\ref{sec:setup}.",
    ),
    (
        "Pedestrian Dead Reckoning (PDR) propagates short-term relative motion between absolute position updates and is a standard component of hybrid indoor navigation~\\cite{nnwifipdr,axesmapping}. Using the causal acceleration magnitude $a_t$ and heading observation $\\hat{\\theta}_t$ defined in Section~\\ref{sec:measurements}, the module detects steps and converts them into planar controls. We use a first-order exponential moving average (EMA) to track the slowly varying gravitational baseline,",
        "PDR propagates relative motion between absolute updates~\\cite{nnwifipdr,axesmapping}. From causal acceleration magnitude $a_t$ and heading $\\hat{\\theta}_t$, it detects steps and forms planar controls. An exponential moving average (EMA) tracks the gravitational baseline,",
    ),
    (
        "where $\\hat{\\theta}_t$ is the causal heading observation and $L_s$ is the assumed stride length. The evaluated estimator uses the fixed nominal value $L_s=0.65$~m and does not adapt it from ground-truth path length. The estimator treats $\\hat{\\theta}_t$ as an external causal input; the synthetic procedure used to generate that observation, including drift and white noise, is specified in Section~\\ref{sec:setup}.",
        "where $L_s=0.65$~m is fixed in the evaluated estimator and $\\hat{\\theta}_t$ is an external causal heading input; its synthetic noise model is specified in Section~\\ref{sec:setup}. No ground-truth path length is used to adapt $L_s$.",
    ),
    (
        "The Wi-Fi and magnetic models provide absolute Cartesian measurements, while PDR supplies relative motion. DualKalmanNet combines these signals through a causal prediction/correction update with separate learned gains for the two absolute modalities.\n\n",
        "",
    ),
    (
        "The learned Wi-Fi and magnetic measurement models are trained separately from the temporal fusion network. Their target construction and loss functions are defined here rather than in the measurement/model descriptions.\n\n",
        "The Wi-Fi and magnetic measurement models are trained separately from temporal fusion.\n\n",
    ),
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError(f'missing expected text for compacting: {old[:100]!r}')
    text = text.replace(old, new, 1)

# Do not allow the structural separation to regress.
method = text.split('\\section{Proposed State-Space Estimator}', 1)[1].split('\\section{Experimental Setup and Training}', 1)[0]
assert '\\mathcal{L}_{MLP}' not in method
assert '\\mathcal{L}_{CNN}' not in method
assert '\\section{Sensor Measurements and Preprocessing}\\label{sec:measurements}' in text
assert '\\paragraph{Wi-Fi preprocessing.}' in text
assert '\\paragraph{Magnetic preprocessing.}' in text
assert '\\paragraph{Inertial inputs.}' in text

paper_path.write_text(text, encoding='utf-8')
print('Compacted structural prose while preserving G3/J10 separation.')
