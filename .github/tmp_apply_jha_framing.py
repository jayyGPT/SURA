from pathlib import Path

paper = Path('paper/main.tex')
text = paper.read_text(encoding='utf-8')

# Title: remove the unbenchmarked real-time claim and foreground the causal design.
text = text.replace(
    r'\title{Dual-Innovation Neural-Kalman Fusion for Real-Time Indoor Localization}',
    r'\title{Dual-Innovation Neural-Kalman Fusion for Causal Indoor Localization}',
    1,
)

# Abstract: narrow claims to the implemented architecture and evaluated protocol.
abstract_start = text.index(r'\begin{abstract}')
abstract_end = text.index(r'\end{abstract}', abstract_start) + len(r'\end{abstract}')
new_abstract = r'''\begin{abstract}
Indoor localization can combine Wi-Fi, magnetic-field, and inertial sensing, but practical fusion requires a causal mechanism for reconciling sparse spatial measurements with drifting pedestrian dead reckoning (PDR) when measurement availability and reliability vary over time. We propose a decoupled learned state-space estimator in which a Wi-Fi heatmap Multi-Layer Perceptron (MLP) and an 84-frame gravity-referenced magnetic Convolutional Neural Network (CNN) independently produce Cartesian position measurements, while PDR supplies relative motion. A DualKalmanNet forms separate Wi-Fi and magnetic innovations and uses a Gated Recurrent Unit (GRU) to predict independent $2\times2$ correction gains. Availability masks remove missing modalities, and a training-normalized magnetic log-uncertainty score scales the magnetic correction. Temporal fusion is evaluated on 60 independent map-constrained test trajectories within a fixed surveyed environment. With 1~Hz Wi-Fi, the uncertainty-weighted model obtains 0.494~m mean error (0.437~m median), compared with 0.473~m (0.449~m median) for Wi-Fi-only KalmanNet. Under degraded Wi-Fi (5~s updates with 40\% AP dropout), it reduces mean error from 1.533~m to 1.154~m, a 24.7\% reduction, while lowering the P90 error from 2.643~m to 1.612~m.
\end{abstract}'''
text = text[:abstract_start] + new_abstract + text[abstract_end:]

keywords_start = text.index(r'\begin{IEEEkeywords}')
keywords_end = text.index(r'\end{IEEEkeywords}', keywords_start) + len(r'\end{IEEEkeywords}')
new_keywords = r'''\begin{IEEEkeywords}
Indoor positioning, KalmanNet, learned state-space estimation, sensor fusion, pedestrian dead reckoning, Wi-Fi fingerprinting, magnetic localization.
\end{IEEEkeywords}'''
text = text[:keywords_start] + new_keywords + text[keywords_end:]

# Introduction: directly answer Prof. Jha's questions about prior three-sensor work,
# the mathematical target problem, and the actual architectural contribution.
intro_start = text.index(r'\section{Introduction}')
intro_end = text.index(r'\section{Proposed System Architecture}', intro_start)
new_intro = r'''\section{Introduction}

Indoor localization supports location-based services, smart buildings, and emergency navigation, where satellite-based positioning is often unreliable indoors~\cite{radar}. Wi-Fi fingerprinting provides infrastructure-based absolute position estimates~\cite{horus}, pedestrian dead reckoning (PDR) provides high-rate relative motion but accumulates drift~\cite{nnwifipdr}, and indoor magnetic structure provides an additional spatial fingerprint~\cite{overviewdatasets}. These modalities have already been combined in prior systems: Li \emph{et al.} integrate PDR, Wi-Fi fingerprinting, and magnetic matching~\cite{li2016hybrid}; Bolat and Akcakoca combine Wi-Fi, magnetic-field, and inertial navigation~\cite{hybridwifi}; and DeepPositioning learns joint Wi-Fi--magnetic localization~\cite{deeppos}. Thus, combining the three sensing sources is not itself our contribution. The problem addressed here is how to apply spatial corrections from asynchronous learned measurements to a drifting motion state when measurement availability and reliability change over time.

For a planar state, PDR produces a causal prediction $\mathbf{x}_{t}^{-}=\mathbf{x}_{t-1}+\mathbf{u}_{t}$. A spatial measurement $\mathbf{z}_{j,t}$ from modality $j\in\{wifi,mag\}$ then produces an innovation $\mathbf{y}_{j,t}=\mathbf{z}_{j,t}-\mathbf{x}_{t}^{-}$. Fusion therefore requires deciding how strongly each correction $\mathbf{K}_{j,t}\mathbf{y}_{j,t}$ should modify the state. Prior work addresses parts of this problem using adaptive factor-graph weighting for Wi-Fi/PDR~\cite{wang2020factorgraph}, neural Wi-Fi/PDR fusion~\cite{nnwifipdr}, and learned Kalman-style state estimation~\cite{kalmannet}. The distinction sought here is architectural rather than an initialization fix: two learned spatial measurements are represented in one Cartesian state space and are allowed to influence the causal state update through separate, history-dependent gain matrices.

Learned spatial models provide complementary measurement functions. DeepPositioning combines Wi-Fi and magnetic fingerprints using deep learning~\cite{deeppos}; MINLOC applies a CNN to magnetic-field patterns~\cite{minloc}; and Zardkoohi \emph{et al.} use a Bi-LSTM for magnetic-sequence localization~\cite{bilstmmag}. A bidirectional recurrent estimator uses both past and future sequence context and therefore is not directly usable for streaming causal inference without reformulation. Our objective is not to claim that these models fail because of poor initialization, but to separate spatial measurement extraction from temporal correction and to make the contribution of each available measurement explicit in the state update.

In this work, we propose a decoupled causal learned state-space architecture for multimodal indoor localization, where \emph{multimodal} denotes heterogeneous Wi-Fi, magnetic-field, and inertial measurements. First, a Wi-Fi heatmap MLP and gravity-referenced magnetic CNN independently produce Cartesian position measurements; the magnetic CNN also provides a scalar relative-uncertainty score. Second, DualKalmanNet forms separate Cartesian innovations and predicts independent $2\times2$ Wi-Fi and magnetic gain matrices, while availability masks remove unavailable modalities. Third, a training-normalized magnetic confidence weight explicitly attenuates uncertain magnetic corrections. The temporal fusion experiment evaluates independent map-constrained trajectories within one surveyed environment; device generalization is evaluated separately for the Wi-Fi and static-fingerprint experiments. The architecture can be reused after site-specific surveying and training, but the present fusion results do not constitute building-agnostic or plug-and-play unseen-device validation.

The remainder of this paper is structured as follows. Section \ref{sec:architecture} formalizes the proposed state-space architecture. Section \ref{sec:setup} details the experimental setup, dataset processing, and augmentation. Section \ref{sec:results} presents the performance evaluation, and Section \ref{sec:conclusion} concludes the paper.

'''
text = text[:intro_start] + new_intro + text[intro_end:]

old_arch = r'''Our proposed architecture functions as a modular, learned Bayesian filter designed for general hybrid indoor environments. It relies entirely on past and present observations, maintaining strict causality. The complete system architecture is illustrated in Fig.~\ref{fig:arch_diagram}.'''
new_arch = r'''Indoor localization is formulated as a causal learned state-space estimation problem. The estimator maintains a 2D Cartesian state, site-specific learned Wi-Fi and magnetic models provide spatial measurements, and DualKalmanNet performs recurrent prediction and correction using only past and present observations. The complete signal flow is illustrated in Fig.~\ref{fig:arch_diagram}.'''
if old_arch not in text:
    raise SystemExit('architecture opening not found')
text = text.replace(old_arch, new_arch, 1)

# Wi-Fi normalization: remove the redundant absent-AP branch noted by Prof. Jha.
old_norm = r'''The raw Wi-Fi scan $\mathbf{s}_t \in \mathbb{R}^N$ is preprocessed into a normalized input $\tilde{\mathbf{s}}_t \in [0,1]^N$. Absent APs, i.e., those that are not detected by the device, are assigned a constant floor value of $-100$~dBm. Each element is then clipped and linearly rescaled as
\begin{equation}
\tilde{s}_{t,i} = \begin{cases} 0 & \text{if } s_{t,i} \leq -100 \text{ (AP absent)} \\ \frac{\text{clip}(s_{t,i},\, -90,\, -30) + 90}{60} & \text{otherwise} \end{cases} \label{eq:wifi_norm}
\end{equation}'''
new_norm = r'''The raw Wi-Fi scan $\mathbf{s}_t \in \mathbb{R}^N$ is preprocessed into $\tilde{\mathbf{s}}_t \in [0,1]^N$. Undetected APs are stored at the $-100$~dBm floor; clipping and affine rescaling then give
\begin{equation}
\tilde{s}_{t,i}=\frac{\operatorname{clip}(s_{t,i},-90,-30)+90}{60},
\label{eq:wifi_norm}
\end{equation}
so an undetected AP maps to zero without a separate branch.'''
if old_norm not in text:
    raise SystemExit('Wi-Fi normalization block not found')
text = text.replace(old_norm, new_norm, 1)

# Make p-vs-Gaussian-target semantics explicit rather than describing p as a Gaussian posterior.
old_p = r'''To map this highly non-linear signal to spatial coordinates, we employ a Multi-Layer Perceptron (MLP) with Dropout regularization. The MLP processes $\tilde{\mathbf{s}}_t$ and outputs a probability vector $\mathbf{p} \in \mathbb{R}^M$ via a Softmax layer. Each element $p_c$ represents the posterior probability that the device occupies grid cell $c$ given the observed RSSI vector:
\begin{equation}
p_c = P(\mathbf{x}_t \in \text{cell } c \mid \tilde{\mathbf{s}}_t), \quad \sum_{c=1}^{M} p_c = 1 \label{eq:softmax_def}
\end{equation}'''
new_p = r'''To map this signal to the surveyed grid, the MLP processes $\tilde{\mathbf{s}}_t$ and outputs a Softmax heatmap $\mathbf{p}\in[0,1]^M$ satisfying
\begin{equation}
p_c\ge 0,\qquad \sum_{c=1}^{M}p_c=1.
\label{eq:softmax_def}
\end{equation}
Here $p_c$ is the learned discrete probability mass assigned to grid cell $c$.'''
if old_p not in text:
    raise SystemExit('Wi-Fi p definition not found')
text = text.replace(old_p, new_p, 1)

old_target_tail = r'''where $\sigma$ is the spatial spread parameter. A Gaussian target is preferable to a one-hot label because it assigns partial probability mass to cells in the vicinity of the true location, providing the network with a smooth, spatially-graded gradient signal rather than a binary right-or-wrong penalty. This significantly stabilizes training in regions of sparse survey coverage. The network is then optimized by minimizing the KL divergence between $\mathbf{q}$ and the network output $\mathbf{p}$:'''
new_target_tail = r'''where $\sigma$ is the spatial spread parameter. The Gaussian is used only to construct the discrete training target $\mathbf{q}$; the network output $\mathbf{p}$ remains an unconstrained learned Softmax distribution over cells. Relative to a one-hot target, $\mathbf{q}$ assigns nonzero mass to nearby cells and therefore provides spatially graded supervision. The network is optimized by minimizing the KL divergence between $\mathbf{q}$ and $\mathbf{p}$:'''
if old_target_tail not in text:
    raise SystemExit('Wi-Fi target explanation not found')
text = text.replace(old_target_tail, new_target_tail, 1)

old_heatmap_infer = r'''During real-time inference, the continuous coordinate estimate $\mathbf{z}_{wifi}$ and the associated measurement covariance matrix $\mathbf{R}_{hm}$ are computed as the probability-weighted expectation (soft-argmax):
\begin{equation}
\mathbf{z}_{wifi} = \sum_{c=1}^{M} p_c \cdot \mathbf{c}_c \label{eq:heatmap_centroid}
\end{equation}
\begin{equation}
\mathbf{R}_{hm} = \sum_{c=1}^{M} p_c\,(\mathbf{c}_c - \mathbf{z}_{wifi})(\mathbf{c}_c - \mathbf{z}_{wifi})^T \label{eq:heatmap_cov}
\end{equation}
where $\mathbf{c}_c \in \mathbb{R}^2$ denotes the physical coordinates of grid cell $c$. A concentrated heatmap yields a tightly bounded covariance, providing an honest, per-scan confidence estimate to the downstream filter.'''
new_heatmap_infer = r'''At inference, the Cartesian Wi-Fi measurement used by the final fusion network is the Softmax expectation
\begin{equation}
\mathbf{z}_{wifi}=\sum_{c=1}^{M}p_c\mathbf{c}_c,
\label{eq:heatmap_centroid}
\end{equation}
where $\mathbf{c}_c\in\mathbb{R}^2$ is the physical coordinate of grid cell $c$. The final DualKalmanNet consumes $\mathbf{z}_{wifi}$ directly; a heatmap covariance is not part of its 13-dimensional GRU input.'''
if old_heatmap_infer not in text:
    raise SystemExit('Wi-Fi inference/covariance block not found')
text = text.replace(old_heatmap_infer, new_heatmap_infer, 1)

# PDR motivation + explicit reference at the location Prof. Jha marked.
old_pdr_intro = r'''The Pedestrian Dead Reckoning (PDR) module converts a causal acceleration stream and a heading observation into relative planar displacement at the IMU rate ($f_s=16.7$~Hz). Let $a_t=\|\mathbf{a}_t\|$ denote acceleration magnitude. A first-order exponential moving average (EMA) tracks the slowly varying gravitational baseline,'''
new_pdr_intro = r'''Pedestrian Dead Reckoning (PDR) uses smartphone inertial measurements to propagate short-term relative motion between absolute position updates and is a standard component of hybrid indoor navigation~\cite{nnwifipdr,axesmapping}. In our estimator, PDR converts a causal acceleration stream and a heading observation into relative planar displacement at the IMU rate ($f_s=16.7$~Hz). Let $a_t=\|\mathbf{a}_t\|$ denote acceleration magnitude. We use a first-order exponential moving average (EMA) to track the slowly varying gravitational baseline,'''
if old_pdr_intro not in text:
    raise SystemExit('PDR intro not found')
text = text.replace(old_pdr_intro, new_pdr_intro, 1)

# Mathematically distinguish EKF analytical gain from KalmanNet learned gain, and define dual innovation.
kn_start = text.index(r'\subsection{Dual-Innovation KalmanNet Fusion}\label{sec:kalmannet}')
kn_marker = r'Both learned environment models are then expressed in the same Cartesian state space.'
kn_mid = text.index(kn_marker, kn_start)
new_kn_open = r'''\subsection{Dual-Innovation KalmanNet Fusion}\label{sec:kalmannet}

For an Extended Kalman Filter (EKF), after linearizing the measurement model with Jacobian $\mathbf{H}_t$, the analytical correction gain is
\begin{equation}
\mathbf{K}_t=\mathbf{P}_t^-\mathbf{H}_t^T
\left(\mathbf{H}_t\mathbf{P}_t^-\mathbf{H}_t^T+\mathbf{R}_t\right)^{-1},
\label{eq:ekf_gain}
\end{equation}
where $\mathbf{P}_t^-$ is the predicted state-error covariance and $\mathbf{R}_t$ is the assumed measurement-noise covariance. KalmanNet preserves a state-space prediction/correction structure but replaces this analytical gain computation with a recurrent learned mapping when parts of the dynamics or statistics are imperfectly known~\cite{kalmannet}. Our dual extension predicts separate gains for Wi-Fi and magnetic measurements. The causal PDR prediction is
\begin{equation}
\mathbf{x}_{\text{pred}}=\mathbf{x}_{t-1}+\mathbf{u}_t.
\label{eq:kn_predict}
\end{equation}
'''
text = text[:kn_start] + new_kn_open + text[kn_mid:]

old_mag_transition = r'''The magnetic CNN output $\mathbf{z}_{mag}$ therefore serves directly as the magnetic position measurement consumed by KalmanNet.'''
new_mag_transition = r'''The pair $(\mathbf{y}_{wifi},\mathbf{y}_{mag})$ constitutes the \emph{dual innovation}: two Cartesian measurement residuals formed against the same PDR prediction. The magnetic CNN output $\mathbf{z}_{mag}$ therefore enters the correction in the same state space as the Wi-Fi measurement.'''
if old_mag_transition not in text:
    raise SystemExit('magnetic innovation transition not found')
text = text.replace(old_mag_transition, new_mag_transition, 1)

# Remove unsupported qualitative language from training/result claims.
old_window = r'''The 1D-CNN magnetic sequence matcher utilized an empirically optimal temporal window of $T = 84$ frames (5.0~s at 16.7~Hz), selected via a sweep over candidate $\{50, 84, 134, 167\}$ frames, providing sufficient spatial variation for unambiguous matching without overfitting long topological routes.'''
new_window = r'''The 1D-CNN magnetic sequence matcher uses $T=84$ frames (5.0~s at 16.7~Hz), selected from the candidate set $\{50,84,134,167\}$ by the development sweep.'''
if old_window not in text:
    raise SystemExit('window-selection sentence not found')
text = text.replace(old_window, new_window, 1)

old_wifi_result = r'''The standalone performance of the Wi-Fi heatmap environment model prior to temporal fusion is shown in Table \ref{tab:wifi_heatmap}. The MLP achieves a robust MAE of 1.43~m on standard splits. Crucially, it maintains a high degree of accuracy of 2.02~m MAE on entirely unseen smartphone hardware. '''
new_wifi_result = r'''The standalone performance of the Wi-Fi heatmap environment model prior to temporal fusion is shown in Table \ref{tab:wifi_heatmap}. The mixed-device random split gives 1.43~m MAE, while the separate phone-split experiment gives 2.02~m MAE when Samsung Galaxy S9+ fingerprints are held out from model fitting. '''
if old_wifi_result not in text:
    raise SystemExit('Wi-Fi result prose not found')
text = text.replace(old_wifi_result, new_wifi_result, 1)

text = text.replace(
    r'''This paper presented a decoupled Neural-Kalman architecture that separates spatial inference from temporal tracking.''',
    r'''This paper presented a decoupled learned state-space architecture that separates spatial inference from temporal tracking.''',
    1,
)

# Global claim guards for this review pass.
for forbidden in (
    'trajectory memorization',
    'learned Bayesian filter',
    'generalized, decoupled',
    'optimally fuses',
    'empirically optimal temporal window',
    'entirely unseen smartphone hardware',
):
    if forbidden.lower() in text.lower():
        raise SystemExit(f'unsupported/obsolete wording remains: {forbidden}')

paper.write_text(text, encoding='utf-8')

# Add directly relevant literature identified during the citation audit.
bib = Path('paper/Ref.bib')
b = bib.read_text(encoding='utf-8')
if '@article{li2016hybrid,' not in b:
    b += r'''

@article{li2016hybrid,
  title={A Hybrid WiFi/Magnetic Matching/PDR Approach for Indoor Navigation With Smartphone Sensors},
  author={Li, You and Zhuang, Yuan and Lan, Haiyu and Zhou, Qifan and Niu, Xiaoji and El-Sheimy, Naser},
  journal={IEEE Communications Letters},
  volume={20},
  number={1},
  pages={169--172},
  year={2016},
  doi={10.1109/LCOMM.2015.2496940}
}
'''
if '@article{wang2020factorgraph,' not in b:
    b += r'''

@article{wang2020factorgraph,
  title={Deep Neural Network-Based Wi-Fi/Pedestrian Dead Reckoning Indoor Positioning System Using Adaptive Robust Factor Graph Model},
  author={Wang, Yifan and Li, Zengke and Gao, Jingxiang and Zhao, Long},
  journal={IET Radar, Sonar \& Navigation},
  volume={14},
  number={1},
  pages={36--47},
  year={2020},
  doi={10.1049/iet-rsn.2019.0260}
}
'''
bib.write_text(b, encoding='utf-8')

# Update the review tracker only for comments actually resolved by this pass.
review = Path('paper/reviews/prof_read_ieee_comments_draft.md')
r = review.read_text(encoding='utf-8')
for key in ('G1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J8', 'J11', 'J14', 'J16'):
    r = r.replace(f'### [ ] {key}.', f'### [x] {key}.', 1)

notes = {
    'G1': '**Resolved in Jha framing pass:** audited strong claims in the abstract, introduction, training/results prose, and conclusion; removed unsupported generalized/Bayesian/trajectory-memorization/optimality language, narrowed evaluation scope, and removed the unused Wi-Fi heatmap-covariance claim from the active fusion description.',
    'J2': '**Resolved:** the opening now explicitly acknowledges prior Wi-Fi+magnetic+PDR systems and states that sensor combination is not the contribution; it ends on the causal, reliability-varying state-correction problem targeted here.',
    'J3': '**Resolved:** replaced the old grouped `[6]-[8]` claim with paper-specific statements and citations, added direct prior three-sensor and adaptive Wi-Fi/PDR references, and placed a PDR citation at the location marked in the annotated PDF.',
    'J4': '**Resolved:** the limitation argument is now written through the PDR prediction, modality innovations, and correction matrices rather than trajectory-memorization/generalization jargon.',
    'J5': '**Resolved:** the introduction distinguishes direct spatial models, adaptive factor-graph/neural fusion, and the proposed separate-gain causal correction structure; it explicitly states that the contribution is not an initialization fix.',
    'J6': '**Resolved:** removed the learned-Bayesian-filter wording, uses learned state-space estimation terminology, defines multimodal sensing, and standardizes KalmanNet/DualKalmanNet usage.',
    'J8': '**Resolved:** the redundant absent-AP branch is removed; the paper now states the -100 dBm missing-AP convention followed by a single clipped affine normalization that maps missing APs to zero.',
    'J11': '**Resolved for module motivation/citation:** PDR now begins with its role between absolute updates and cites relevant smartphone PDR/fusion literature. Remaining global section restructuring is tracked separately under G3/J10.',
    'J14': '**Resolved:** added the analytical EKF gain equation with definitions, explained that KalmanNet replaces analytical gain generation by a recurrent learned mapping, and explicitly defined the pair of Cartesian residuals as the dual innovation.',
    'J16': '**Resolved:** removed building-agnostic/generalized framing and explicitly separates reusable architecture from site-specific survey/training and the scope of the current fixed-environment fusion evaluation.',
}

for key, note in notes.items():
    marker = f'### [x] {key}.'
    pos = r.find(marker)
    if pos < 0:
        raise SystemExit(f'review marker missing: {key}')
    next_heading = r.find('\n### ', pos + len(marker))
    if next_heading < 0:
        next_heading = len(r)
    block = r[pos:next_heading]
    if note not in block:
        r = r[:next_heading] + '\n\n' + note + '\n' + r[next_heading:]

review.write_text(r, encoding='utf-8')
