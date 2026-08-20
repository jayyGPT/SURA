from pathlib import Path

PAPER = Path('paper/main.tex')
TRACKER = Path('paper/reviews/prof_read_ieee_comments_draft.md')

text = PAPER.read_text(encoding='utf-8')
tracker = TRACKER.read_text(encoding='utf-8')


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one match, found {count}')
    return source.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Introduction: use one prior-state notation and roman modality subscripts.
# ---------------------------------------------------------------------------
text = replace_once(
    text,
    r'''For a planar state, PDR produces a causal prediction $\mathbf{x}_{t}^{-}=\mathbf{x}_{t-1}+\mathbf{u}_{t}$. A spatial measurement $\mathbf{z}_{j,t}$ from modality $j\in\{wifi,mag\}$ then produces an innovation $\mathbf{y}_{j,t}=\mathbf{z}_{j,t}-\mathbf{x}_{t}^{-}$. Fusion therefore requires deciding how strongly each correction $\mathbf{K}_{j,t}\mathbf{y}_{j,t}$ should modify the state. Prior work addresses parts of this problem using adaptive factor-graph weighting for Wi-Fi/PDR~\cite{wang2020factorgraph}, neural Wi-Fi/PDR fusion~\cite{nnwifipdr}, and learned Kalman-style state estimation~\cite{kalmannet}. The distinction sought here is architectural rather than an initialization fix: two learned spatial measurements are represented in one Cartesian state space and are allowed to influence the causal state update through separate, history-dependent gain matrices.''',
    r'''For fusion step $t\geq 1$, PDR produces the prior state $\mathbf{x}_{t}^{-}=\mathbf{x}_{t-1}+\mathbf{u}_{t}$. A spatial measurement $\mathbf{z}_{j,t}$ from modality $j\in\{\mathrm{wifi},\mathrm{mag}\}$ produces the innovation $\mathbf{y}_{j,t}=\mathbf{z}_{j,t}-\mathbf{x}_{t}^{-}$. Fusion therefore requires deciding how strongly each correction $\mathbf{K}_{j,t}\mathbf{y}_{j,t}$ should modify the prior. Prior work addresses parts of this problem using adaptive factor-graph weighting for Wi-Fi/PDR~\cite{wang2020factorgraph}, neural Wi-Fi/PDR fusion~\cite{nnwifipdr}, and learned Kalman-style state estimation~\cite{kalmannet}. The distinction sought here is architectural rather than an initialization fix: two learned spatial measurements are represented in one Cartesian state space and influence the causal state update through separate, history-dependent gain matrices.''',
    'introduction prior/innovation notation',
)

# ---------------------------------------------------------------------------
# Measurements: separate raw-sample n from fusion-step t and define masks.
# ---------------------------------------------------------------------------
text = replace_once(
    text,
    r'''Assume a targeted indoor environment with $N$ identifiable Wi-Fi Access Points (APs) and a Cartesian reference grid containing $M$ spatial cells. At time $t$, the phone-side observations relevant to this work are a Wi-Fi RSSI vector $\mathbf{s}_t\in\mathbb{R}^N$, a magnetometer vector $\mathbf{m}_t\in\mathbb{R}^3$, and an accelerometer vector $\mathbf{a}_t\in\mathbb{R}^3$. The PDR module also receives a causal heading observation $\hat{\theta}_t$. The objective is to estimate the planar Cartesian state $\mathbf{x}_t=[x_t,y_t]^T\in\mathbb{R}^2$. Missing Wi-Fi or magnetic measurements are represented explicitly by modality-availability masks during fusion.''',
    r'''Assume a targeted indoor environment with $N$ identifiable Wi-Fi Access Points (APs) and a Cartesian reference grid containing $M$ spatial cells. We use $n$ for raw inertial/magnetic samples and $t$ for the lower-rate fusion steps. The phone-side quantities are a sparse Wi-Fi RSSI vector $\mathbf{s}_t\in\mathbb{R}^N$, a magnetometer vector $\mathbf{m}_n\in\mathbb{R}^3$, an accelerometer vector $\mathbf{a}_n\in\mathbb{R}^3$, and a causal heading observation $\hat{\theta}_n$. The estimator state is $\mathbf{x}_t=[x_t,y_t]^T\in\mathbb{R}^2$. Availability indicators $m_{\mathrm{wifi},t},m_{\mathrm{mag},t}\in\{0,1\}$ specify whether a new Wi-Fi or magnetic measurement is available at fusion step $t$.''',
    'measurement indices and masks',
)

text = replace_once(
    text,
    r'''The magnetic path begins with gravity-referenced survey preprocessing. For each raw static sample, let $\mathbf{m}_t\in\mathbb{R}^3$ be the magnetometer vector and $\mathbf{a}_t\in\mathbb{R}^3$ the accelerometer vector. The implementation uses the instantaneous normalized acceleration
\begin{equation}
\hat{\mathbf{a}}_t=\frac{\mathbf{a}_t}{\|\mathbf{a}_t\|}
\label{eq:mag_gravity_proxy}
\end{equation}
as a gravity-direction proxy for the static survey record, and computes
\begin{align}
m_N &= \|\mathbf{m}_t\|, &
m_V &= \mathbf{m}_t^T\hat{\mathbf{a}}_t, \nonumber\\
m_H &= \sqrt{\max(m_N^2-m_V^2,0)}, &
\delta &= \operatorname{atan2}(m_V,m_H).
\label{eq:mag_features}
\end{align}
These quantities are invariant to a common rigid rotation of the phone when $\hat{\mathbf{a}}_t$ is a valid gravity proxy. The processed survey database stores the mean and standard deviation of each quantity over a static visit. The magnetic-map trainer then subtracts each phone's mean feature value, averages the centered visit means at each surveyed node, and interpolates the four channels $(m_N,m_V,m_H,\delta)$ onto the same 1~m Cartesian environment grid used by the Wi-Fi model.''',
    r'''The magnetic path begins with gravity-referenced survey preprocessing. For raw sample $n$, the implementation uses the instantaneous normalized acceleration
\begin{equation}
\hat{\mathbf{a}}_n=\frac{\mathbf{a}_n}{\|\mathbf{a}_n\|}
\label{eq:mag_gravity_proxy}
\end{equation}
as a gravity-direction proxy and computes
\begin{align}
m_{N,n} &= \|\mathbf{m}_n\|, &
m_{V,n} &= \mathbf{m}_n^T\hat{\mathbf{a}}_n, \nonumber\\
m_{H,n} &= \sqrt{\max(m_{N,n}^2-m_{V,n}^2,0)}, &
\delta_n &= \operatorname{atan2}(m_{V,n},m_{H,n}).
\label{eq:mag_features}
\end{align}
We collect these channels in
\begin{equation}
\mathbf{f}^{\mathrm{mag}}_n=
\begin{bmatrix}m_{N,n}&m_{V,n}&m_{H,n}&\delta_n\end{bmatrix}^{T}
\in\mathbb{R}^{4}.
\label{eq:mag_feature_vector}
\end{equation}
These quantities are invariant to a common rigid rotation of the phone when $\hat{\mathbf{a}}_n$ is a valid gravity proxy. The processed survey database stores the mean and standard deviation of each quantity over a static visit. The magnetic-map trainer subtracts each phone's mean feature value, averages the centered visit means at each surveyed node, and interpolates the four channels onto the same 1~m Cartesian environment grid used by the Wi-Fi model.''',
    'magnetic sample-indexed preprocessing',
)

text = replace_once(
    text,
    r'''For PDR, the accelerometer stream is reduced to its magnitude $a_t=\|\mathbf{a}_t\|$. The heading observation $\hat{\theta}_t$ is treated as a causal external input to the estimator rather than as a quantity inferred from future trajectory information. The present synthetic evaluation generates this heading observation from the latent path plus a specified noise process in Section~\ref{sec:setup}; a deployable raw-sensor heading estimator is outside the scope of the current experiment.''',
    r'''For PDR, the accelerometer stream is reduced to $a_n=\|\mathbf{a}_n\|$. The heading observation $\hat{\theta}_n$ is treated as a causal external input rather than as a quantity inferred from future trajectory information. The synthetic evaluation generates this observation from the latent path plus the noise process in Section~\ref{sec:setup}; a deployable raw-sensor heading estimator is outside the scope of the current experiment.''',
    'inertial raw-sample notation',
)

# ---------------------------------------------------------------------------
# Wi-Fi and magnetic measurement functions.
# ---------------------------------------------------------------------------
text = replace_once(
    text,
    r'''To map this signal to the surveyed grid, the MLP processes $\tilde{\mathbf{s}}_t$ and outputs a Softmax heatmap $\mathbf{p}\in[0,1]^M$ satisfying
\begin{equation}
p_c\ge 0,\qquad \sum_{c=1}^{M}p_c=1.
\label{eq:softmax_def}
\end{equation}
Here $p_c$ is the learned discrete probability mass assigned to grid cell $c$.''',
    r'''The MLP maps the normalized scan $\tilde{\mathbf{s}}_t$ to logits over the $M$ grid cells and then applies Softmax, producing $\mathbf{p}_t\in[0,1]^M$ with
\begin{equation}
p_{t,c}\ge 0,\qquad \sum_{c=1}^{M}p_{t,c}=1.
\label{eq:softmax_def}
\end{equation}
Here $p_{t,c}$ is the learned probability mass assigned to grid cell $c$ at fusion step $t$.''',
    'wifi heatmap time indexing',
)

text = replace_once(
    text,
    r'''At inference, the Cartesian Wi-Fi measurement used by the final fusion network is the Softmax expectation
\begin{equation}
\mathbf{z}_{wifi}=\sum_{c=1}^{M}p_c\mathbf{c}_c,
\label{eq:heatmap_centroid}
\end{equation}
where $\mathbf{c}_c\in\mathbb{R}^2$ is the physical coordinate of grid cell $c$. The final DualKalmanNet consumes $\mathbf{z}_{wifi}$ directly; a heatmap covariance is not part of its 13-dimensional GRU input.''',
    r'''At inference, the Cartesian Wi-Fi measurement is the Softmax expectation
\begin{equation}
\mathbf{z}_{\mathrm{wifi},t}=\sum_{c=1}^{M}p_{t,c}\mathbf{c}_c,
\label{eq:heatmap_centroid}
\end{equation}
where $\mathbf{c}_c\in\mathbb{R}^2$ is the physical coordinate of grid cell $c$. DualKalmanNet consumes $\mathbf{z}_{\mathrm{wifi},t}$ directly; a heatmap covariance is not part of its 13-dimensional GRU input.''',
    'wifi Cartesian measurement notation',
)

text = replace_once(
    text,
    r'''The magnetic model consumes the four preprocessed channels over a causal window ending at time $t$:
\begin{equation}
\mathbf{M}_t\in\mathbb{R}^{T\times4},\qquad T=84,
\label{eq:mag_window}
\end{equation}
No future frame is included. The CNN maps $\mathbf{M}_t$ to $\mathbf{z}_{mag}$ and $\ell_{mag}$.''',
    r'''Let $n_t$ denote the raw-sample index at the end of fusion step $t$. The magnetic input is the causal feature window
\begin{equation}
\mathbf{M}_t=
\begin{bmatrix}
(\mathbf{f}^{\mathrm{mag}}_{n_t-T+1})^T\\
\vdots\\
(\mathbf{f}^{\mathrm{mag}}_{n_t})^T
\end{bmatrix}
\in\mathbb{R}^{T\times4},\qquad T=84.
\label{eq:mag_window}
\end{equation}
No future raw sample is included. The CNN maps $\mathbf{M}_t$ to the Cartesian measurement $\mathbf{z}_{\mathrm{mag},t}\in\mathbb{R}^2$ and scalar log-uncertainty score $\ell_{\mathrm{mag},t}\in\mathbb{R}$.''',
    'magnetic window definition',
)

text = replace_once(
    text,
    r'''$\ell_{mag}$ is used only as a relative uncertainty indicator; its training objective is given in Section~\ref{sec:setup}.''',
    r'''$\ell_{\mathrm{mag},t}$ is used only as a relative uncertainty indicator; its training objective is given in Section~\ref{sec:setup}.''',
    'magnetic uncertainty notation',
)

# Keep Fig. 1 notation synchronized until its later dedicated redesign.
text = text.replace(r'($\mathbf{z}_{wifi}$)', r'($\mathbf{z}_{\mathrm{wifi},t}$)')
text = text.replace(r'($\mathbf{z}_{mag},\,\ell_{mag}$)', r'($\mathbf{z}_{\mathrm{mag},t},\,\ell_{\mathrm{mag},t}$)')
text = text.replace(r'$\mathbf{x}_{\text{pred}}$', r'$\mathbf{x}_{t}^{-}$')
text = text.replace(r'$\mathbf{y}_{wifi}, \mathbf{y}_{mag}$', r'$\mathbf{y}_{\mathrm{wifi},t}, \mathbf{y}_{\mathrm{mag},t}$')
text = text.replace(r'$\mathbf{K}_{wifi}, \mathbf{K}_{mag}$', r'$\mathbf{K}_{\mathrm{wifi},t}, \mathbf{K}_{\mathrm{mag},t}$')

# ---------------------------------------------------------------------------
# PDR: raw-sample indicator, per-step displacement, and fusion-bin control.
# ---------------------------------------------------------------------------
pdr_start = text.index(r'\subsection{Causal PDR Motion Model}\label{sec:pdr}')
pdr_figure = text.index(r'\begin{figure}[htbp]', pdr_start)
new_pdr = r'''\subsection{Causal PDR Motion Model}\label{sec:pdr}

PDR propagates relative motion between absolute updates~\cite{nnwifipdr,axesmapping}. At raw sampling rate $f_s=16.7$~Hz, an exponential moving average provides the baseline used by the causal threshold detector,
\begin{equation}
\bar{a}_n = \alpha\bar{a}_{n-1} + (1-\alpha)a_n,
\qquad \alpha=0.98,
\label{eq:lpf}
\end{equation}
with $\bar{a}_0=9.81$~m/s$^2$, and
\begin{equation}
\tilde{a}_n=a_n-\bar{a}_n.
\label{eq:hpf}
\end{equation}
Let $n_{\mathrm{last}}$ be the raw index of the most recently detected step. The binary step indicator is
\begin{equation}
d_n=\mathbb{1}\!\left[
\tilde{a}_n>\tau
\;\land\;
(n-n_{\mathrm{last}})>\Delta_r
\right],
\label{eq:step_trigger}
\end{equation}
where $\tau=0.6$~m/s$^2$ and $\Delta_r=\lfloor0.3f_s\rfloor=5$ frames; when $d_n=1$, the detector sets $n_{\mathrm{last}}\leftarrow n$. This detector uses only samples observed up to $n$.

The displacement contributed by raw sample $n$ is
\begin{equation}
\mathbf{v}_n=d_n L_s
\begin{bmatrix}
\cos\hat{\theta}_n\\
\sin\hat{\theta}_n
\end{bmatrix},
\qquad L_s=0.65~\mathrm{m}.
\label{eq:pdr_step}
\end{equation}
Let $\mathcal{B}_t$ denote the raw-sample indices assigned to fusion bin $t$. The PDR control consumed by DualKalmanNet is the accumulated bin displacement
\begin{equation}
\mathbf{u}_t=\sum_{n\in\mathcal{B}_t}\mathbf{v}_n
\in\mathbb{R}^2.
\label{eq:pdr}
\end{equation}
The evaluated estimator keeps $L_s$ fixed and treats $\hat{\theta}_n$ as an external causal heading observation; its synthetic noise model is specified in Section~\ref{sec:setup}. No ground-truth path length is used to adapt $L_s$.

'''
text = text[:pdr_start] + new_pdr + text[pdr_figure:]

# Update the single-step geometry figure to use v_n rather than the bin-summed u_t.
text = text.replace(r'{foot at $t\!-\!1$}', r'{foot before step $n$}')
text = text.replace(r'{$\mathbf{u}_t$, $\|\mathbf{u}_t\|=L_s$}', r'{$\mathbf{v}_n$, $\|\mathbf{v}_n\|=L_s$}')
text = text.replace(r'{foot at $t$}', r'{foot after step $n$}')
text = text.replace(r'{$\hat{\theta}_t$}', r'{$\hat{\theta}_n$}')
text = text.replace(r'{$L_s\cos\hat{\theta}_t$}', r'{$L_s\cos\hat{\theta}_n$}')
text = text.replace(r'{$L_s\sin\hat{\theta}_t$}', r'{$L_s\sin\hat{\theta}_n$}')
text = replace_once(
    text,
    r'''\caption{Geometry of the PDR control used by the estimator. A detected step of nominal length $L_s$ is projected along the currently available heading observation $\hat{\theta}_t$. In the synthetic fusion evaluation, $\hat{\theta}_t$ is a noisy simulated heading measurement rather than a ground-truth-calibrated \texttt{Orn\_z} signal.}''',
    r'''\caption{Geometry of one detected PDR step. The per-sample displacement $\mathbf{v}_n$ has nominal length $L_s$ and is projected along the causal heading observation $\hat{\theta}_n$; fusion control $\mathbf{u}_t$ sums these step displacements over the raw samples in bin $t$.}''',
    'PDR figure caption',
)

# ---------------------------------------------------------------------------
# DualKalmanNet: consistent prior/innovation notation and explicit GRU mapping.
# ---------------------------------------------------------------------------
text = replace_once(
    text,
    r'''Our dual extension predicts separate gains for Wi-Fi and magnetic measurements. The causal PDR prediction is
\begin{equation}
\mathbf{x}_{\text{pred}}=\mathbf{x}_{t-1}+\mathbf{u}_t.
\label{eq:kn_predict}
\end{equation}
Both learned environment models are then expressed in the same Cartesian state space. The Wi-Fi innovation is
\begin{equation}
\mathbf{y}_{wifi} = \mathbf{z}_{wifi} - \mathbf{x}_{\text{pred}} \in \mathbb{R}^2, \label{eq:kn_innovate_w}
\end{equation}
and, crucially, the magnetic innovation directly uses the 2D output of the magnetic CNN:
\begin{equation}
\mathbf{y}_{mag} = \mathbf{z}_{mag} - \mathbf{x}_{\text{pred}} \in \mathbb{R}^2. \label{eq:kn_innovate_m}
\end{equation}
The pair $(\mathbf{y}_{wifi},\mathbf{y}_{mag})$ constitutes the \emph{dual innovation}: two Cartesian measurement residuals formed against the same PDR prediction. The magnetic CNN output $\mathbf{z}_{mag}$ therefore enters the correction in the same state space as the Wi-Fi measurement.''',
    r'''Our dual extension predicts separate gains for Wi-Fi and magnetic measurements. Using the bin-level PDR control from~(\ref{eq:pdr}), the prior state is
\begin{equation}
\mathbf{x}_{t}^{-}=\mathbf{x}_{t-1}+\mathbf{u}_t.
\label{eq:kn_predict}
\end{equation}
The two learned spatial measurements are expressed in the same Cartesian state space and produce
\begin{align}
\mathbf{y}_{\mathrm{wifi},t}
&=\mathbf{z}_{\mathrm{wifi},t}-\mathbf{x}_{t}^{-}
\in\mathbb{R}^{2},
\label{eq:kn_innovate_w}\\
\mathbf{y}_{\mathrm{mag},t}
&=\mathbf{z}_{\mathrm{mag},t}-\mathbf{x}_{t}^{-}
\in\mathbb{R}^{2}.
\label{eq:kn_innovate_m}
\end{align}
The pair $(\mathbf{y}_{\mathrm{wifi},t},\mathbf{y}_{\mathrm{mag},t})$ is the \emph{dual innovation}: two Cartesian measurement residuals formed against the same prior state.''',
    'Kalman prior and innovations',
)

text = replace_once(
    text,
    r'''The CNN additionally predicts the scalar log-uncertainty score $\ell_{mag,t}$ and its positive scale $q_{mag,t}=\exp(\ell_{mag,t})$. Because only relative reliability is used in fusion, we normalize the score using a training-only reference
\begin{equation}
\begin{aligned}
\ell_{ref} &= \operatorname{median}_{(w,t)\in\mathcal{D}_{train}:\,m_{mag}=1}
\ell_{mag,w,t},\\
q_{ref} &= \exp(\ell_{ref}).
\end{aligned}
\label{eq:mag_ref}
\end{equation}
The relative magnetic confidence weight is then
\begin{equation}
\begin{aligned}
w_{mag,t}
&= \frac{1}{1+\exp(\ell_{mag,t}-\ell_{ref})} \\
&= \frac{1}{1+q_{mag,t}/q_{ref}},
\qquad 0<w_{mag,t}<1 .
\end{aligned}
\label{eq:mag_weight}
\end{equation}
A magnetic prediction with an uncertainty score above the training reference is therefore down-weighted, whereas a relatively confident fix retains a larger correction. The log-uncertainty score itself is also retained as a GRU input so that the network may learn additional context-dependent reliability beyond this explicit weighting.''',
    r'''The CNN additionally predicts $\ell_{\mathrm{mag},t}$ and the positive scale $q_{\mathrm{mag},t}=\exp(\ell_{\mathrm{mag},t})$. Because fusion uses only relative reliability, the reference score is computed from the fusion-training trajectories:
\begin{equation}
\begin{aligned}
\ell_{\mathrm{ref}}
&=\operatorname{median}_{(w,t)\in\mathcal{D}_{\mathrm{train}}:\,m_{\mathrm{mag},w,t}=1}
\ell_{\mathrm{mag},w,t},\\
q_{\mathrm{ref}}&=\exp(\ell_{\mathrm{ref}}),
\end{aligned}
\label{eq:mag_ref}
\end{equation}
where $w$ indexes a fusion-training walk. The relative magnetic confidence is
\begin{equation}
\begin{aligned}
w_{\mathrm{mag},t}
&=\frac{1}{1+\exp(\ell_{\mathrm{mag},t}-\ell_{\mathrm{ref}})}\\
&=\frac{1}{1+q_{\mathrm{mag},t}/q_{\mathrm{ref}}},
\qquad 0<w_{\mathrm{mag},t}<1.
\end{aligned}
\label{eq:mag_weight}
\end{equation}
Thus scores above the training reference receive smaller explicit magnetic weights. The same log-uncertainty score also enters the recurrent gain network as a clipped, masked feature.''',
    'magnetic reference and weight notation',
)

text = replace_once(
    text,
    r'''For readability, the 13-dimensional GRU input is listed explicitly below. Define the previous posterior displacement and the masked, numerically clipped magnetic-confidence feature as
\begin{equation}
\begin{aligned}
\Delta\mathbf{x}_{t-1} &= \mathbf{x}_{t-1}-\mathbf{x}_{t-2},\\
c_{mag,t} &= m_{mag}\,\operatorname{clip}(\ell_{mag,t},-6,8).
\end{aligned}
\label{eq:gru_aux}
\end{equation}''',
    r'''For readability, the 13-dimensional recurrent input is listed explicitly below. Define
\begin{equation}
\begin{aligned}
\Delta\mathbf{x}_{t-1}&=\mathbf{x}_{t-1}-\mathbf{x}_{t-2},\\
c_{\mathrm{mag},t}&=m_{\mathrm{mag},t}\,
\operatorname{clip}(\ell_{\mathrm{mag},t},-6,8),
\end{aligned}
\label{eq:gru_aux}
\end{equation}
with $\Delta\mathbf{x}_{0}=\mathbf{0}$ at the first fusion update.''',
    'GRU auxiliary features',
)

# Table labels.
text = text.replace(r'Wi-Fi innovation $\mathbf{y}_{wifi}$', r'Masked Wi-Fi innovation $m_{\mathrm{wifi},t}\mathbf{y}_{\mathrm{wifi},t}$')
text = text.replace(r'Magnetic innovation $\mathbf{y}_{mag}$', r'Masked magnetic innovation $m_{\mathrm{mag},t}\mathbf{y}_{\mathrm{mag},t}$')
text = text.replace(r'Wi-Fi fix difference $\Delta\mathbf{z}_{wifi}$', r'Wi-Fi fix difference $\Delta\mathbf{z}_{\mathrm{wifi},t}$')
text = text.replace(r'Wi-Fi availability $m_{wifi}$', r'Wi-Fi availability $m_{\mathrm{wifi},t}$')
text = text.replace(r'Magnetic availability $m_{mag}$', r'Magnetic availability $m_{\mathrm{mag},t}$')
text = text.replace(r'Magnetic confidence $c_{mag,t}$', r'Magnetic confidence $c_{\mathrm{mag},t}$')

wifi_delta_old = r'''Here,
\begin{equation}
\Delta\mathbf{z}_{wifi,t}=\mathbf{z}_{wifi,t}-\mathbf{z}_{wifi,t^-},
\label{eq:wifi_delta}
\end{equation}
where $t^-$ is the most recent time with an available Wi-Fi fix; the feature is zero when no new Wi-Fi measurement is present. It provides a short-term Wi-Fi consistency cue that the GRU interprets jointly with the current innovation, PDR displacement, and recurrent history. The GRU outputs eight values, reshaped into independent $2\!\times\!2$ matrices $\mathbf{K}_{wifi}$ and $\mathbf{K}_{mag}$. The posterior update is
\begin{equation}
\mathbf{x}_t = \mathbf{x}_{\text{pred}}
+ m_{wifi}\,\mathbf{K}_{wifi}\mathbf{y}_{wifi}
+ m_{mag}\,w_{mag,t}\,\mathbf{K}_{mag}\mathbf{y}_{mag}. \label{eq:kn_correct}
\end{equation}
The availability masks preserve operation under missing modalities: setting $m_{wifi}=0$ or $m_{mag}=0$ removes the corresponding correction. When a magnetic window is available but its predicted uncertainty score is large, $w_{mag,t}$ suppresses that correction before it can dominate the state estimate. This combines the learned, context-dependent gains of KalmanNet with an explicit relative-confidence prior from the magnetic CNN.'''

wifi_delta_new = r'''Let
\begin{equation}
\kappa_t^{\mathrm{wifi}}
=\max\{k<t:\,m_{\mathrm{wifi},k}=1\}
\label{eq:wifi_previous_index}
\end{equation}
denote the previous fusion step with a Wi-Fi update, when such a step exists. The temporal Wi-Fi feature is
\begin{equation}
\Delta\mathbf{z}_{\mathrm{wifi},t}=
\begin{cases}
m_{\mathrm{wifi},t}
\left(\mathbf{z}_{\mathrm{wifi},t}-
\mathbf{z}_{\mathrm{wifi},\kappa_t^{\mathrm{wifi}}}\right),
& \kappa_t^{\mathrm{wifi}}\ \text{exists},\\
\mathbf{0}, & \text{otherwise}.
\end{cases}
\label{eq:wifi_delta}
\end{equation}
This is a short-term Wi-Fi consistency cue; it is zero when no new Wi-Fi fix is present. The eight feature groups in the table, concatenated in the displayed order, define $\boldsymbol{\phi}_t\in\mathbb{R}^{13}$. The recurrent gain generator implemented in code is
\begin{align}
\mathbf{h}_t
&=\operatorname{GRUCell}(\boldsymbol{\phi}_t,\mathbf{h}_{t-1})
\in\mathbb{R}^{64},
\label{eq:gru_recurrence}\\
\mathbf{g}_t
&=\mathbf{W}_K\mathbf{h}_t+\mathbf{b}_K
\in\mathbb{R}^{8},
\label{eq:gain_head}
\end{align}
with $\mathbf{h}_0=\mathbf{0}$. The first four components of $\mathbf{g}_t$ are reshaped into $\mathbf{K}_{\mathrm{wifi},t}\in\mathbb{R}^{2\times2}$ and the final four into $\mathbf{K}_{\mathrm{mag},t}\in\mathbb{R}^{2\times2}$.

Combining the prior in~(\ref{eq:kn_predict}), innovations in~(\ref{eq:kn_innovate_w})--(\ref{eq:kn_innovate_m}), learned gains, availability masks, and magnetic confidence in~(\ref{eq:mag_weight}) gives the posterior update
\begin{equation}
\mathbf{x}_t=\mathbf{x}_t^{-}
+m_{\mathrm{wifi},t}\mathbf{K}_{\mathrm{wifi},t}\mathbf{y}_{\mathrm{wifi},t}
+m_{\mathrm{mag},t}w_{\mathrm{mag},t}\mathbf{K}_{\mathrm{mag},t}\mathbf{y}_{\mathrm{mag},t}.
\label{eq:kn_correct}
\end{equation}
Setting an availability mask to zero removes that modality's correction. A large magnetic uncertainty score additionally reduces the magnetic term through $w_{\mathrm{mag},t}$.'''
text = replace_once(text, wifi_delta_old, wifi_delta_new, 'Wi-Fi delta, GRU mapping, and posterior')

# ---------------------------------------------------------------------------
# Training objectives: use explicit training-example indices.
# ---------------------------------------------------------------------------
text = replace_once(
    text,
    r'''The training target $\mathbf{q} \in \mathbb{R}^M$ is a discrete probability distribution constructed by evaluating a 2D isotropic Gaussian, centered at the true position $\mathbf{x}_{true}$, at each of the $M$ grid cell coordinates $\mathbf{c}_c$, and normalizing the result to sum to unity:
\begin{equation}
q_c = \frac{\exp\!\left(-\dfrac{\|\mathbf{c}_c - \mathbf{x}_{true}\|^2}{2\sigma^2}\right)}{\displaystyle\sum_{j=1}^{M} \exp\!\left(-\dfrac{\|\mathbf{c}_j - \mathbf{x}_{true}\|^2}{2\sigma^2}\right)}, \quad \sum_{c=1}^{M} q_c = 1 \label{eq:target_q}
\end{equation}
where $\sigma$ is the spatial spread parameter. The Gaussian is used only to construct the discrete training target $\mathbf{q}$; the network output $\mathbf{p}$ remains an unconstrained learned Softmax distribution over cells. Relative to a one-hot target, $\mathbf{q}$ assigns nonzero mass to nearby cells and therefore provides spatially graded supervision. The network is optimized by minimizing the KL divergence between $\mathbf{q}$ and $\mathbf{p}$:
\begin{equation}
\mathcal{L}_{MLP} = D_{KL}(\mathbf{q} \parallel \mathbf{p}) = \sum_{c=1}^{M} q_c \log \left( \frac{q_c}{p_c} \right) \label{eq:kl_loss}
\end{equation}''',
    r'''For training example $i$, let $\mathbf{x}^{(i)}\in\mathbb{R}^2$ be the surveyed coordinate and $\mathbf{p}^{(i)}$ the MLP Softmax output. The spatially graded target $\mathbf{q}^{(i)}\in\mathbb{R}^M$ is
\begin{equation}
q_c^{(i)}=
\frac{\exp\!\left(-\dfrac{\|\mathbf{c}_c-\mathbf{x}^{(i)}\|^2}{2\sigma^2}\right)}
{\displaystyle\sum_{r=1}^{M}\exp\!\left(-\dfrac{\|\mathbf{c}_r-\mathbf{x}^{(i)}\|^2}{2\sigma^2}\right)},
\qquad \sum_{c=1}^{M}q_c^{(i)}=1,
\label{eq:target_q}
\end{equation}
where $\sigma$ is the target spread. For a minibatch of size $B$, the implemented KL objective is
\begin{equation}
\mathcal{L}_{\mathrm{MLP}}
=\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{M}
q_c^{(i)}\log\!\left(\frac{q_c^{(i)}}{p_c^{(i)}}\right).
\label{eq:kl_loss}
\end{equation}''',
    'Wi-Fi training target and KL notation',
)

text = text.replace(r'$\ell_{mag}^{(i)}$', r'$\ell_{\mathrm{mag}}^{(i)}$')
text = text.replace(r'$q_{mag}^{(i)}=\exp(\ell_{mag}^{(i)})$', r'$q_{\mathrm{mag}}^{(i)}=\exp(\ell_{\mathrm{mag}}^{(i)})$')
text = text.replace(r'\mathbf{z}_{mag}^{(i)}', r'\mathbf{z}_{\mathrm{mag}}^{(i)}')
text = text.replace(r'\mathbf{z}_{true}^{(i)}', r'\mathbf{x}_{\mathrm{true}}^{(i)}')
text = text.replace(r'\tilde q_{mag}^{(i)}', r'\tilde q_{\mathrm{mag}}^{(i)}')
text = text.replace(r'\ell_{mag}^{(i)}', r'\ell_{\mathrm{mag}}^{(i)}')
text = text.replace(r'q_{mag}^{(i)}', r'q_{\mathrm{mag}}^{(i)}')
text = text.replace(r'$q_{mag}$', r'$q_{\mathrm{mag}}$')

# ---------------------------------------------------------------------------
# Evaluation protocol: raw-sample heading notation and explicit initialization.
# ---------------------------------------------------------------------------
text = replace_once(
    text,
    r'''    \item \textbf{Trajectory generation:} Random endpoint pairs are sampled on the connected corridor graph and joined by shortest paths. Each path is interpolated at $16.7$~Hz with walking speed sampled uniformly from $1.0$ to $1.35$~m/s. Fusion-training and fusion-test walks are generated independently with fixed seeds 1 and 2, respectively. The implementation hashes the resulting binned target trajectories and aborts if an identical trajectory occurs in both sets.
    \item \textbf{PDR measurements:} The simulator computes the geometric path tangent $\theta^{true}_t$ and forms a noisy heading observation
    \begin{equation}
    \hat{\theta}_t=\theta^{true}_t+b_t+\epsilon_t,
    \label{eq:sim_heading}
    \end{equation}
    where $b_t$ is a Gaussian random walk with per-frame standard deviation $0.5^{\circ}/\sqrt{16.7}$ and $\epsilon_t$ is zero-mean white noise with standard deviation $8.8^{\circ}$. Acceleration magnitude is synthesized as a step-frequency sinusoid (frequency sampled from 1.7--2.0~Hz) around gravity with additive Gaussian noise, then passed through the causal step detector of Section~\ref{sec:pdr}. Every detected step uses the fixed nominal $L_s=0.65$~m.''',
    r'''    \item \textbf{Trajectory generation:} Random endpoint pairs are sampled on the connected corridor graph and joined by shortest paths. Each path is interpolated at $16.7$~Hz with walking speed sampled uniformly from $1.0$ to $1.35$~m/s. Fusion-training and fusion-test walks are generated independently with fixed seeds 1 and 2, respectively. The implementation hashes the resulting binned target trajectories and aborts if an identical trajectory occurs in both sets.
    \item \textbf{Estimator initialization:} For each generated walk, the starting coordinate $\mathbf{x}_{\mathrm{start}}$ is subtracted from the Wi-Fi fixes, magnetic fixes, and target trajectory before KalmanNet training/evaluation. The recurrent estimator therefore starts from $\mathbf{x}_0=\mathbf{0}$ in translated coordinates, which is equivalent to assuming that the initial 2D position is known. No subsequent ground-truth state is supplied as an estimator input.
    \item \textbf{PDR measurements:} At raw sample $n$, the simulator computes the geometric path tangent $\theta^{\mathrm{true}}_n$ and forms
    \begin{equation}
    \hat{\theta}_n=\theta^{\mathrm{true}}_n+b_n+\epsilon_n,
    \label{eq:sim_heading}
    \end{equation}
    where $b_n$ is a Gaussian random walk with per-frame standard deviation $0.5^{\circ}/\sqrt{16.7}$ and $\epsilon_n$ is zero-mean white noise with standard deviation $8.8^{\circ}$. Acceleration magnitude is synthesized as a step-frequency sinusoid (frequency sampled from 1.7--2.0~Hz) around gravity with additive Gaussian noise, then passed through the causal step detector of Section~\ref{sec:pdr}. Raw step displacements are summed within each of the 160 fusion bins according to~(\ref{eq:pdr}).''',
    'evaluation initialization and raw heading notation',
)

text = text.replace(r'produces $\mathbf{z}_{wifi}$', r'produces $\mathbf{z}_{\mathrm{wifi},t}$')
text = text.replace(r'produce $\mathbf{z}_{mag}$ and $\ell_{mag}$', r'produce $\mathbf{z}_{\mathrm{mag},t}$ and $\ell_{\mathrm{mag},t}$')
text = text.replace(r'$\ell_{ref}$ in~(\ref{eq:mag_ref})', r'$\ell_{\mathrm{ref}}$ in~(\ref{eq:mag_ref})')

# ---------------------------------------------------------------------------
# Tracker: close the mathematically addressed reviewer items.
# ---------------------------------------------------------------------------
tracker = replace_once(
    tracker,
    '### [ ] G2. Improve mathematical definitions of variables',
    '### [x] G2. Improve mathematical definitions of variables',
    'tracker G2',
)
tracker = replace_once(
    tracker,
    '### [ ] J9. General notation/spacing/capitalization cleanup in the problem formulation',
    '### [x] J9. General notation/spacing/capitalization cleanup in the problem formulation',
    'tracker J9',
)
tracker = replace_once(
    tracker,
    '### [ ] J12. Do not oversell standard signal-processing operations such as the low-pass filter',
    '### [x] J12. Do not oversell standard signal-processing operations such as the low-pass filter',
    'tracker J12',
)
tracker = replace_once(
    tracker,
    '### [ ] J13. Define a step-detection indicator and say explicitly what condition (11) detects',
    '### [x] J13. Define a step-detection indicator and say explicitly what condition (11) detects',
    'tracker J13',
)
tracker = replace_once(
    tracker,
    '### [ ] J17. Improve equation-to-equation prose flow',
    '### [x] J17. Improve equation-to-equation prose flow',
    'tracker J17',
)

tracker = replace_once(
    tracker,
    '**Status:** applicable; exact page positions may have moved.\n\n### [x] J10.',
    '**Resolved in notation/math-flow pass:** standardized prior/posterior and modality notation, separated raw-sample index $n$ from fusion-step index $t$, and aligned symbols in the problem formulation with the code-backed estimator equations.\n\n### [x] J10.',
    'J9 resolution note',
)
tracker = replace_once(
    tracker,
    '**Status:** applicable.\n\n### [ ] J13.',
    '**Resolved in notation/math-flow pass:** the EMA is presented only as the baseline used by the standard causal threshold detector; no novelty is attributed to the filter itself.\n\n### [x] J13.',
    'J12 resolution note',
)
tracker = replace_once(
    tracker,
    '**Status:** applicable if the PDR equations remain after the methodology audit.\n\n### [x] J14.',
    '**Resolved in notation/math-flow pass:** introduced the binary detector output $d_n$, defined its threshold/refractory event explicitly, defined the per-sample step displacement, and then defined the fusion-bin PDR control as the sum of detected-step displacements.\n\n### [x] J14.',
    'J13 resolution note',
)
tracker = replace_once(
    tracker,
    '**Status:** applicable and overlaps G2/G4.\n\n### [ ] J18.',
    '**Resolved in notation/math-flow pass:** state prediction, dual innovations, relative magnetic confidence, GRU gain generation, and posterior correction now form an explicit equation-to-equation chain with cross-references rather than disconnected symbol introductions.\n\n### [ ] J18.',
    'J17 resolution note',
)
tracker = replace_once(
    tracker,
    'Perform a notation audit across the whole paper. Define each signal/function/state before first use; distinguish scalars, vectors, matrices, functions, sets, random variables, and constants consistently; include time indices where relevant; avoid symbols whose domain/codomain is unclear; reuse equation numbers instead of redefining quantities informally.\n\n### [x] G3.',
    'Perform a notation audit across the whole paper. Define each signal/function/state before first use; distinguish scalars, vectors, matrices, functions, sets, random variables, and constants consistently; include time indices where relevant; avoid symbols whose domain/codomain is unclear; reuse equation numbers instead of redefining quantities informally.\n\n**Resolved in notation/math-flow pass:** raw sample $n$ and fusion step $t$ are now distinct; magnetic features are collected into an explicit $\mathbb{R}^4$ vector; Wi-Fi/magnetic measurements, masks, prior/posterior states, uncertainty scores, and gain matrices carry consistent time/modality indices; the 13-D GRU input is named explicitly; the GRU recurrence and 8-D gain head are defined mathematically; and the benchmark now states its known-start initialization condition.\n\n### [x] G3.',
    'G2 resolution note',
)

# Sanity checks for the intended final structure.
required = [
    r'\mathbf{f}^{\mathrm{mag}}_n',
    r'd_n=\mathbb{1}',
    r'\mathbf{u}_t=\sum_{n\in\mathcal{B}_t}\mathbf{v}_n',
    r'\mathbf{x}_{t}^{-}=\mathbf{x}_{t-1}+\mathbf{u}_t',
    r'\boldsymbol{\phi}_t\in\mathbb{R}^{13}',
    r'\operatorname{GRUCell}',
    r'\kappa_t^{\mathrm{wifi}}',
    r'Estimator initialization',
]
for needle in required:
    if needle not in text:
        raise RuntimeError(f'missing required final text: {needle}')

if r'\mathbf{x}_{\text{pred}}' in text:
    raise RuntimeError('legacy x_pred notation remains in manuscript')
if '### [ ] G2.' in tracker or '### [ ] J13.' in tracker or '### [ ] J17.' in tracker:
    raise RuntimeError('tracker items were not closed')

PAPER.write_text(text, encoding='utf-8')
TRACKER.write_text(tracker, encoding='utf-8')
print('Applied notation and mathematical-flow pass.')
