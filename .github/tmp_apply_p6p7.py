from pathlib import Path
import re

paper_path = Path('paper/main.tex')
text = paper_path.read_text(encoding='utf-8')

# Abstract terminology: the magnetic head is a relative uncertainty score, not a calibrated variance.
old = "while a 1D Convolutional Neural Network (CNN) maps an 84-frame rotation-invariant magnetic sequence to a 2D position fix $\\mathbf{z}_{mag}$ and a predicted log-variance. These spatial measurements are fused with a Pedestrian Dead Reckoning (PDR) motion model using a DualKalmanNet whose GRU predicts independent $2\\times2$ Wi-Fi and magnetic gain matrices. The magnetic correction is further scaled by a relative confidence weight derived from the CNN variance and normalized using the median uncertainty observed on the fusion training trajectories. On 60 independent map-constrained test trajectories, the variance-weighted model achieves"
new = "while a 1D Convolutional Neural Network (CNN) maps an 84-frame gravity-referenced magnetic sequence to a 2D position fix $\\mathbf{z}_{mag}$ and a learned scalar log-uncertainty score. These spatial measurements are fused with a Pedestrian Dead Reckoning (PDR) motion model using a DualKalmanNet whose GRU predicts independent $2\\times2$ Wi-Fi and magnetic gain matrices. The magnetic correction is further scaled by a relative confidence weight obtained by normalizing this score against the median uncertainty observed on the fusion training trajectories. On 60 independent map-constrained test trajectories, the uncertainty-weighted model achieves"
if old not in text:
    raise SystemExit('abstract target not found')
text = text.replace(old, new)

# Architecture figure labels/caption.
text = text.replace('($\\mathbf{z}_{mag},\\,\\log\\sigma^2_{mag}$)', '($\\mathbf{z}_{mag},\\,\\ell_{mag}$)')
text = text.replace('node[midway, below, font=\\scriptsize] {$\\log\\sigma^2_{mag}$}', 'node[midway, below, font=\\scriptsize] {$\\ell_{mag}$}')
text = text.replace(
    'The magnetic CNN also provides a log-variance confidence signal. A recurrent Dual GRU predicts independent $2\\times2$ gain matrices, while relative magnetic uncertainty explicitly weights the magnetic correction before the posterior update.',
    'The magnetic CNN also provides a learned scalar log-uncertainty score $\\ell_{mag}$. A recurrent Dual GRU predicts independent $2\\times2$ gain matrices, while relative magnetic uncertainty explicitly weights the magnetic correction before the posterior update.'
)

# Replace the entire magnetic subsection with a code-accurate description.
pattern = re.compile(r"\\subsection\{Magnetic Sequence Matcher\}.*?(?=\\subsection\{Causal Pedestrian Dead Reckoning \(PDR\)\})", re.S)
replacement = r'''\subsection{Magnetic Survey Processing and Sequence Matcher}

The magnetic path begins with gravity-referenced survey preprocessing. For each raw static sample, let $\mathbf{m}_t\in\mathbb{R}^3$ be the magnetometer vector and $\mathbf{a}_t\in\mathbb{R}^3$ the accelerometer vector. The implementation uses the instantaneous normalized acceleration
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
These quantities are invariant to a common rigid rotation of the phone when $\hat{\mathbf{a}}_t$ is a valid gravity proxy. The processed survey database stores the mean and standard deviation of each quantity over a static visit. The magnetic-map trainer then subtracts each phone's mean feature value, averages the centered visit means at each surveyed node, and interpolates the four channels $(m_N,m_V,m_H,\delta)$ onto the same 1~m Cartesian environment grid used by the Wi-Fi model.

The CNN is trained on survey-derived, map-constrained magnetic sequences rather than directly on the raw MagWi continuous recordings. Random corridor paths are sampled from the surveyed node graph, interpolated at 16.7~Hz, and used to bilinearly sample the four-channel magnetic map; channel-wise Gaussian noise is added using the measured within-node variability. A causal window ending at time $t$ is therefore
\begin{equation}
\mathbf{M}_t\in\mathbb{R}^{T\times4},\qquad T=84,
\label{eq:mag_window}
\end{equation}
with the Cartesian position at the final frame used as its regression target. In the final fusion benchmark, the same 84-frame causal convention is used: a magnetic fix is emitted only after a complete window ending at the current fusion time is available.

\begin{figure*}[htbp]
\centering
\resizebox{\textwidth}{!}{
\begin{tikzpicture}[
    conv/.style={rectangle, draw=black!60, thick, fill=orange!15, text centered, rounded corners=2pt, minimum height=1cm, drop shadow, text width=2cm},
    pool/.style={rectangle, draw=black!60, thick, fill=yellow!15, text centered, rounded corners=2pt, minimum height=1cm, drop shadow, text width=1.6cm},
    head/.style={rectangle, draw=black!60, thick, fill=green!15, text centered, rounded corners=2pt, minimum height=1cm, drop shadow, text width=2.2cm},
    arrow/.style={draw, thick, -latex, color=black!70},
    groupbox/.style={rectangle, draw=black!30, dashed, thick, inner sep=6pt, rounded corners}
]
    \node (input) {$\mathbb{R}^{84 \times 4}$};
    \node[conv, right=0.5cm of input] (conv1) {Conv1D(32)\\$k\!=\!7$, BN, ReLU};
    \node[pool, right=0.25cm of conv1] (pool1) {MaxPool\\$84\!\to\!42$};
    \node[conv, right=0.25cm of pool1] (conv2) {Conv1D(64)\\$k\!=\!5$, BN, ReLU};
    \node[pool, right=0.25cm of conv2] (pool2) {MaxPool\\$42\!\to\!21$};
    \node[conv, right=0.25cm of pool2] (conv3) {Conv1D(128)\\$k\!=\!3$, BN, ReLU};
    \node[pool, right=0.25cm of conv3] (gap) {Adaptive\\AvgPool};
    \begin{scope}[on background layer]
        \node[groupbox, fill=gray!5, fit=(conv1) (gap)] (encoder_box) {};
    \end{scope}
    \node[head, right=1cm of gap, yshift=0.8cm] (pos_head) {FC(64), ReLU\\Drop(0.2), FC(2)};
    \node[head, right=1cm of gap, yshift=-0.8cm] (unc_head) {FC(32), ReLU\\FC(1)};
    \node[right=0.5cm of pos_head] (outpos) {$\mathbf{z}_{mag}$};
    \node[right=0.5cm of unc_head] (outunc) {$\ell_{mag}$};
    \path [arrow] (input) -- (conv1);
    \path [arrow] (conv1) -- (pool1);
    \path [arrow] (pool1) -- (conv2);
    \path [arrow] (conv2) -- (pool2);
    \path [arrow] (pool2) -- (conv3);
    \path [arrow] (conv3) -- (gap);
    \draw [arrow] (gap.east) -- ++(0.3,0) |- (pos_head.west);
    \draw [arrow] (gap.east) -- ++(0.3,0) |- (unc_head.west);
    \path [arrow] (pos_head) -- (outpos);
    \path [arrow] (unc_head) -- (outunc);
\end{tikzpicture}
}
\caption{Code-accurate architecture of the magnetic sequence CNN for $T=84$. The shared Conv1D encoder reduces the temporal length $84\to42\to21$ before adaptive global averaging into a 128-dimensional representation. Separate heads predict the 2D magnetic position fix $\mathbf{z}_{mag}$ and a scalar log-uncertainty score $\ell_{mag}$.}
\label{fig:cnn_arch}
\end{figure*}

For $T=84$, the encoder maps $[B,4,84]$ to $[B,32,84]$, $[B,32,42]$, $[B,64,42]$, $[B,64,21]$, and finally $[B,128,21]$. Adaptive average pooling aggregates all 21 temporal locations into a shared $128$-dimensional vector. The position head is $128\!\to\!64\!\to\!2$ with ReLU and dropout $p=0.2$, while the scalar uncertainty head is $128\!\to\!32\!\to\!1$ with ReLU in its hidden layer. Thus the second output is learned from the same complete 84-frame representation as the position estimate; it is not computed analytically from the position residual at inference time.

Let $\ell_{mag}^{(i)}$ denote the scalar score for training window $i$ and define $q_{mag}^{(i)}=\exp(\ell_{mag}^{(i)})$. The historical training implementation uses the uncertainty-weighted regression objective
\begin{equation}
\mathcal{L}_{CNN}=\frac{1}{B}\sum_{i=1}^{B}\left[
\frac{\|\mathbf{z}_{mag}^{(i)}-\mathbf{z}_{true}^{(i)}\|^2}{2\,\tilde q_{mag}^{(i)}}
+\frac{1}{2}\ell_{mag}^{(i)}\right],
\label{eq:nll_loss}
\end{equation}
where $\tilde q_{mag}^{(i)}=\max(q_{mag}^{(i)},0.01)$. A difficult sample can therefore receive a larger learned scale, reducing the first term, while the positive log-scale penalty discourages arbitrarily large uncertainty scores.

Importantly, (\ref{eq:nll_loss}) is \emph{not} the exact negative log-likelihood of a two-dimensional isotropic Gaussian: for a covariance $q\mathbf{I}_2$, the Gaussian normalization contributes $\log q$, rather than $\tfrac{1}{2}\log q$. We therefore do not interpret $q_{mag}$ as a calibrated Cartesian variance or covariance. It is used only as a learned relative uncertainty/difficulty scale. The separate calibration analysis shows that this score has useful reliability ordering but a conservative absolute scale; the fusion stage consequently uses only its training-normalized relative value.

The current synthetic fusion evaluation samples its magnetic inputs from the per-phone-centered survey map described above. It therefore evaluates temporal fusion within that surveyed magnetic domain; it does not establish that an uncalibrated unseen handset can be mapped into the same centered feature domain without an additional causal normalization procedure.

'''
text, count = pattern.subn(lambda m: replacement, text, count=1)
if count != 1:
    raise SystemExit(f'magnetic subsection replacement count={count}')

# Fusion terminology and equations: q is a positive relative uncertainty scale, not a covariance.
old = r'''The CNN additionally predicts $\ell_{mag,t}=\log \sigma^2_{mag,t}$. Because the uncertainty head is used primarily to rank the reliability of magnetic fixes rather than as an absolute covariance, we normalize it using a training-only reference
\begin{equation}
\ell_{ref}=\operatorname{median}_{(w,t)\in\mathcal{D}_{train}:\,m_{mag}=1}\ell_{mag,w,t},
\qquad \sigma^2_{ref}=\exp(\ell_{ref}). \label{eq:mag_ref}
\end{equation}
The relative magnetic confidence weight is then
\begin{equation}
\begin{aligned}
w_{mag,t}
&= \frac{1}{1+\exp(\ell_{mag,t}-\ell_{ref})} \\
&= \frac{1}{1+\sigma^2_{mag,t}/\sigma^2_{ref}},
\qquad 0<w_{mag,t}<1 .
\end{aligned}
\label{eq:mag_weight}
\end{equation}
A prediction whose variance is above the training-set reference is therefore automatically down-weighted, whereas a relatively confident magnetic fix retains a larger correction. The log-variance itself is also retained as a GRU input so that the network may learn additional context-dependent reliability beyond this explicit weighting.'''
new = r'''The CNN additionally predicts the scalar log-uncertainty score $\ell_{mag,t}$ and its positive scale $q_{mag,t}=\exp(\ell_{mag,t})$. Because only relative reliability is used in fusion, we normalize the score using a training-only reference
\begin{equation}
\ell_{ref}=\operatorname{median}_{(w,t)\in\mathcal{D}_{train}:\,m_{mag}=1}\ell_{mag,w,t},
\qquad q_{ref}=\exp(\ell_{ref}). \label{eq:mag_ref}
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
A magnetic prediction with an uncertainty score above the training reference is therefore down-weighted, whereas a relatively confident fix retains a larger correction. The log-uncertainty score itself is also retained as a GRU input so that the network may learn additional context-dependent reliability beyond this explicit weighting.'''
if old not in text:
    raise SystemExit('fusion uncertainty block target not found')
text = text.replace(old, new)

text = text.replace('the scalar magnetic log-variance $\\ell_{mag,t}$', 'the scalar magnetic log-uncertainty score $\\ell_{mag,t}$')
text = text.replace('When a magnetic window is available but its predicted variance is large, $w_{mag,t}$ suppresses that correction', 'When a magnetic window is available but its predicted uncertainty score is large, $w_{mag,t}$ suppresses that correction')
text = text.replace('magnetic log-variance. The generated ground-truth positions', 'magnetic log-uncertainty score. The generated ground-truth positions')
text = text.replace('reference magnetic log-variance $\\ell_{ref}$', 'reference magnetic log-uncertainty score $\\ell_{ref}$')

# Results/captions terminology only; numerical results remain unchanged.
text = text.replace('CNN Dual + rel. variance', 'CNN Dual + rel. confidence')
text = text.replace('\\textbf{CNN Dual + rel. variance}', '\\textbf{CNN Dual + rel. confidence}')
text = text.replace('magnetic-CNN fix and log-variance', 'magnetic-CNN fix and log-uncertainty score')
text = text.replace('magnetic-CNN position and log-variance', 'magnetic-CNN position and log-uncertainty score')
text = text.replace('variance-weighted CNN fusion', 'uncertainty-weighted CNN fusion')
text = text.replace('variance-weighted DualKalmanNet', 'uncertainty-weighted DualKalmanNet')
text = text.replace('without the explicit variance weight', 'without the explicit uncertainty weight')
text = text.replace('relative magnetic-variance weighting', 'relative magnetic-uncertainty weighting')
text = text.replace('relative magnetic-variance weighting; circles', 'relative magnetic-uncertainty weighting; circles')

paper_path.write_text(text, encoding='utf-8')

# Update proofreading tracker and preserve a newly uncovered deployment concern.
review_path = Path('paper/reviews/prof_read_ieee_comments_draft.md')
review = review_path.read_text(encoding='utf-8')
review = review.replace('### [ ] P6. Explain the CNN variance output precisely (observation 7)', '### [x] P6. Explain the CNN uncertainty output precisely (observation 7)')
review = review.replace('### [ ] P7. Deep code-backed walkthrough of the complete magnetic CNN architecture (observation 9)', '### [x] P7. Deep code-backed walkthrough of the complete magnetic CNN architecture (observation 9)')
marker = '## Priority C - paper presentation / proofreading\n'
if marker not in review:
    raise SystemExit('review priority-C marker not found')
insert = '''### P6-P7 implementation note\n\n- **P6:** the second CNN head is now described as a scalar **log-uncertainty score** rather than a calibrated Cartesian variance. The active loss is an uncertainty-weighted radial regression objective, `0.5*||e||^2/exp(ell) + 0.5*ell`; because a true 2-D isotropic Gaussian would carry a full `+ log(q)` normalization term, the paper no longer calls this exact objective a 2-D Gaussian NLL or treats `exp(ell)` as a covariance. The existing calibration benchmark supports relative reliability ranking, which is the only role used by final fusion.\n- **P7:** added `docs/architecture/magnetic_sequence_cnn.md` with the exact preprocessing/data-generation path and tensor shapes. For `T=84`, the Conv1D encoder follows `84 -> 42 -> 21` temporal samples and produces a shared 128-D representation, followed by `128->64->2` position and `128->32->1` uncertainty heads. The paper now states that current CNN training uses survey-derived map-constrained sequences rather than raw continuous MagWi trajectories, and corrects the static feature extractor to the actual instantaneous normalized-acceleration gravity proxy.\n\n### [ ] P11. Real-device magnetic centering / deployment gap uncovered during P6-P7\n\nThe current magnetic-map trainer subtracts each phone's mean feature value before node averaging and interpolation. The synthetic CNN/fusion evaluation then samples directly from this centered survey map. A physical unseen phone would need a causal normalization/calibration procedure to map live `magN/magV/magH/dip` features into the same centered domain, but this step is not presently implemented or evaluated. Keep held-out-device claims separate from the magnetic fusion experiment and decide whether to add an online centering strategy in a future experiment or explicitly scope the current paper to the surveyed magnetic domain.\n\n'''
review = review.replace(marker, insert + marker, 1)
review_path.write_text(review, encoding='utf-8')
