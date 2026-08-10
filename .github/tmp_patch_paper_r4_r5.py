from pathlib import Path
import re

path = Path('paper/main.tex')
text = path.read_text(encoding='utf-8')

new_abstract = r'''\begin{abstract}
Indoor positioning using hybrid smartphone sensors (Wi-Fi, magnetic field, and IMU) has seen growing interest, but deep-learning models often struggle to generalize in real-world deployments. This paper identifies key limitations in current regression methods, including trajectory memorization, body-frame magnetic direction dependency, and non-causal look-ahead bias. To address these issues, we propose a generalized, decoupled, and strictly causal Neural-Kalman fusion framework for hybrid indoor localization. A Multi-Layer Perceptron (MLP) environment model maps Wi-Fi fingerprints from $N$ Access Points to a 2D probability heatmap over $M$ discrete nodes, while a 1D Convolutional Neural Network (CNN) maps an 84-frame rotation-invariant magnetic sequence to a 2D position fix $\mathbf{z}_{mag}$ and a predicted log-variance. These spatial measurements are fused with a Pedestrian Dead Reckoning (PDR) motion model using a DualKalmanNet whose GRU predicts independent $2\times2$ Wi-Fi and magnetic gain matrices. The magnetic correction is further scaled by a relative confidence weight derived from the CNN variance and normalized using the median uncertainty observed on the fusion training trajectories. On 60 independent map-constrained test trajectories, the variance-weighted model achieves 0.494~m mean error (0.437~m median) under 1~Hz Wi-Fi, compared with 0.473~m (0.449~m median) for Wi-Fi-only KalmanNet. Under degraded Wi-Fi (5~s updates with 40\% AP dropout), it reduces the mean error from 1.533~m to 1.154~m, a 24.7\% improvement, while substantially reducing the high-error tail.
\end{abstract}'''
text, n = re.subn(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda m: new_abstract, text, count=1, flags=re.S)
if n != 1:
    raise SystemExit('abstract replacement failed')

replacements = {
    r'''    \node [block, fill=orange!15, below=0.6cm of wifi] (mag) {Mag CNN \\ ($\mathbf{z}_{mag}$)};''':
    r'''    \node [block, fill=orange!15, below=0.6cm of wifi] (mag) {Mag CNN \\ ($\mathbf{z}_{mag},\,\log\sigma^2_{mag}$)};''',
    r'''    \node [block, fill=green!15, right=1.5cm of innov] (update) {Update \\ $\mathbf{x}_t$};''':
    r'''    \node [block, fill=green!15, right=1.5cm of innov] (update) {Weighted Update \\ $\mathbf{x}_t$};''',
    r'''    \path [line] (mag.east) -- ++(0.4,0) |- (innov.200); % Routes mag into innovations
    
    \path [line] (predict) -- (innov);''':
    r'''    \path [line] (mag.east) -- ++(0.4,0) |- (innov.200); % Routes z_mag into innovations
    \path [line] (mag) -- (gru) node[midway, below, font=\scriptsize] {$\log\sigma^2_{mag}$};
    
    \path [line] (predict) -- (innov);''',
    r'''\caption{The proposed Neural-Kalman Fusion Architecture. Both the Wi-Fi probability heatmap and the continuous 1D-CNN magnetic fix generate innovations against the PDR prediction. A recurrent Dual GRU observes these innovations to compute optimal Kalman gain matrices for the state update.}''':
    r'''\caption{The proposed Neural-Kalman fusion architecture. The Wi-Fi heatmap and magnetic CNN produce Cartesian innovations against the PDR prediction. The magnetic CNN also provides a log-variance confidence signal. A recurrent Dual GRU predicts independent $2\times2$ gain matrices, while relative magnetic uncertainty explicitly weights the magnetic correction before the posterior update.}''',
    r'''The CNN processes the input to yield both a continuous position fix $\mathbf{z}_{mag} \in \mathbb{R}^2$ and a calibrated scalar log-variance $\log(\sigma^2_{mag})$.''':
    r'''The CNN processes the input to yield both a continuous position fix $\mathbf{z}_{mag} \in \mathbb{R}^2$ and a predicted scalar log-variance $\ell_{mag}=\log(\sigma^2_{mag})$.''',
    r'''where $B$ is the batch size, $\mathbf{z}_{true}$ is the ground-truth coordinate, and the variance $\sigma^2_{mag} = \exp(\text{logvar})$ is clamped to a minimum of 0.01 for numerical stability.''':
    r'''where $B$ is the batch size, $\mathbf{z}_{true}$ is the ground-truth coordinate, and the variance $\sigma^2_{mag} = \exp(\ell_{mag})$ is clamped to a minimum of 0.01 for numerical stability. In the fusion stage, the raw variance is not treated as an absolutely calibrated Kalman covariance. Instead, it is used as a relative confidence signal normalized by a reference uncertainty computed exclusively from the fusion training trajectories, as described in Section~\ref{sec:kalmannet}.''',
    r'''Figure~\ref{fig:merged_cdfs} (top)  displays the continuous spatial matching error CDF for the 1D-CNN magnetic sequence model (MAE of 3.58~m), validating its capability as an effective high-rate structural anchor during periods of Wi-Fi attenuation.''':
    r'''Figure~\ref{fig:cdf_mag} displays the continuous spatial matching error CDF for the 1D-CNN magnetic sequence model (MAE of 3.58~m), validating its capability as a structural spatial anchor during periods of Wi-Fi attenuation.'''
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f'expected manuscript text not found: {old[:80]}')
    text = text.replace(old, new, 1)

dual_section = r'''\subsection{Dual-Innovation KalmanNet Fusion}\label{sec:kalmannet}

Classical EKFs rely on fixed covariance assumptions that do not adequately capture the time-varying reliability of smartphone measurements. We therefore adopt KalmanNet~\cite{kalmannet}, replacing the analytical Kalman gain with gain matrices predicted by a Gated Recurrent Unit (GRU). The causal PDR prediction is
\begin{equation}
\mathbf{x}_{\text{pred}} = \mathbf{x}_{t-1} + \mathbf{u}_t . \label{eq:kn_predict}
\end{equation}
Both learned environment models are then expressed in the same Cartesian state space. The Wi-Fi innovation is
\begin{equation}
\mathbf{y}_{wifi} = \mathbf{z}_{wifi} - \mathbf{x}_{\text{pred}} \in \mathbb{R}^2, \label{eq:kn_innovate_w}
\end{equation}
and, crucially, the magnetic innovation directly uses the 2D output of the magnetic CNN:
\begin{equation}
\mathbf{y}_{mag} = \mathbf{z}_{mag} - \mathbf{x}_{\text{pred}} \in \mathbb{R}^2. \label{eq:kn_innovate_m}
\end{equation}
Thus the active fusion path contains no separate scalar anomaly observation $A_{\text{obs}}$, anomaly map $A(\mathbf{x})$, or spatial anomaly gradient $\nabla A$; the same CNN output introduced in Section~II-C is the magnetic measurement consumed by KalmanNet.

The CNN additionally predicts $\ell_{mag,t}=\log \sigma^2_{mag,t}$. Because the uncertainty head is used primarily to rank the reliability of magnetic fixes rather than as an absolute covariance, we normalize it using a training-only reference
\begin{equation}
\ell_{ref}=\operatorname{median}_{(w,t)\in\mathcal{D}_{train}:\,m_{mag}=1}\ell_{mag,w,t},
\qquad \sigma^2_{ref}=\exp(\ell_{ref}). \label{eq:mag_ref}
\end{equation}
The relative magnetic confidence weight is then
\begin{equation}
w_{mag,t}=\frac{1}{1+\exp(\ell_{mag,t}-\ell_{ref})}
=\frac{1}{1+\sigma^2_{mag,t}/\sigma^2_{ref}}, \qquad 0<w_{mag,t}<1. \label{eq:mag_weight}
\end{equation}
A prediction whose variance is above the training-set reference is therefore automatically down-weighted, whereas a relatively confident magnetic fix retains a larger correction. The log-variance itself is also retained as a GRU input so that the network may learn additional context-dependent reliability beyond this explicit weighting.

The GRU receives 13 scalar inputs per time step: the Wi-Fi innovation $\mathbf{y}_{wifi}\in\mathbb{R}^2$, magnetic-CNN innovation $\mathbf{y}_{mag}\in\mathbb{R}^2$, temporal difference of consecutive Wi-Fi fixes $\Delta\mathbf{z}_{wifi}\in\mathbb{R}^2$, PDR control $\mathbf{u}_t\in\mathbb{R}^2$, previous state update $\Delta\mathbf{x}_{t-1}\in\mathbb{R}^2$, two binary availability masks $m_{wifi},m_{mag}\in\{0,1\}$, and the scalar magnetic log-variance $\ell_{mag,t}$. The GRU outputs eight values, reshaped into independent $2\!\times\!2$ matrices $\mathbf{K}_{wifi}$ and $\mathbf{K}_{mag}$. The posterior update is
\begin{equation}
\mathbf{x}_t = \mathbf{x}_{\text{pred}}
+ m_{wifi}\,\mathbf{K}_{wifi}\mathbf{y}_{wifi}
+ m_{mag}\,w_{mag,t}\,\mathbf{K}_{mag}\mathbf{y}_{mag}. \label{eq:kn_correct}
\end{equation}
The availability masks preserve operation under missing modalities: setting $m_{wifi}=0$ or $m_{mag}=0$ removes the corresponding correction. When a magnetic window is available but its predicted variance is large, $w_{mag,t}$ suppresses that correction before it can dominate the state estimate. This combines the learned, context-dependent gains of KalmanNet with an explicit relative-confidence prior from the magnetic CNN.

'''
pattern = r'\\subsection\{Dual-Innovation KalmanNet Fusion\}\\label\{sec:kalmannet\}.*?(?=\\section\{Experimental Setup and Dataset\})'
text, n = re.subn(pattern, lambda m: dual_section, text, count=1, flags=re.S)
if n != 1:
    raise SystemExit('DualKalmanNet section replacement failed')

old_training = r'''The KalmanNet GRU was trained for 150 epochs via Adam with a learning rate of $2 \times 10^{-3}$, weight decay of $10^{-5}$, and MSE loss against ground-truth trajectories.'''
new_training = r'''The KalmanNet GRU was trained for 150 epochs via Adam with a learning rate of $2 \times 10^{-3}$, weight decay of $10^{-5}$, and MSE loss against ground-truth trajectories. For each signal-availability regime, the reference magnetic log-variance $\ell_{ref}$ in~(\ref{eq:mag_ref}) is computed only from the 250 fusion-training trajectories and is then frozen for evaluation.'''
if old_training not in text:
    raise SystemExit('training-details sentence not found')
text = text.replace(old_training, new_training, 1)

new_table = r'''\begin{table}[htbp]
\caption{CNN-Output DualKalmanNet Comparison (60 Test Walks)}
\label{tab:fusion_comparison}
\centering
\small
\begin{tabular}{@{}llcc@{}}
\toprule
\textbf{Regime} & \textbf{Model Configuration} & \textbf{Mean Err.} & \textbf{Med.} \\ \midrule
\multirow{3}{*}{\shortstack[l]{Full Wi-Fi\\(1 Hz)}}
& WiFi-only KalmanNet & \textbf{0.473 $\pm$ 0.035 m} & 0.449 m \\
& CNN DualKalmanNet & 0.506 $\pm$ 0.056 m & 0.440 m \\
& CNN Dual + relative variance & 0.494 $\pm$ 0.046 m & \textbf{0.437 m} \\ \midrule
\multirow{3}{*}{\shortstack[l]{Degraded Wi-Fi\\(5s gap, 40\% Drop)}}
& WiFi-only KalmanNet & 1.533 $\pm$ 0.193 m & 1.392 m \\
& CNN DualKalmanNet & 1.171 $\pm$ 0.139 m & \textbf{1.042 m} \\
& \textbf{CNN Dual + relative variance} & \textbf{1.154 $\pm$ 0.129 m} & 1.113 m \\ \bottomrule
\end{tabular}
\end{table}'''
table_pattern = r'\\begin\{table\}\[htbp\]\n\\caption\{Dual-Innovation Fusion Comparison \(60 Test Walks\)\}.*?\\end\{table\}'
text, n = re.subn(table_pattern, lambda m: new_table, text, count=1, flags=re.S)
if n != 1:
    raise SystemExit('fusion table replacement failed')

new_mag_figure = r'''\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\columnwidth]{figures/mag_cdf_cropped.png}
\caption{Standalone 1D-CNN magnetic sequence matcher error CDF (MAE 3.58~m).}
\label{fig:cdf_mag}
\end{figure}'''
fig_pattern = r'\\begin\{figure\}\[htbp\]\n\\centering\n\\begin\{subfigure\}.*?\\label\{fig:merged_cdfs\}\n\\end\{figure\}'
text, n = re.subn(fig_pattern, lambda m: new_mag_figure, text, count=1, flags=re.S)
if n != 1:
    raise SystemExit('legacy fusion CDF figure replacement failed')

new_results = r'''Under full 1~Hz Wi-Fi, the Wi-Fi-only KalmanNet retains the lowest mean error at 0.473~m, while the variance-weighted CNN fusion obtains 0.494~m. However, its median error improves slightly from 0.449~m to 0.437~m, and relative variance weighting improves the CNN-fusion mean from 0.506~m to 0.494~m. The benefit becomes substantially clearer when Wi-Fi is degraded. With one Wi-Fi update every 5~s and 40\% AP dropout, directly fusing the CNN output reduces the mean error from 1.533~m to 1.171~m (23.6\%). Adding relative magnetic-variance weighting further reduces the mean to 1.154~m, corresponding to a 24.7\% reduction over Wi-Fi-only KalmanNet. The weighting particularly suppresses poor magnetic fixes: the degraded-regime P90 decreases from 2.064~m for the unweighted CNN fusion to 1.612~m with relative variance weighting. These results show that the magnetic CNN is most valuable as an absolute spatial anchor during sparse Wi-Fi periods, while its predicted uncertainty helps limit the high-error tail.'''
result_pattern = r'In standard operating conditions featuring high-frequency 1~Hz Wi-Fi fixes,.*?(?=\\begin\{figure\}\[htbp\]\n\\centering\n\\includegraphics\[width=0\.9\\columnwidth\]\{figures/trajectory_example\.png\})'
text, n = re.subn(result_pattern, lambda m: new_results + '\n\n', text, count=1, flags=re.S)
if n != 1:
    raise SystemExit('fusion results paragraph replacement failed')

trajectory_pattern = r'\\begin\{figure\}\[htbp\]\n\\centering\n\\includegraphics\[width=0\.9\\columnwidth\]\{figures/trajectory_example\.png\}.*?preventing further error accumulation\.\n\n'
text, n = re.subn(trajectory_pattern, '', text, count=1, flags=re.S)
if n != 1:
    raise SystemExit('legacy anomaly trajectory removal failed')

old_conclusion = r'''In challenging, degraded scenarios—a common reality in physical deployments—the dual-innovation system effectively prevents inertial drift, delivering an impressive 25.3\% performance increase over standard models.'''
new_conclusion = r'''In challenging degraded scenarios, the CNN-output dual-innovation system effectively limits inertial drift. Relative magnetic-variance weighting reduces mean error from 1.533~m for Wi-Fi-only KalmanNet to 1.154~m, a 24.7\% improvement, while reducing the high-error tail by suppressing uncertain magnetic fixes.'''
if old_conclusion not in text:
    raise SystemExit('conclusion metric sentence not found')
text = text.replace(old_conclusion, new_conclusion, 1)

old_conclusion_arch = r'''The integration of a 1D-CNN magnetic sequence matcher alongside an MLP Wi-Fi probability heatmap enables robust spatial updates. Fusing these estimates through an extended KalmanNet architecture enables the system to continuously calculate context-dependent $2\!\times\!2$ matrix gains, leveraging availability masks to maintain resilience against missing sensor modalities.'''
new_conclusion_arch = r'''The integration of a 1D-CNN magnetic sequence matcher alongside an MLP Wi-Fi probability heatmap provides two Cartesian spatial measurements in the same state space. Fusing these estimates through DualKalmanNet enables context-dependent $2\!\times\!2$ matrix gains, while the CNN-predicted magnetic variance supplies a relative confidence weight that suppresses uncertain magnetic corrections and availability masks preserve operation under missing sensor modalities.'''
if old_conclusion_arch not in text:
    raise SystemExit('conclusion architecture sentence not found')
text = text.replace(old_conclusion_arch, new_conclusion_arch, 1)

for token in ['A_{\\text{obs}}', 'y_{mag} \\cdot \\nabla A', 'scalar magnetic anomaly corrections at 16.7~Hz']:
    if token in text:
        raise SystemExit(f'legacy R4/R5 token still present: {token}')

path.write_text(text, encoding='utf-8')
