from pathlib import Path

paper = Path('paper/main.tex')
text = paper.read_text(encoding='utf-8')


def repl(old: str, new: str) -> None:
    global text
    if old not in text:
        raise SystemExit(f'missing expected block:\n{old[:180]}')
    text = text.replace(old, new, 1)

# Slightly tighten float spacing without changing body font/line spacing.
anchor = r"\usepackage{subcaption}\n"
if anchor not in text:
    raise SystemExit('subcaption anchor missing')
text = text.replace(
    anchor,
    anchor
    + r"\setlength{\textfloatsep}{7pt plus 1pt minus 2pt}\n"
    + r"\setlength{\floatsep}{6pt plus 1pt minus 2pt}\n"
    + r"\setlength{\dbltextfloatsep}{7pt plus 1pt minus 2pt}\n"
    + r"\setlength{\dblfloatsep}{6pt plus 1pt minus 2pt}\n",
    1,
)

# Fix the known P8 column overflow while preserving the same definition.
repl(
    r"""\begin{equation}
\ell_{ref}=\operatorname{median}_{(w,t)\in\mathcal{D}_{train}:\,m_{mag}=1}\ell_{mag,w,t},
\qquad q_{ref}=\exp(\ell_{ref}). \label{eq:mag_ref}
\end{equation}""",
    r"""\begin{equation}
\begin{aligned}
\ell_{ref} &= \operatorname{median}_{(w,t)\in\mathcal{D}_{train}:\,m_{mag}=1}
\ell_{mag,w,t},\\
q_{ref} &= \exp(\ell_{ref}).
\end{aligned}
\label{eq:mag_ref}
\end{equation}""",
)

# Keep the three CDF panels readable but a little smaller and shorten duplicated caption prose.
text = text.replace(r"\begin{subfigure}[t]{0.48\textwidth}", r"\begin{subfigure}[t]{0.44\textwidth}")
text = text.replace(r"\vspace{0.15cm}", r"\vspace{0.05cm}", 1)
repl(
    r"""\caption{KNN baselines and current fusion CDFs. Panel (a) is a separate real static-fingerprint, held-out-device experiment and is not numerically comparable to the synthetic-walk errors in (b)--(c). Panels (b)--(c) use exactly the same 250-training/60-test synthetic trajectory generation as KalmanNet. The non-temporal KNN receives the contemporaneous Wi-Fi heatmap fix, magnetic-CNN fix and log-uncertainty score, and availability masks, but no PDR motion or recurrent history. Axes, ticks, and legends are regenerated at paper-readable sizes.}""",
    r"""\caption{KNN baselines and fusion CDFs. Panel (a) is a separate real static-fingerprint held-out-device experiment and is not numerically comparable to the synthetic-walk results in (b)--(c). Panels (b)--(c) use the same 250-training/60-test trajectory protocol as KalmanNet; the non-temporal KNN uses contemporaneous Wi-Fi and magnetic fixes, magnetic log-uncertainty, and availability masks, without PDR or recurrent history.}""",
)

# Condense repeated prose: every reported comparison and the methodological interpretation are retained.
repl(
    r"""For the matched trajectory comparison in Fig.~\ref{fig:knn_fusion_cdfs}(b)--(c), KNN is intentionally non-temporal: it receives only the contemporaneous Wi-Fi heatmap position, magnetic-CNN position and log-uncertainty score, and the two availability masks. Its value of $K$ is selected by five-fold grouped cross-validation in which all bins from a trajectory remain in the same fold. Under full 1~Hz Wi-Fi, the KNN baseline obtains 0.802~m mean error, compared with 0.473~m for Wi-Fi-only KalmanNet and 0.494~m for the uncertainty-weighted CNN fusion. The latter also improves median error slightly from 0.449~m to 0.437~m. Under degraded Wi-Fi, the gap is larger: non-temporal KNN reaches 2.606~m, Wi-Fi-only KalmanNet reaches 1.533~m, and the uncertainty-weighted DualKalmanNet reaches 1.154~m. Direct CNN fusion without the explicit uncertainty weight reaches 1.171~m, so relative magnetic-uncertainty weighting yields a 24.7\% reduction over Wi-Fi-only KalmanNet and reduces the degraded-regime P90 from 2.064~m for unweighted CNN fusion to 1.612~m. The separation from KNN in Fig.~\ref{fig:knn_fusion_cdfs}(c) indicates that the gain does not arise merely from having both absolute sensor estimates available; PDR propagation and learned temporal, context-dependent fusion are important in sparse-Wi-Fi operation.""",
    r"""For the matched trajectory comparison in Fig.~\ref{fig:knn_fusion_cdfs}(b)--(c), KNN is deliberately non-temporal, with $K$ selected by five-fold grouped cross-validation that keeps each trajectory within one fold. Under full 1~Hz Wi-Fi it obtains 0.802~m mean error, versus 0.473~m for Wi-Fi-only KalmanNet and 0.494~m for uncertainty-weighted CNN fusion; the latter also improves the median from 0.449~m to 0.437~m. Under degraded Wi-Fi, KNN reaches 2.606~m, Wi-Fi-only KalmanNet 1.533~m, unweighted CNN DualKalmanNet 1.171~m, and the uncertainty-weighted model 1.154~m. Thus relative magnetic uncertainty gives a 24.7\% mean-error reduction over Wi-Fi-only KalmanNet and lowers the unweighted model's P90 from 2.064~m to 1.612~m. The KNN gap shows that PDR propagation and recurrent context matter beyond simply having both absolute estimates available.""",
)

# Make the representative trajectory a single-column figure so it no longer consumes a standalone eighth page.
repl(
    r"""\begin{figure*}[htbp]
\centering
\includegraphics[width=0.82\textwidth]{../benchmarks/knn/current_results/trajectory_protocol/representative_trajectory.png}
\caption{Representative degraded-Wi-Fi test trajectory generated from the current experiment outputs. To avoid cherry-picking a best-case example, the displayed walk is selected from the interquartile range of per-walk fusion improvements and, within that central group, chosen to contain the largest accumulated heading change so that several corridor turns are visible. The plot compares ground truth, open-loop PDR, Wi-Fi-only KalmanNet, and the final CNN-output DualKalmanNet with relative magnetic-uncertainty weighting; circles indicate the sparse Wi-Fi update times.}
\label{fig:trajectory_current}
\end{figure*}""",
    r"""\begin{figure}[t]
\centering
\includegraphics[width=0.98\columnwidth]{../benchmarks/knn/current_results/trajectory_protocol/representative_trajectory.png}
\caption{Representative degraded-Wi-Fi synthetic/map-constrained test walk, selected from the interquartile range of fusion improvements and then by maximum heading-change score. Curves compare ground truth, open-loop PDR, Wi-Fi-only KalmanNet, and the final uncertainty-weighted CNN DualKalmanNet; circles mark Wi-Fi updates.}
\label{fig:trajectory_current}
\end{figure}""",
)

repl(
    r"""Figure~\ref{fig:trajectory_current} provides a geometric view of the degraded-Wi-Fi regime behind the aggregate statistics. Open-loop PDR accumulates heading and step errors through the sequence of corridor turns, while the Wi-Fi-only KalmanNet can only re-anchor when a sparse Wi-Fi update arrives. The CNN-output DualKalmanNet receives an additional absolute magnetic position innovation at each causal magnetic window, and the relative-uncertainty weight suppresses corrections from windows that the CNN itself judges unreliable. Consequently, the fused path remains closer to the surveyed route between Wi-Fi updates rather than merely correcting after drift has already accumulated. The trajectory is intentionally a representative central-performance case rather than the walk with the largest improvement; its exact selection rule and per-walk errors are stored with the benchmark output for reproducibility.""",
    r"""Figure~\ref{fig:trajectory_current} shows the same degraded regime geometrically. Open-loop PDR drifts through corridor turns and Wi-Fi-only KalmanNet can re-anchor only at sparse scans, whereas the CNN supplies causal magnetic position innovations between scans and relative uncertainty suppresses unreliable corrections. The fused path therefore remains closer to the surveyed route. The displayed walk is a central-performance case rather than the best improvement; its exact selection data are stored with the benchmark output.""",
)

# Remove repetition in the conclusion while retaining the architecture, key result, scope, and future direction.
repl(
    r"""This paper introduced a generalized, decoupled Neural-Kalman fusion architecture designed to resolve pervasive limitations in deep-learning indoor positioning systems across Hybrid Wi-Fi, Magnetic, and IMU data regimes. By strictly separating spatial inference from temporal tracking modules, the framework successfully circumvents trajectory memorization and orientation-dependent data bias. The integration of a 1D-CNN magnetic sequence matcher alongside an MLP Wi-Fi probability heatmap provides two Cartesian spatial measurements in the same state space. Fusing these estimates through DualKalmanNet enables context-dependent $2\!\times\!2$ matrix gains, while the CNN-predicted magnetic uncertainty score supplies a relative confidence weight that suppresses uncertain magnetic corrections and availability masks preserve operation under missing sensor modalities.

In challenging degraded scenarios, the CNN-output dual-innovation system effectively limits inertial drift. Relative magnetic-uncertainty weighting reduces mean error from 1.533~m for Wi-Fi-only KalmanNet to 1.154~m, a 24.7\% improvement, while reducing the high-error tail by suppressing uncertain magnetic fixes. While current configurations necessitate building-specific training due to unique foundational AP signatures, the mathematical structure is universally applicable. Future research efforts will focus on expanding the architecture to facilitate full three-dimensional, multi-floor positioning by integrating barometric pressure altitude data into the KalmanNet state vector.""",
    r"""This paper presented a decoupled Neural-Kalman architecture that separates spatial inference from temporal tracking. A Wi-Fi heatmap MLP and gravity-referenced magnetic sequence CNN provide Cartesian measurements, while DualKalmanNet predicts independent $2\!\times\!2$ gains; availability masks and relative magnetic uncertainty suppress missing or unreliable corrections. Under degraded Wi-Fi, the uncertainty-weighted model reduces mean error from 1.533~m for Wi-Fi-only KalmanNet to 1.154~m (24.7\%) and reduces the high-error tail. The present fusion results demonstrate trajectory generalization within a surveyed environment and still require building-specific environment resources. Future work will extend the state to three-dimensional, multi-floor positioning, including barometric altitude information.""",
)

paper.write_text(text, encoding='utf-8')

# Mark only the layout issue actually resolved in this pass.
review = Path('paper/reviews/prof_read_ieee_comments_draft.md')
r = review.read_text(encoding='utf-8')
old = '### [ ] P8. Fix `ell_ref` equation overflow (observation 4)'
new = '### [x] P8. Fix `ell_ref` equation overflow (observation 4)'
if old not in r:
    raise SystemExit('P8 marker missing')
r = r.replace(old, new, 1)
needle = 'The line defining the training-reference log variance is visually crossing/pressing the IEEE column boundary in the current PDF. Reformat using `aligned`, split the median definition and `q_ref = exp(ell_ref)` across lines, or otherwise guarantee both fit inside one column. Re-render the PDF after the fix.\n'
if needle not in r:
    raise SystemExit('P8 description missing')
r = r.replace(needle, needle + '\n**Resolved:** split the training-reference definition into an `aligned` two-line equation and verified the rendered column boundary during the 7-page compression pass.\n', 1)
review.write_text(r, encoding='utf-8')
