from pathlib import Path

paper = Path('paper/main.tex')
text = paper.read_text(encoding='utf-8')

old_eq = r'''\begin{equation}
w_{mag,t}=\frac{1}{1+\exp(\ell_{mag,t}-\ell_{ref})}
=\frac{1}{1+\sigma^2_{mag,t}/\sigma^2_{ref}}, \qquad 0<w_{mag,t}<1. \label{eq:mag_weight}
\end{equation}'''
new_eq = r'''\begin{equation}
\begin{aligned}
w_{mag,t}
&= \frac{1}{1+\exp(\ell_{mag,t}-\ell_{ref})} \\
&= \frac{1}{1+\sigma^2_{mag,t}/\sigma^2_{ref}},
\qquad 0<w_{mag,t}<1 .
\end{aligned}
\label{eq:mag_weight}
\end{equation}'''
if old_eq not in text:
    raise SystemExit('relative-variance equation pattern not found')
text = text.replace(old_eq, new_eq, 1)

old_table = r'''\begin{table}[htbp]
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
new_table = r'''\begin{table}[htbp]
\caption{CNN-Output DualKalmanNet Comparison (60 Test Walks)}
\label{tab:fusion_comparison}
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}p{0.18\columnwidth}p{0.35\columnwidth}p{0.25\columnwidth}p{0.12\columnwidth}@{}}
\toprule
\textbf{Regime} & \textbf{Model} & \textbf{Mean Err.} & \textbf{Med.} \\ \midrule
Full Wi-Fi (1 Hz)
& WiFi-only KalmanNet & \textbf{0.473 $\pm$ 0.035 m} & 0.449 m \\
& CNN DualKalmanNet & 0.506 $\pm$ 0.056 m & 0.440 m \\
& CNN Dual + rel. variance & 0.494 $\pm$ 0.046 m & \textbf{0.437 m} \\ \midrule
Degraded Wi-Fi (5 s, 40\% drop)
& WiFi-only KalmanNet & 1.533 $\pm$ 0.193 m & 1.392 m \\
& CNN DualKalmanNet & 1.171 $\pm$ 0.139 m & \textbf{1.042 m} \\
& \textbf{CNN Dual + rel. variance} & \textbf{1.154 $\pm$ 0.129 m} & 1.113 m \\ \bottomrule
\end{tabular}
\end{table}'''
if old_table not in text:
    raise SystemExit('fusion table pattern not found')
text = text.replace(old_table, new_table, 1)
paper.write_text(text, encoding='utf-8')

review = Path('paper/reviews/professor_feedback.md')
r = review.read_text(encoding='utf-8')
r = r.replace(
    '- [ ] R4: Explicitly define $A_{\\text{obs}} = m_N - \\bar{m}_N^{(\\text{dev})} = \\|\\mathbf{m}_t\\| - \\bar{m}_N^{(\\text{dev})}$ (linked to Eq. for $m_N$)',
    '- [x] R4: Resolved by removing the legacy scalar-anomaly path from the active architecture; $A_{\\text{obs}}$, $A(\\mathbf{x})$, and $\\nabla A$ are no longer used by the proposed fusion model.'
)
r = r.replace(
    '- [ ] R5: Resolved CNN-vs-scalar-gradient inconsistency — added "Note on the CNN" paragraph explaining CNN is standalone benchmark; scalar-gradient mechanism justified by causality constraint',
    '- [x] R5: Resolved by integrating the magnetic CNN output directly into DualKalmanNet: $\\mathbf{y}_{mag}=\\mathbf{z}_{mag}-\\mathbf{x}_{pred}$, with CNN log-variance used as a GRU confidence input and a training-normalized relative correction weight.'
)
marker = '- [x] R9: Moved postural independence discussion from Section IV.C to end of Section II.C (Magnetic Sequence Matcher)\n'
todo = '''\n### Follow-up figure TODO (not an unresolved professor comment)\n\n- [ ] Recreate the trajectory visualization using the current CNN-output, relative-variance-weighted DualKalmanNet. The legacy trajectory image represented the old scalar-anomaly fusion path and was removed when R4/R5 were resolved. Once regenerated, re-add/update the accompanying trajectory discussion if the figure remains useful.\n'''
if todo.strip() not in r:
    if marker not in r:
        raise SystemExit('round-2 status marker not found')
    r = r.replace(marker, marker + todo, 1)
review.write_text(r, encoding='utf-8')
