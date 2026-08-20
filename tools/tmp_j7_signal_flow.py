from pathlib import Path

paper_path = Path('paper/main.tex')
tracker_path = Path('paper/reviews/prof_read_ieee_comments_draft.md')
text = paper_path.read_text(encoding='utf-8')
tracker = tracker_path.read_text(encoding='utf-8')

old_intro = r'''The estimator forms Wi-Fi and magnetic Cartesian measurements, propagates PDR motion, and combines them with DualKalmanNet (Fig.~\ref{fig:arch_diagram}).'''
new_intro = r'''The estimator forms Wi-Fi and magnetic Cartesian measurements, propagates PDR motion, and combines them with DualKalmanNet. Figure~\ref{fig:arch_diagram} shows the functional signal flow of one fusion update; the Wi-Fi, magnetic, and PDR measurement models that produce its input signals are detailed in the following subsections.'''
if text.count(old_intro) != 1:
    raise RuntimeError(f'expected one estimator-intro sentence, found {text.count(old_intro)}')
text = text.replace(old_intro, new_intro, 1)

label = r'\label{fig:arch_diagram}'
label_pos = text.find(label)
if label_pos < 0:
    raise RuntimeError('architecture figure label not found')
start = text.rfind(r'\begin{figure}', 0, label_pos)
end_marker = r'\end{figure}'
end = text.find(end_marker, label_pos)
if start < 0 or end < 0:
    raise RuntimeError('architecture figure boundaries not found')
end += len(end_marker)

new_figure = r'''\begin{figure*}[t]
\centering
\resizebox{0.98\textwidth}{!}{
\begin{tikzpicture}[
    signal/.style={rectangle, rounded corners=7pt, draw=blue!70!black, thick, fill=blue!8, minimum height=0.72cm, minimum width=2.05cm, text centered, font=\scriptsize},
    block/.style={rectangle, rounded corners=2pt, draw=black!70, thick, fill=gray!8, minimum height=1.05cm, text width=3.55cm, text centered, font=\scriptsize},
    learned/.style={block, fill=purple!10, text width=3.65cm},
    confidence/.style={block, fill=orange!12, text width=3.55cm},
    correction/.style={block, fill=green!10, text width=5.1cm},
    line/.style={draw, -latex, thick, color=black!80},
    auxline/.style={draw, -latex, thick, dashed, color=black!55},
    lab/.style={font=\scriptsize, fill=white, inner sep=1pt}
]
    % External/current-step signals.
    \node[signal] (xprev) at (0,2.6) {$\mathbf{x}_{t-1}$};
    \node[signal] (motion) at (0,1.55) {$\mathbf{u}_t$};
    \node[signal] (wifi) at (0,0.20) {$\mathbf{z}_{\mathrm{wifi},t}$};
    \node[signal] (mag) at (0,-0.85) {$\mathbf{z}_{\mathrm{mag},t}$};
    \node[signal] (ell) at (0,-1.90) {$\ell_{\mathrm{mag},t}$};
    \node[signal, minimum width=2.65cm] (masks) at (0,-2.95) {$m_{\mathrm{wifi},t},\ m_{\mathrm{mag},t}$};
    \node[signal, minimum width=3.35cm] (history) at (0,-4.00) {$\Delta\mathbf{z}_{\mathrm{wifi},t},\ \Delta\mathbf{x}_{t-1}$};

    % Deterministic state-space transformations.
    \node[block] (prior) at (3.45,2.10) {Prior propagation\\[-1pt]$\mathbf{x}_t^-=\mathbf{x}_{t-1}+\mathbf{u}_t$};
    \node[block, text width=4.45cm] (innov) at (5.45,0.25) {Innovation formation\\[-1pt]
      $\mathbf{y}_{\mathrm{wifi},t}=\mathbf{z}_{\mathrm{wifi},t}-\mathbf{x}_t^-$\\[-1pt]
      $\mathbf{y}_{\mathrm{mag},t}=\mathbf{z}_{\mathrm{mag},t}-\mathbf{x}_t^-$};
    \node[block, text width=4.3cm] (features) at (5.45,-2.65) {13-D feature assembly\\[-1pt]
      $\boldsymbol{\phi}_t\in\mathbb{R}^{13}$\\[-1pt]
      innovations, motion, history, masks, confidence};
    \node[confidence] (conf) at (4.00,-4.35) {Relative magnetic confidence\\[-1pt]
      $w_{\mathrm{mag},t}=\bigl[1+e^{\ell_{\mathrm{mag},t}-\ell_{\mathrm{ref}}}\bigr]^{-1}$};

    % Learned recurrent gain generation.
    \node[signal] (hprev) at (8.85,-4.00) {$\mathbf{h}_{t-1}$};
    \node[learned] (gru) at (9.00,-2.10) {GRUCell + 8-D linear head\\[-1pt]
      $\mathbf{h}_t=\operatorname{GRUCell}(\boldsymbol{\phi}_t,\mathbf{h}_{t-1})$\\[-1pt]
      $\mathbf{g}_t\mapsto(\mathbf{K}_{\mathrm{wifi},t},\mathbf{K}_{\mathrm{mag},t})$};

    % Posterior correction and explicit output signal.
    \node[correction] (update) at (12.65,0.55) {Posterior correction\\[-1pt]
      $\mathbf{x}_t=\mathbf{x}_t^-+m_{\mathrm{wifi},t}\mathbf{K}_{\mathrm{wifi},t}\mathbf{y}_{\mathrm{wifi},t}$\\[-1pt]
      $\quad +m_{\mathrm{mag},t}w_{\mathrm{mag},t}\mathbf{K}_{\mathrm{mag},t}\mathbf{y}_{\mathrm{mag},t}$};
    \node[signal, draw=green!55!black, fill=green!8, minimum width=1.85cm] (xout) at (16.10,0.55) {$\mathbf{x}_t$};

    % Main state/measurement paths.
    \path[line] (xprev) -- (prior);
    \path[line] (motion) -- (prior);
    \path[line] (prior.east) -- ++(0.55,0) |- (innov.north) node[pos=0.28, lab] {$\mathbf{x}_t^-$};
    \path[line] (wifi) -- (innov.west);
    \path[line] (mag.east) -- ++(1.15,0) |- (innov.205);
    \path[line] (innov.south) -- (features.north) node[midway, lab] {$\mathbf{y}_{\mathrm{wifi},t},\ \mathbf{y}_{\mathrm{mag},t}$};

    % Recurrent feature paths.
    \path[auxline] (motion.east) -- ++(1.0,0) |- (features.165);
    \path[auxline] (history.east) -- ++(2.0,0) |- (features.210);
    \path[auxline] (masks.east) -- ++(1.45,0) |- (features.195);
    \path[auxline] (ell.east) -- ++(1.05,0) |- (features.180) node[pos=0.63, lab] {$c_{\mathrm{mag},t}$};
    \path[line] (features) -- (gru) node[midway, above, lab] {$\boldsymbol{\phi}_t$};
    \path[auxline] (hprev) -- (gru.south);

    % Confidence and posterior paths.
    \path[line] (ell.east) -- ++(0.65,0) |- (conf.west);
    \path[auxline] (masks.east) -- ++(0.85,0) |- (conf.165);
    \path[line] (conf.east) -- ++(5.25,0) |- (update.south) node[pos=0.73, lab] {$w_{\mathrm{mag},t}$};
    \path[line] (gru.east) -- ++(1.05,0) |- (update.205) node[pos=0.30, lab] {$\mathbf{K}_{\mathrm{wifi},t},\ \mathbf{K}_{\mathrm{mag},t}$};
    \path[line] (innov.east) -- ++(1.0,0) |- (update.175) node[pos=0.38, lab] {$\mathbf{y}_{\mathrm{wifi},t},\ \mathbf{y}_{\mathrm{mag},t}$};
    \path[line] (prior.east) -- ++(6.15,0) |- (update.155) node[pos=0.73, lab] {$\mathbf{x}_t^-$};
    \path[auxline] (masks.east) -- ++(10.1,0) |- (update.190);
    \path[line] (update) -- (xout);

    % Recurrent state retained for the next update.
    \node[signal, minimum width=1.75cm] (hout) at (10.95,-4.00) {$\mathbf{h}_t$};
    \path[auxline] (gru.south east) -- ++(0.40,-0.45) -| (hout.north);
    \node[font=\scriptsize, text=black!60] at (10.95,-4.62) {retained for step $t+1$};
\end{tikzpicture}
}
\caption{Functional signal flow of DualKalmanNet at fusion step $t$. Blue rounded nodes denote input or retained signals, gray blocks deterministic state-space transformations, the purple block the learned recurrent gain generator, the orange block explicit relative magnetic-confidence weighting, and the green block the posterior correction. The PDR control $\mathbf{u}_t$ and learned Cartesian fixes $\mathbf{z}_{\mathrm{wifi},t}$ and $\mathbf{z}_{\mathrm{mag},t}$ enter as signals. The previous posterior $\mathbf{x}_{t-1}$ and PDR control form the prior $\mathbf{x}_t^-$; both absolute measurements form innovations against that same prior. These innovations and the auxiliary history/mask/confidence features form $\boldsymbol{\phi}_t$, from which the GRU predicts independent gain matrices. The posterior block applies the availability masks and $w_{\mathrm{mag},t}$ and emits the final state $\mathbf{x}_t$.}
\label{fig:arch_diagram}
\end{figure*}'''

text = text[:start] + new_figure + text[end:]

old_heading = '### [ ] J7. Redesign/recheck Fig. 1 as a signal-flow / functional-block diagram'
new_heading = '### [x] J7. Redesign/recheck Fig. 1 as a signal-flow / functional-block diagram'
if tracker.count(old_heading) != 1:
    raise RuntimeError(f'expected one open J7 heading, found {tracker.count(old_heading)}')
tracker = tracker.replace(old_heading, new_heading, 1)

status = '**Status:** still applicable even though the figure has since changed; redraw/re-audit rather than assuming current Fig. 1 is final.'
resolution = ('**Resolved in signal-flow figure pass:** Fig. 1 now treats $\\mathbf{u}_t$, '
              '$\\mathbf{z}_{\\mathrm{wifi},t}$, and $\\mathbf{z}_{\\mathrm{mag},t}$ as consistently styled input signals; '
              'shows $\\mathbf{x}_{t-1}$ entering the explicit prior-propagation function; defines both Cartesian innovations inside their functional block; '
              'shows the 13-D feature assembly, recurrent hidden state, GRU/linear gain generator, relative magnetic-confidence path, masks, and posterior correction; '
              'and places $\\mathbf{x}_t$ outside the update block as the estimator output. The diagram is aligned with the active 13-input DualKalmanNet implementation rather than the obsolete anomaly-fusion architecture.')
if tracker.count(status) != 1:
    raise RuntimeError(f'expected one J7 status line, found {tracker.count(status)}')
tracker = tracker.replace(status, resolution, 1)

# Structural assertions that mirror the review request.
for needle in [
    r'\mathbf{x}_{t-1}',
    r'\mathbf{u}_t',
    r'\mathbf{z}_{\mathrm{wifi},t}',
    r'\mathbf{z}_{\mathrm{mag},t}',
    r'\mathbf{x}_t^-=\mathbf{x}_{t-1}+\mathbf{u}_t',
    r'\mathbf{y}_{\mathrm{wifi},t}=\mathbf{z}_{\mathrm{wifi},t}-\mathbf{x}_t^-',
    r'\mathbf{y}_{\mathrm{mag},t}=\mathbf{z}_{\mathrm{mag},t}-\mathbf{x}_t^-',
    r'\boldsymbol{\phi}_t\in\mathbb{R}^{13}',
    r'\operatorname{GRUCell}',
    r'w_{\mathrm{mag},t}',
    r'\mathbf{x}_t$',
]:
    if needle not in new_figure:
        raise RuntimeError(f'missing required figure signal: {needle}')

paper_path.write_text(text, encoding='utf-8')
tracker_path.write_text(tracker, encoding='utf-8')
print('Applied J7 signal-flow figure pass.')
