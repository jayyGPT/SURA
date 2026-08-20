from pathlib import Path

path = Path('paper/main.tex')
text = path.read_text(encoding='utf-8')
label = r'\label{fig:arch_diagram}'
label_pos = text.find(label)
if label_pos < 0:
    raise RuntimeError('architecture figure label not found')
start = text.rfind(r'\begin{figure*}', 0, label_pos)
end = text.find(r'\end{figure*}', label_pos)
if start < 0 or end < 0:
    raise RuntimeError('architecture figure* boundaries not found')
end += len(r'\end{figure*}')

fig = r'''\begin{figure*}[t]
\centering
\resizebox{0.97\textwidth}{!}{
\begin{tikzpicture}[
    signal/.style={rectangle, rounded corners=7pt, draw=blue!70!black, thick, fill=blue!8, minimum height=0.72cm, minimum width=2.05cm, text centered, font=\scriptsize},
    block/.style={rectangle, rounded corners=2pt, draw=black!70, thick, fill=gray!8, minimum height=1.05cm, text width=3.65cm, text centered, font=\scriptsize},
    learned/.style={block, fill=purple!10, text width=3.7cm},
    confidence/.style={block, fill=orange!12, text width=3.7cm},
    correction/.style={block, fill=green!10, text width=3.8cm},
    line/.style={draw, -latex, thick, color=black!80},
    auxline/.style={draw, -latex, thick, dashed, color=black!55},
    lab/.style={font=\scriptsize, fill=white, inner sep=1pt}
]
    % External/current-step signals.
    \node[signal] (xprev) at (0,3.55) {$\mathbf{x}_{t-1}$};
    \node[signal] (motion) at (0,2.55) {$\mathbf{u}_t$};
    \node[signal] (wifi) at (0,1.15) {$\mathbf{z}_{\mathrm{wifi},t}$};
    \node[signal] (mag) at (0,0.15) {$\mathbf{z}_{\mathrm{mag},t}$};
    \node[signal] (ell) at (0,-1.25) {$\ell_{\mathrm{mag},t}$};
    \node[signal, minimum width=2.65cm] (masks) at (0,-2.25) {$m_{\mathrm{wifi},t},\ m_{\mathrm{mag},t}$};
    \node[signal, minimum width=3.35cm] (history) at (0,-3.25) {$\Delta\mathbf{z}_{\mathrm{wifi},t},\ \Delta\mathbf{x}_{t-1}$};

    % Top lane: prior state propagation and posterior output.
    \node[block] (prior) at (3.25,3.05) {Prior propagation\\[-1pt]$\mathbf{x}_t^-=\mathbf{x}_{t-1}+\mathbf{u}_t$};
    \node[correction] (update) at (13.35,2.85) {Posterior correction\\[-1pt]Eq.~(\ref{eq:kn_correct})};
    \node[signal, draw=green!55!black, fill=green!8, minimum width=1.85cm] (xout) at (16.25,2.85) {$\mathbf{x}_t$};

    % Middle lane: innovations -> recurrent features -> gains.
    \node[block, text width=4.55cm] (innov) at (4.15,0.70) {Innovation formation\\[-1pt]
      $\mathbf{y}_{\mathrm{wifi},t}=\mathbf{z}_{\mathrm{wifi},t}-\mathbf{x}_t^-$\\[-1pt]
      $\mathbf{y}_{\mathrm{mag},t}=\mathbf{z}_{\mathrm{mag},t}-\mathbf{x}_t^-$};
    \node[block, text width=3.65cm] (features) at (7.85,0.70) {13-D feature assembly\\[-1pt]$\boldsymbol{\phi}_t\in\mathbb{R}^{13}$};
    \node[learned] (gru) at (11.05,0.70) {GRUCell + linear gain head\\[-1pt]
      $(\boldsymbol{\phi}_t,\mathbf{h}_{t-1})\mapsto$\\[-1pt]
      $\mathbf{K}_{\mathrm{wifi},t},\ \mathbf{K}_{\mathrm{mag},t},\ \mathbf{h}_t$};

    % Bottom lane: auxiliary recurrent inputs and explicit confidence weighting.
    \node[block, text width=4.5cm] (aux) at (4.15,-1.75) {Auxiliary recurrent features\\[-1pt]
      $\mathbf{u}_t,\ \Delta\mathbf{z}_{\mathrm{wifi},t},\ \Delta\mathbf{x}_{t-1},$\\[-1pt]
      $m_{\mathrm{wifi},t},\ m_{\mathrm{mag},t},\ c_{\mathrm{mag},t}$};
    \node[confidence] (conf) at (8.25,-2.55) {Relative magnetic confidence\\[-1pt]
      $w_{\mathrm{mag},t}=\bigl[1+e^{\ell_{\mathrm{mag},t}-\ell_{\mathrm{ref}}}\bigr]^{-1}$};
    \node[signal] (hprev) at (11.05,-2.55) {$\mathbf{h}_{t-1}$};

    % State and measurement flow.
    \path[line] (xprev) -- (prior);
    \path[line] (motion) -- (prior);
    \path[line] (prior.east) -- ++(7.0,0) -- ++(0, -0.20) -- (update.west) node[pos=0.48, lab] {$\mathbf{x}_t^-$};
    \path[line] (prior.south) -- ++(0,-0.65) -| (innov.north) node[pos=0.30, lab] {$\mathbf{x}_t^-$};
    \path[line] (wifi) -- (innov.west);
    \path[line] (mag.east) -- ++(1.05,0) |- (innov.205);
    \path[line] (innov) -- (features) node[midway, above, lab] {$\mathbf{y}_{\mathrm{wifi},t},\ \mathbf{y}_{\mathrm{mag},t}$};
    \path[line] (features) -- (gru) node[midway, above, lab] {$\boldsymbol{\phi}_t$};
    \path[line] (gru.north) -- ++(0,1.00) -| (update.south) node[pos=0.42, lab] {$\mathbf{K}_{\mathrm{wifi},t},\ \mathbf{K}_{\mathrm{mag},t}$};
    \path[line] (innov.north east) -- ++(0.65,0.55) -- ++(5.85,0) |- (update.195) node[pos=0.48, lab] {$\mathbf{y}_{\mathrm{wifi},t},\ \mathbf{y}_{\mathrm{mag},t}$};

    % Auxiliary feature assembly.
    \path[auxline] (motion.east) -- ++(1.25,0) |- (aux.155);
    \path[auxline] (ell.east) -- ++(1.15,0) |- (aux.175) node[pos=0.58, lab] {$c_{\mathrm{mag},t}$};
    \path[auxline] (masks) -- (aux.west);
    \path[auxline] (history.east) -- ++(1.15,0) |- (aux.205);
    \path[auxline] (aux.east) -- ++(1.15,0) |- (features.south) node[pos=0.55, lab] {auxiliary features};

    % Confidence and recurrent hidden-state paths.
    \path[line] (ell.east) -- ++(5.05,0) |- (conf.west);
    \path[line] (conf.east) -- ++(3.00,0) |- (update.south east) node[pos=0.48, lab] {$w_{\mathrm{mag},t}$};
    \path[auxline] (hprev.north) -- (gru.south);
    \path[auxline] (masks.east) -- ++(11.25,0) |- (update.south west) node[pos=0.78, lab] {availability masks};

    \path[line] (update) -- (xout);
    \node[font=\scriptsize, text=black!60] at (11.05,-3.15) {$\mathbf{h}_t$ is retained for fusion step $t+1$};
\end{tikzpicture}
}
\caption{Functional signal flow of DualKalmanNet at fusion step $t$. Blue rounded nodes are input or retained signals; gray blocks are deterministic transformations; the purple block is the learned recurrent gain generator; the orange block computes explicit relative magnetic confidence; and the green block performs the posterior correction. The previous state and PDR control form $\mathbf{x}_t^-$, while the Wi-Fi and magnetic fixes form separate innovations against that prior. The innovations and auxiliary recurrent features form $\boldsymbol{\phi}_t$, the GRU produces independent $2\times2$ gain matrices, and the posterior correction applies the gains together with availability masks and $w_{\mathrm{mag},t}$ before emitting $\mathbf{x}_t$. Dashed arrows denote auxiliary or recurrent paths.}
\label{fig:arch_diagram}
\end{figure*}'''

text = text[:start] + fig + text[end:]
path.write_text(text, encoding='utf-8')
print('Applied cleaner J7 signal-flow layout.')
