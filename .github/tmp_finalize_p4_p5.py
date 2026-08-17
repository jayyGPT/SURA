from pathlib import Path

paper = Path('paper/main.tex')
text = paper.read_text(encoding='utf-8')
old = r'''Thus the active fusion path contains no separate scalar anomaly observation $A_{\text{obs}}$, anomaly map $A(\mathbf{x})$, or spatial anomaly gradient $\nabla A$; the same CNN output introduced in Section~II-C is the magnetic measurement consumed by KalmanNet.'''
new = r'''The magnetic CNN output $\mathbf{z}_{mag}$ therefore serves directly as the magnetic position measurement consumed by KalmanNet.'''
if old not in text:
    raise SystemExit('P4 paper sentence not found')
text = text.replace(old, new, 1)

old = r'''The GRU receives 13 scalar inputs per time step: the Wi-Fi innovation $\mathbf{y}_{wifi}\in\mathbb{R}^2$, magnetic-CNN innovation $\mathbf{y}_{mag}\in\mathbb{R}^2$, temporal difference of consecutive Wi-Fi fixes $\Delta\mathbf{z}_{wifi}\in\mathbb{R}^2$, PDR control $\mathbf{u}_t\in\mathbb{R}^2$, previous state update $\Delta\mathbf{x}_{t-1}\in\mathbb{R}^2$, two binary availability masks $m_{wifi},m_{mag}\in\{0,1\}$, and the scalar magnetic log-variance $\ell_{mag,t}$. The GRU outputs eight values, reshaped into independent $2\!\times\!2$ matrices $\mathbf{K}_{wifi}$ and $\mathbf{K}_{mag}$. The posterior update is'''
new = r'''The GRU receives 13 scalar inputs per time step: the Wi-Fi innovation $\mathbf{y}_{wifi}\in\mathbb{R}^2$, magnetic-CNN innovation $\mathbf{y}_{mag}\in\mathbb{R}^2$, temporal difference of consecutive Wi-Fi fixes $\Delta\mathbf{z}_{wifi}\in\mathbb{R}^2$, PDR control $\mathbf{u}_t\in\mathbb{R}^2$, previous state update $\Delta\mathbf{x}_{t-1}\in\mathbb{R}^2$, two binary availability masks $m_{wifi},m_{mag}\in\{0,1\}$, and the scalar magnetic log-variance $\ell_{mag,t}$. Here,
\begin{equation}
\Delta\mathbf{z}_{wifi,t}=\mathbf{z}_{wifi,t}-\mathbf{z}_{wifi,t^-},
\label{eq:wifi_delta}
\end{equation}
where $t^-$ denotes the most recent time at which a Wi-Fi fix was available; the feature is set to zero when no new Wi-Fi measurement is present. It provides the recurrent gain network with a short-term consistency cue: the current Wi-Fi innovation can be interpreted jointly with how much the Wi-Fi estimate itself has moved since its previous update, the current PDR displacement, and the GRU history. The GRU outputs eight values, reshaped into independent $2\!\times\!2$ matrices $\mathbf{K}_{wifi}$ and $\mathbf{K}_{mag}$. The posterior update is'''
if old not in text:
    raise SystemExit('P5 GRU paragraph not found')
text = text.replace(old, new, 1)
paper.write_text(text, encoding='utf-8')

# Update ablation README with the architecture decision and caveat about absolute values.
readme = Path('benchmarks/wifi_delta_ablation/README.md')
r = readme.read_text(encoding='utf-8')
r = r.replace(
    'Interpretation should be based on the paired differences above; this file does not automatically choose the final architecture.\n',
    '''## Decision for the current paper\n\nRetain `delta_z_wifi` in the final CNN DualKalmanNet. In the paired DualKalmanNet comparison, removing the feature increased mean error by `+0.0280 m` in full Wi-Fi, with a paired 95% CI of `[+0.0016, +0.0545] m`. In degraded Wi-Fi it increased mean error by `+0.0918 m`, although that interval `[-0.0224, +0.2060] m` includes zero. The Wi-Fi-only ablations were inconclusive.\n\nThe absolute with-delta values in this ablation are not replacements for the headline paper metrics. For this experiment the model-initialization and minibatch-shuffle seeds were reset before every paired training so the architectural comparison differed only in the two `delta_z_wifi` inputs. Interpret the paired differences, not cross-experiment changes in absolute error.\n'''
)
readme.write_text(r, encoding='utf-8')

# Mark P4/P5 resolved and record the evidence.
todo = Path('paper/reviews/prof_read_ieee_comments_draft.md')
t = todo.read_text(encoding='utf-8')
t = t.replace('### [ ] P4. Remove obsolete magnetic-anomaly notation', '### [x] P4. Remove obsolete magnetic-anomaly notation', 1)
t = t.replace('### [ ] P5. Explain why `Delta z_wifi` is a GRU input', '### [x] P5. Explain why `Delta z_wifi` is a GRU input', 1)
anchor = '### [ ] P6. Explain the CNN variance output precisely'
note = '''### P4-P5 implementation note\n\n- **P4:** removed the legacy anomaly notation from the active KalmanNet subsection. The paper now states only that the magnetic CNN position output is consumed directly by the fusion network.\n- **P5:** retained `Delta z_wifi` after a paired full-protocol ablation (`benchmarks/wifi_delta_ablation/`). For the CNN DualKalmanNet, removing the two-scalar feature worsened mean error by `+0.0280 m` in full Wi-Fi (paired 95% CI `[+0.0016, +0.0545] m`) and by `+0.0918 m` in degraded Wi-Fi (CI `[-0.0224, +0.2060] m`). Wi-Fi-only differences were inconclusive. The manuscript now defines the feature using the most recent available Wi-Fi fix and explains it only as a short-term consistency cue, without claiming that every large Wi-Fi delta is an outlier.\n\n'''
if anchor not in t:
    raise SystemExit('P6 anchor not found')
t = t.replace(anchor, note + anchor, 1)
todo.write_text(t, encoding='utf-8')
