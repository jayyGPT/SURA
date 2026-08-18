from pathlib import Path

paper = Path('paper/main.tex')
text = paper.read_text(encoding='utf-8')

old = r'''The GRU receives 13 scalar inputs per time step: the Wi-Fi innovation $\mathbf{y}_{wifi}\in\mathbb{R}^2$, magnetic-CNN innovation $\mathbf{y}_{mag}\in\mathbb{R}^2$, temporal difference of consecutive Wi-Fi fixes $\Delta\mathbf{z}_{wifi}\in\mathbb{R}^2$, PDR control $\mathbf{u}_t\in\mathbb{R}^2$, previous state update $\Delta\mathbf{x}_{t-1}\in\mathbb{R}^2$, two binary availability masks $m_{wifi},m_{mag}\in\{0,1\}$, and the scalar magnetic log-uncertainty score $\ell_{mag,t}$. Here,
\begin{equation}
\Delta\mathbf{z}_{wifi,t}=\mathbf{z}_{wifi,t}-\mathbf{z}_{wifi,t^-},
\label{eq:wifi_delta}
\end{equation}
where $t^-$ denotes the most recent time at which a Wi-Fi fix was available; the feature is set to zero when no new Wi-Fi measurement is present. It provides the recurrent gain network with a short-term consistency cue: the current Wi-Fi innovation can be interpreted jointly with how much the Wi-Fi estimate itself has moved since its previous update, the current PDR displacement, and the GRU history. The GRU outputs eight values, reshaped into independent $2\!\times\!2$ matrices $\mathbf{K}_{wifi}$ and $\mathbf{K}_{mag}$. The posterior update is'''

new = r'''For readability, the 13-dimensional GRU input is listed explicitly below. Define the previous posterior displacement and the masked, numerically clipped magnetic-confidence feature as
\begin{equation}
\begin{aligned}
\Delta\mathbf{x}_{t-1} &= \mathbf{x}_{t-1}-\mathbf{x}_{t-2},\\
c_{mag,t} &= m_{mag}\,\operatorname{clip}(\ell_{mag,t},-6,8).
\end{aligned}
\label{eq:gru_aux}
\end{equation}
Then the input groups are
\begin{center}
\footnotesize
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{GRU feature} & \textbf{Dim.} \\ \midrule
Wi-Fi innovation $\mathbf{y}_{wifi}$ & 2 \\
Magnetic innovation $\mathbf{y}_{mag}$ & 2 \\
Wi-Fi fix difference $\Delta\mathbf{z}_{wifi}$ & 2 \\
PDR control $\mathbf{u}_t$ & 2 \\
Previous posterior displacement $\Delta\mathbf{x}_{t-1}$ & 2 \\
Wi-Fi availability $m_{wifi}$ & 1 \\
Magnetic availability $m_{mag}$ & 1 \\
Magnetic confidence $c_{mag,t}$ & 1 \\ \midrule
\textbf{Total} & $\mathbf{13}$ \\ \bottomrule
\end{tabular}
\end{center}
Here,
\begin{equation}
\Delta\mathbf{z}_{wifi,t}=\mathbf{z}_{wifi,t}-\mathbf{z}_{wifi,t^-},
\label{eq:wifi_delta}
\end{equation}
where $t^-$ is the most recent time with an available Wi-Fi fix; the feature is zero when no new Wi-Fi measurement is present. It provides a short-term Wi-Fi consistency cue that the GRU interprets jointly with the current innovation, PDR displacement, and recurrent history. The GRU outputs eight values, reshaped into independent $2\!\times\!2$ matrices $\mathbf{K}_{wifi}$ and $\mathbf{K}_{mag}$. The posterior update is'''

if old in text:
    text = text.replace(old, new, 1)
elif 'For readability, the 13-dimensional GRU input is listed explicitly below.' not in text:
    raise SystemExit('expected GRU-input block not found')

# Normalize the first P9 draft, whose auxiliary equation was too wide for one IEEE column.
wide_aux = r'''\begin{equation}
\Delta\mathbf{x}_{t-1}=\mathbf{x}_{t-1}-\mathbf{x}_{t-2},\qquad
c_{mag,t}=m_{mag}\,\operatorname{clip}(\ell_{mag,t},-6,8).
\label{eq:gru_aux}
\end{equation}'''
split_aux = r'''\begin{equation}
\begin{aligned}
\Delta\mathbf{x}_{t-1} &= \mathbf{x}_{t-1}-\mathbf{x}_{t-2},\\
c_{mag,t} &= m_{mag}\,\operatorname{clip}(\ell_{mag,t},-6,8).
\end{aligned}
\label{eq:gru_aux}
\end{equation}'''
if wide_aux in text:
    text = text.replace(wide_aux, split_aux, 1)

paper.write_text(text, encoding='utf-8')

review = Path('paper/reviews/prof_read_ieee_comments_draft.md')
r = review.read_text(encoding='utf-8')

p9_open = '### [ ] P9. Present the 13 GRU inputs as a readable list (observation 5)'
p9_done = '### [x] P9. Present the 13 GRU inputs as a readable list (observation 5)'
if p9_open in r:
    r = r.replace(p9_open, p9_done, 1)
elif p9_done not in r:
    raise SystemExit('P9 marker missing')
needle9 = '`2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 = 13`.\n\nThis should make it much easier for a reviewer to verify the GRU input dimension against the implementation.\n'
note9 = '\n**Resolved:** replaced the long prose sentence with an eight-row feature/dimension list. The paper now also defines `Delta x_(t-1) = x_(t-1) - x_(t-2)` and the actual masked/clipped GRU confidence input `c_mag,t = m_mag clip(ell_mag,t,-6,8)`, matching the active implementation.\n'
if note9.strip() not in r:
    if needle9 not in r:
        raise SystemExit('P9 description missing')
    r = r.replace(needle9, needle9 + note9, 1)

p10_open = '### [ ] P10. Remove/standardize the em dash on page 2 (observation 8)'
p10_done = '### [x] P10. Remove/standardize the em dash on page 2 (observation 8)'
if p10_open in r:
    r = r.replace(p10_open, p10_done, 1)
elif p10_done not in r:
    raise SystemExit('P10 marker missing')
needle10 = 'Locate the page-2 final-line em dash and replace it with IEEE-consistent punctuation/wording if it is stylistically awkward. Perform this only during the final copy-edit pass so pagination changes do not make the page reference stale.\n'
note10 = '\n**Resolved after re-pagination:** the current seven-page draft was visually checked and no manuscript-authored em dash remains at the end of page 2. The visible em dashes in `Abstract--`/`Index Terms--` are generated by the IEEEtran template and are standard IEEE formatting, so they are intentionally retained.\n'
if note10.strip() not in r:
    if needle10 not in r:
        raise SystemExit('P10 description missing')
    r = r.replace(needle10, needle10 + note10, 1)

review.write_text(r, encoding='utf-8')
