from pathlib import Path

paper = Path("paper/main.tex")
text = paper.read_text(encoding="utf-8")

old_intro = r'''Figure~\ref{fig:cdf_mag} displays the continuous spatial matching error CDF for the 1D-CNN magnetic sequence model (MAE of 3.58~m), validating its capability as a structural spatial anchor during periods of Wi-Fi attenuation.'''
new_intro = r'''For a classical non-neural reference, we additionally evaluate K-nearest-neighbor (KNN) localization on the MagWi fingerprints~\cite{magwi}. Figure~\ref{fig:knn_fusion_cdfs}(a) reports a real held-out-device experiment: 726 fingerprint visits from A8, G7, and S8 are used for training and 90 S9+ visits are reserved for final testing. Hyperparameters are selected only from the training phones using leave-one-phone-out cross-validation. Wi-Fi-only KNN achieves 3.31~m mean error (1.57~m median), whereas magnetic-only KNN reaches 17.54~m and a simple Wi-Fi--magnetic feature concatenation reaches 7.46~m. The degradation of the naive magnetic variants highlights the strong cross-device sensitivity of static magnetic fingerprints and motivates learned, sequence-based magnetic matching rather than direct raw-feature concatenation.'''
if old_intro not in text:
    raise SystemExit("environment-model KNN insertion point not found")
text = text.replace(old_intro, new_intro, 1)

old_table = r'''\begin{table}[htbp]
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
new_table = r'''\begin{table}[htbp]
\caption{Fusion Comparison on the Matched 60-Walk Protocol}
\label{tab:fusion_comparison}
\centering
\footnotesize
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}p{0.17\columnwidth}p{0.35\columnwidth}p{0.25\columnwidth}p{0.12\columnwidth}@{}}
\toprule
\textbf{Regime} & \textbf{Model} & \textbf{Mean Err.} & \textbf{Med.} \\ \midrule
Full Wi-Fi (1 Hz)
& WiFi-only KalmanNet & \textbf{0.473 $\pm$ 0.035 m} & 0.449 m \\
& WiFi+Mag KNN (non-temp.) & 0.802 $\pm$ 0.038 m & 0.772 m \\
& CNN DualKalmanNet & 0.506 $\pm$ 0.056 m & 0.440 m \\
& CNN Dual + rel. variance & 0.494 $\pm$ 0.046 m & \textbf{0.437 m} \\ \midrule
Degraded Wi-Fi (5 s, 40\% drop)
& WiFi-only KalmanNet & 1.533 $\pm$ 0.193 m & 1.392 m \\
& WiFi+Mag KNN (non-temp.) & 2.606 $\pm$ 0.184 m & 2.560 m \\
& CNN DualKalmanNet & 1.171 $\pm$ 0.139 m & \textbf{1.042 m} \\
& \textbf{CNN Dual + rel. variance} & \textbf{1.154 $\pm$ 0.129 m} & 1.113 m \\ \bottomrule
\end{tabular}
\end{table}'''
if old_table not in text:
    raise SystemExit("fusion table not found")
text = text.replace(old_table, new_table, 1)

old_figure = r'''\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\columnwidth]{figures/mag_cdf_cropped.png}
\caption{Standalone 1D-CNN magnetic sequence matcher error CDF (MAE 3.58~m).}
\label{fig:cdf_mag}
\end{figure}'''
new_figure = r'''\begin{figure*}[htbp]
\centering
\begin{subfigure}[t]{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../benchmarks/knn/current_results/static_heldout_device/cdf.png}
  \caption{Real static fingerprints with S9+ held out.}
  \label{fig:knn_static}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../benchmarks/knn/current_results/trajectory_protocol/full/cdf.png}
  \caption{Matched trajectory protocol, full Wi-Fi (1~Hz).}
  \label{fig:knn_full}
\end{subfigure}

\vspace{0.15cm}
\begin{subfigure}[t]{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../benchmarks/knn/current_results/trajectory_protocol/degraded/cdf.png}
  \caption{Matched trajectory protocol, degraded Wi-Fi.}
  \label{fig:knn_degraded}
\end{subfigure}
\caption{KNN baselines and current fusion CDFs. Panel (a) is a separate real static-fingerprint, held-out-device experiment and is not numerically comparable to the synthetic walk errors in (b)--(c). Panels (b)--(c) use exactly the same 250-training/60-test synthetic trajectory generation as KalmanNet. The non-temporal KNN receives the contemporaneous Wi-Fi heatmap fix, magnetic-CNN fix and log-variance, and availability masks, but no PDR motion or recurrent history. Axes, ticks, and legends are regenerated at paper-readable sizes.}
\label{fig:knn_fusion_cdfs}
\end{figure*}'''
if old_figure not in text:
    raise SystemExit("old standalone CDF figure not found")
text = text.replace(old_figure, new_figure, 1)

old_results = r'''Under full 1~Hz Wi-Fi, the Wi-Fi-only KalmanNet retains the lowest mean error at 0.473~m, while the variance-weighted CNN fusion obtains 0.494~m. However, its median error improves slightly from 0.449~m to 0.437~m, and relative variance weighting improves the CNN-fusion mean from 0.506~m to 0.494~m. The benefit becomes substantially clearer when Wi-Fi is degraded. With one Wi-Fi update every 5~s and 40\% AP dropout, directly fusing the CNN output reduces the mean error from 1.533~m to 1.171~m (23.6\%). Adding relative magnetic-variance weighting further reduces the mean to 1.154~m, corresponding to a 24.7\% reduction over Wi-Fi-only KalmanNet. The weighting particularly suppresses poor magnetic fixes: the degraded-regime P90 decreases from 2.064~m for the unweighted CNN fusion to 1.612~m with relative variance weighting. These results show that the magnetic CNN is most valuable as an absolute spatial anchor during sparse Wi-Fi periods, while its predicted uncertainty helps limit the high-error tail.'''
new_results = r'''For the matched trajectory comparison in Fig.~\ref{fig:knn_fusion_cdfs}(b)--(c), KNN is intentionally non-temporal: it receives only the contemporaneous Wi-Fi heatmap position, magnetic-CNN position and log-variance, and the two availability masks. Its value of $K$ is selected by five-fold grouped cross-validation in which all bins from a trajectory remain in the same fold. Under full 1~Hz Wi-Fi, the KNN baseline obtains 0.802~m mean error, compared with 0.473~m for Wi-Fi-only KalmanNet and 0.494~m for the variance-weighted CNN fusion. The latter also improves median error slightly from 0.449~m to 0.437~m. Under degraded Wi-Fi, the gap is larger: non-temporal KNN reaches 2.606~m, Wi-Fi-only KalmanNet reaches 1.533~m, and the variance-weighted DualKalmanNet reaches 1.154~m. Direct CNN fusion without the explicit variance weight reaches 1.171~m, so relative magnetic-variance weighting yields a 24.7\% reduction over Wi-Fi-only KalmanNet and reduces the degraded-regime P90 from 2.064~m for unweighted CNN fusion to 1.612~m. The separation from KNN in Fig.~\ref{fig:knn_fusion_cdfs}(c) indicates that the gain does not arise merely from having both absolute sensor estimates available; PDR propagation and learned temporal, context-dependent fusion are important in sparse-Wi-Fi operation.'''
if old_results not in text:
    raise SystemExit("results discussion not found")
text = text.replace(old_results, new_results, 1)
paper.write_text(text, encoding="utf-8")

review = Path("paper/reviews/professor_feedback.md")
r = review.read_text(encoding="utf-8")
old_r7 = '- [ ] R7: Updated all text references from "top/bottom" to `fig:cdf_full` and `fig:cdf_degraded` subfigure labels (KNN baseline plot regeneration pending)'
new_r7 = '- [x] R7: Regenerated the KNN/fusion CDF panels with paper-readable axes, ticks, and legends. Added (i) a real held-out-S9+ classical fingerprint KNN panel and (ii) matched-protocol non-temporal Wi-Fi+magnetic KNN curves for full/degraded Wi-Fi. Source note: the MagWi paper is used as the dataset reference; no explicit KNN localization specification/result was located in the paper, so these are our reproducible KNN baselines rather than numbers attributed to Ashraf et al.'
if old_r7 not in r:
    raise SystemExit("R7 status line not found")
r = r.replace(old_r7, new_r7, 1)
review.write_text(r, encoding="utf-8")

knn_readme = Path("benchmarks/knn/README.md")
knn_readme.write_text(r'''# KNN baselines

The current reproducible Wi-Fi + magnetic KNN baseline is `wifi_mag_knn.py`. Older proof-of-concept scripts remain under `src/` and their old plots remain under `legacy_results/`; those historical numbers are not used in the paper.

## Current protocols

### 1. Real static held-out-device fingerprinting

Uses `data/processed/fingerprint_db/it_engineering/` directly.

- train phones: A8, G7, S8
- held-out final test phone: S9+
- train/test visits: 726 / 90
- Wi-Fi: canonical 250-AP RSSI encoding
- magnetic: rotation-invariant `magN`, `magV`, `magH`, and `dip` node statistics
- K and Wi-Fi/magnetic block weight: selected using leave-one-phone-out grouped cross-validation on the training phones only

Final S9+ mean errors:

| Variant | K | Mean | Median | P90 |
|---|---:|---:|---:|---:|
| Wi-Fi KNN | 7 | 3.310 m | 1.571 m | 6.352 m |
| Magnetic KNN | 20 | 17.538 m | 10.775 m | 42.698 m |
| Wi-Fi + magnetic KNN | 3 | 7.459 m | 4.359 m | 14.400 m |

The hybrid selects a 0.75 Wi-Fi / 0.25 magnetic distance weighting. Its degradation relative to Wi-Fi-only KNN is evidence that simple static magnetic concatenation does not solve cross-device magnetic heterogeneity.

### 2. Matched synthetic trajectory protocol

Uses the exact 250-training/60-test, 160-bin trajectory generation used by `train/kalmannet_wifiheatmap_magneticCNN_pdr.py`. This KNN is deliberately non-temporal: its features are the current Wi-Fi heatmap fix, magnetic-CNN fix, magnetic log-variance, and availability masks. It receives no PDR motion and no recurrent history. K is selected using GroupKFold with whole trajectories kept together.

| Regime | Selected K | Wi-Fi+Mag KNN | Wi-Fi-only KalmanNet | CNN Dual + relative variance |
|---|---:|---:|---:|---:|
| Full Wi-Fi (1 Hz) | 5 | 0.802 m | 0.473 m | 0.494 m |
| Degraded Wi-Fi (5 s, 40% AP drop) | 20 | 2.606 m | 1.533 m | 1.154 m |

## Outputs

Reviewed outputs are under `current_results/`:

- `summary.json` and protocol-specific `metrics.json`
- real static predictions CSV
- trajectory prediction/error NPZ files
- large-font CDF plots used by the manuscript

Run both protocols with:

```bash
python benchmarks/knn/wifi_mag_knn.py --protocol both
```

### Attribution note

The MagWi paper is a benchmark-dataset/characterization paper. We did not locate an explicit KNN localization method or KNN result in that paper, so the values above are **our reproducible classical baselines on MagWi data**, not results copied from or attributed to the MagWi authors.
''', encoding="utf-8")

results_readme = Path("benchmarks/knn/current_results/README.md")
results_readme.write_text(r'''# Current Wi-Fi + magnetic KNN results

These outputs were generated by `../wifi_mag_knn.py` and are the KNN results used for the current R7 manuscript comparison.

The real static protocol and the synthetic trajectory protocol are deliberately separate. Do not compare their absolute error values as though they were the same test set.

## Real static held-out S9+

- 726 training visits from A8/G7/S8; 90 test visits from S9+
- Wi-Fi KNN: 3.310 m mean
- Magnetic KNN: 17.538 m mean
- Wi-Fi + magnetic KNN: 7.459 m mean

## Matched 250/60 trajectory protocol

- Full Wi-Fi: KNN 0.802 m; Wi-Fi-only KalmanNet 0.473 m; weighted CNN DualKalmanNet 0.494 m
- Degraded Wi-Fi: KNN 2.606 m; Wi-Fi-only KalmanNet 1.533 m; weighted CNN DualKalmanNet 1.154 m

The trajectory KNN uses no PDR or temporal history. Its purpose is to test whether simply having the current Wi-Fi and magnetic learned measurements is enough to match the learned temporal filter.
''', encoding="utf-8")
