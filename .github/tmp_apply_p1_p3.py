from pathlib import Path
import re

# -----------------------------------------------------------------------------
# Active fusion experiment: provenance + trajectory split validation.
# -----------------------------------------------------------------------------
path = Path('train/kalmannet_wifiheatmap_magneticCNN_pdr.py')
text = path.read_text(encoding='utf-8')

text = text.replace('import argparse\nimport copy\nimport json\n', 'import argparse\nimport copy\nimport hashlib\nimport json\n', 1)
text = text.replace('HEADING_WHITE_NOISE_RAD = np.deg2rad(8.8)\nHEADING_DRIFT_STD_RAD = np.deg2rad(0.5) / np.sqrt(SAMPLING_HZ)\nSTEP_LENGTH_M = 0.65\n', 'HEADING_WHITE_NOISE_STD_RAD = np.deg2rad(8.8)\nHEADING_DRIFT_STD_RAD = np.deg2rad(0.5) / np.sqrt(SAMPLING_HZ)\nNOMINAL_STEP_LENGTH_M = 0.65\nFUSION_TRAIN_SEED = 1\nFUSION_TEST_SEED = 2\n', 1)
text = text.replace('    corridor: CorridorGraph\n', '    corridor: CorridorGraph\n    survey_phones: tuple[str, ...]\n', 1)
text = text.replace('        + rng.normal(0.0, HEADING_WHITE_NOISE_RAD, frames)\n', '        + rng.normal(0.0, HEADING_WHITE_NOISE_STD_RAD, frames)\n', 1)
text = text.replace('            controls[index] = STEP_LENGTH_M * np.array(\n', '            controls[index] = NOMINAL_STEP_LENGTH_M * np.array(\n', 1)

old_setup = '''    pool_nodes, pool = build_wifi_pool(database, ap_columns)\n    return Environment(\n        wifi_model=wifi_model,\n        wifi_grid=wifi_grid,\n        wifi_coordinates=wifi_coordinates,\n        wifi_pool_nodes=pool_nodes,\n        wifi_pool=pool,\n        access_points=ap_columns,\n        magnetic_model=magnetic_model,\n        magnetic_window=magnetic_window,\n        magnetic_map=magnetic_map,\n        corridor=corridor,\n    )\n'''
new_setup = '''    pool_nodes, pool = build_wifi_pool(database, ap_columns)\n    survey_phones = tuple(sorted(str(value) for value in database.frame["phone"].dropna().unique()))\n    return Environment(\n        wifi_model=wifi_model,\n        wifi_grid=wifi_grid,\n        wifi_coordinates=wifi_coordinates,\n        wifi_pool_nodes=pool_nodes,\n        wifi_pool=pool,\n        access_points=ap_columns,\n        magnetic_model=magnetic_model,\n        magnetic_window=magnetic_window,\n        magnetic_map=magnetic_map,\n        corridor=corridor,\n        survey_phones=survey_phones,\n    )\n'''
if old_setup not in text:
    raise SystemExit('setup_environment insertion point not found')
text = text.replace(old_setup, new_setup, 1)

marker = '''    if len(rows) != walks:\n        raise RuntimeError(f"generated only {len(rows)}/{walks} requested walks")\n    return tuple(np.stack([row[column] for row in rows]) for column in range(8))\n\n\n# ---------------------------------------------------------------------------\n# Fusion training/evaluation\n# ---------------------------------------------------------------------------\n'''
replacement = '''    if len(rows) != walks:\n        raise RuntimeError(f"generated only {len(rows)}/{walks} requested walks")\n    return tuple(np.stack([row[column] for row in rows]) for column in range(8))\n\n\ndef _trajectory_signature(target: np.ndarray) -> str:\n    """Stable signature for one binned ground-truth trajectory."""\n    rounded = np.round(np.asarray(target, dtype=np.float32), 4)\n    return hashlib.sha256(rounded.tobytes()).hexdigest()\n\n\ndef validate_trajectory_split(\n    training: tuple[np.ndarray, ...],\n    testing: tuple[np.ndarray, ...],\n) -> dict[str, object]:\n    """Fail if an identical generated target trajectory occurs in train and test."""\n    train_signatures = {_trajectory_signature(target) for target in training[6]}\n    test_signatures = {_trajectory_signature(target) for target in testing[6]}\n    overlap = sorted(train_signatures.intersection(test_signatures))\n    if overlap:\n        raise RuntimeError(\n            f"trajectory leakage detected: {len(overlap)} identical train/test trajectory signature(s)"\n        )\n    return {\n        "train_seed": FUSION_TRAIN_SEED,\n        "test_seed": FUSION_TEST_SEED,\n        "train_walks": int(len(training[6])),\n        "test_walks": int(len(testing[6])),\n        "exact_target_trajectory_overlap_count": 0,\n        "signature_rounding_m": 1e-4,\n    }\n\n\n# ---------------------------------------------------------------------------\n# Fusion training/evaluation\n# ---------------------------------------------------------------------------\n'''
if marker not in text:
    raise SystemExit('make_dataset insertion point not found')
text = text.replace(marker, replacement, 1)

text = text.replace('            seed=1,\n            env=env,', '            seed=FUSION_TRAIN_SEED,\n            env=env,', 1)
text = text.replace('            seed=2,\n            env=env,', '            seed=FUSION_TEST_SEED,\n            env=env,', 1)
needle = '        print("  magnetic CNN measurement quality:", magnetic_measurement_summary(testing))\n'
insert = '''        split_audit = validate_trajectory_split(training, testing)\n        print("  trajectory split audit:", split_audit)\n        print("  magnetic CNN measurement quality:", magnetic_measurement_summary(testing))\n'''
if needle not in text:
    raise SystemExit('run_experiment audit insertion point not found')
text = text.replace(needle, insert, 1)

needle = '''            "magnetic_measurement": magnetic_measurement_summary(testing),\n            "magnetic_reference_log_variance_training": magnetic_reference_log_variance,\n'''
insert = '''            "magnetic_measurement": magnetic_measurement_summary(testing),\n            "trajectory_split_audit": split_audit,\n            "simulation_protocol": {\n                "heading_source": "true_path_tangent_plus_random_walk_and_white_noise",\n                "heading_white_noise_std_deg": float(np.rad2deg(HEADING_WHITE_NOISE_STD_RAD)),\n                "heading_random_walk_step_std_deg": float(np.rad2deg(HEADING_DRIFT_STD_RAD)),\n                "nominal_step_length_m": NOMINAL_STEP_LENGTH_M,\n                "step_frequency_range_hz": [STEP_FREQ_MIN_HZ, STEP_FREQ_MAX_HZ],\n                "speed_range_mps": [SPEED_MIN_MPS, SPEED_MAX_MPS],\n                "survey_phones_used_to_construct_environment": list(env.survey_phones),\n                "fusion_device_generalization_claim": False,\n            },\n            "magnetic_reference_log_variance_training": magnetic_reference_log_variance,\n'''
if needle not in text:
    raise SystemExit('report provenance insertion point not found')
text = text.replace(needle, insert, 1)

path.write_text(text, encoding='utf-8')

# -----------------------------------------------------------------------------
# KNN matched protocol: use same explicit seeds and split audit.
# -----------------------------------------------------------------------------
path = Path('benchmarks/knn/wifi_mag_knn.py')
text = path.read_text(encoding='utf-8')
text = text.replace('    FUSION_HIDDEN_SIZE,\n', '    FUSION_HIDDEN_SIZE,\n    FUSION_TEST_SEED,\n    FUSION_TRAIN_SEED,\n', 1)
text = text.replace('    train_filter,\n)', '    train_filter,\n    validate_trajectory_split,\n)', 1)
text = text.replace('            train_walks, seed=1, env=env, device=device,\n', '            train_walks, seed=FUSION_TRAIN_SEED, env=env, device=device,\n', 1)
text = text.replace('            test_walks, seed=2, env=env, device=device,\n', '            test_walks, seed=FUSION_TEST_SEED, env=env, device=device,\n', 1)
needle = '''        testing = make_dataset(\n            test_walks, seed=FUSION_TEST_SEED, env=env, device=device,\n            wifi_period_s=wifi_period, ap_dropout=dropout, bins=bins,\n        )\n\n        print("  tuning/fitting non-temporal Wi-Fi + magnetic KNN")\n'''
insert = '''        testing = make_dataset(\n            test_walks, seed=FUSION_TEST_SEED, env=env, device=device,\n            wifi_period_s=wifi_period, ap_dropout=dropout, bins=bins,\n        )\n        split_audit = validate_trajectory_split(training, testing)\n        print("  trajectory split audit:", split_audit)\n\n        print("  tuning/fitting non-temporal Wi-Fi + magnetic KNN")\n'''
if needle not in text:
    raise SystemExit('KNN split audit insertion point not found')
text = text.replace(needle, insert, 1)
needle = '''            "ap_dropout": dropout,\n            "wifi_only_kalmannet": summarize(baseline_errors),\n'''
insert = '''            "ap_dropout": dropout,\n            "trajectory_split_audit": split_audit,\n            "simulation_protocol": {\n                "nominal_step_length_m": 0.65,\n                "heading_source": "true_path_tangent_plus_random_walk_and_white_noise",\n                "survey_phones_used_to_construct_environment": list(env.survey_phones),\n                "fusion_device_generalization_claim": False,\n            },\n            "wifi_only_kalmannet": summarize(baseline_errors),\n'''
if needle not in text:
    raise SystemExit('KNN report insertion point not found')
text = text.replace(needle, insert, 1)
path.write_text(text, encoding='utf-8')

# -----------------------------------------------------------------------------
# Manuscript: replace PDR calibration story and expand synthetic protocol.
# -----------------------------------------------------------------------------
paper = Path('paper/main.tex')
text = paper.read_text(encoding='utf-8')

pdr_block = r'''\subsection{Causal Pedestrian Dead Reckoning (PDR)}

The Pedestrian Dead Reckoning (PDR) module converts a causal acceleration stream and a heading observation into relative planar displacement at the IMU rate ($f_s=16.7$~Hz). Let $a_t=\|\mathbf{a}_t\|$ denote acceleration magnitude. A first-order exponential moving average (EMA) tracks the slowly varying gravitational baseline,
\begin{equation}
\bar{a}_t = \alpha\bar{a}_{t-1} + (1-\alpha)a_t, \qquad \alpha=0.98,
\label{eq:lpf}
\end{equation}
with $\bar{a}_0=9.81$~m/s$^2$. The high-pass residual is
\begin{equation}
\tilde{a}_t=a_t-\bar{a}_t,
\label{eq:hpf}
\end{equation}
and a step is declared when
\begin{equation}
\tilde{a}_t>\tau \quad\text{and}\quad (t-t_{\mathrm{last}})>\Delta_r,
\label{eq:step_trigger}
\end{equation}
where $\tau=0.6$~m/s$^2$ and $\Delta_r=\lfloor0.3f_s\rfloor=5$ frames. This detector is causal: its state at time $t$ depends only on acceleration samples observed up to $t$.

For a detected step, the PDR control is
\begin{equation}
\mathbf{u}_t=
\begin{bmatrix}
L_s\cos\hat{\theta}_t\\
L_s\sin\hat{\theta}_t
\end{bmatrix},
\label{eq:pdr}
\end{equation}
where $\hat{\theta}_t$ denotes the heading observation supplied to the PDR module and $L_s$ is the assumed stride length. The fusion experiments in this paper use a fixed nominal value $L_s=0.65$~m for every generated walk; no ground-truth path length from either the fusion-training or fusion-test trajectories is used to adapt $L_s$. Likewise, the evaluated fusion protocol does not use the MagWi \texttt{Orn\_z} field and does not estimate a ground-truth-calibrated heading offset $\phi_h$. Instead, the synthetic measurement generator in Section~\ref{sec:setup} produces $\hat{\theta}_t$ by corrupting the known path tangent with drift and white noise. Ground truth is therefore used by the simulator to generate a noisy sensor observation, but the PDR/KalmanNet estimator receives only that noisy observation. In a physical deployment, $\hat{\theta}_t$ would be supplied by a causal device-heading estimator; evaluation of such a raw-sensor heading estimator is outside the present fusion experiment.

\begin{figure}[htbp]
\centering
\resizebox{0.82\columnwidth}{!}{
\begin{tikzpicture}[>=latex, thick, font=\small]
  \draw[->, gray!60, thin] (-0.3,0) -- (4.2,0) node[right, black] {$x$ (map East)};
  \draw[->, gray!60, thin] (0,-0.3) -- (0,3.4) node[above, black] {$y$ (map North)};
  \filldraw[black] (0,0) circle (2.2pt);
  \node[below left, yshift=-2pt] at (0,0) {foot at $t\!-\!1$};
  \draw[->, very thick, blue!70!black] (0,0) -- ({3.2*0.866},{3.2*0.5})
        node[pos=0.56, above left=2pt, blue!70!black] {$\mathbf{u}_t$, $\|\mathbf{u}_t\|=L_s$};
  \filldraw[blue!70!black] ({3.2*0.866},{3.2*0.5}) circle (2.2pt);
  \node[right=4pt, blue!70!black] at ({3.2*0.866},{3.2*0.5}) {foot at $t$};
  \draw[gray!50, dashed, thin] (0,0) -- (3.8,0);
  \draw[->, blue!60!black, thin] (1.25,0) arc[start angle=0, end angle=30, radius=1.25];
  \node[blue!70!black] at (1.48,0.38) {$\hat{\theta}_t$};
  \draw[gray!55, dashed, thin] ({3.2*0.866},{3.2*0.5}) -- ({3.2*0.866},0);
  \draw[<->, gray!70, thin] (0,-0.55) -- ({3.2*0.866},-0.55)
        node[midway, below, black, font=\footnotesize] {$L_s\cos\hat{\theta}_t$};
  \draw[gray!55, dashed, thin] ({3.2*0.866},{3.2*0.5}) -- (0,{3.2*0.5});
  \draw[<->, gray!70, thin] (-0.55,0) -- (-0.55,{3.2*0.5})
        node[midway, left, black, font=\footnotesize] {$L_s\sin\hat{\theta}_t$};
\end{tikzpicture}
}
\caption{Geometry of the PDR control used by the estimator. A detected step of nominal length $L_s$ is projected along the currently available heading observation $\hat{\theta}_t$. In the synthetic fusion evaluation, $\hat{\theta}_t$ is a noisy simulated heading measurement rather than a ground-truth-calibrated \texttt{Orn\_z} signal.}
\label{fig:pdr_geometry}
\end{figure}

'''
text, count = re.subn(
    r'\\subsection\{Causal Pedestrian Dead Reckoning \(PDR\)\}.*?(?=\\subsection\{Dual-Innovation KalmanNet Fusion\})',
    lambda _: pdr_block,
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f'PDR subsection replacement count={count}')

setup_intro = r'''\section{Experimental Setup and Dataset}\label{sec:setup}

We evaluate the proposed fusion architecture in the IT Engineering building of the MagWi Benchmark Dataset~\cite{magwi}. The surveyed fingerprints define the environment model used to generate Wi-Fi and magnetic measurements, while temporal fusion is evaluated on newly generated map-constrained trajectories. Consequently, the fusion experiment tests generalization to unseen trajectories within a fixed surveyed environment; it is not, by itself, a held-out-smartphone experiment.

\subsection{Survey-Derived Synthetic Trajectory Evaluation}

The final fusion benchmark is synthetic because the continuous MagWi recordings do not provide the per-frame trajectory ground truth required for supervised KalmanNet training and controlled temporal evaluation. The simulator has access to the generated ground-truth state in order to produce sensor observations; the estimator does not. Specifically:

\begin{itemize}
    \item \textbf{Survey environment:} An $\varepsilon$-graph ($\varepsilon=1.6$~m) is built from surveyed IT Engineering reference-node coordinates. The processed survey database is also used to construct a pool of real Wi-Fi scans and an interpolated four-channel magnetic map. These environment resources include the available surveyed devices; therefore the fusion results are not claimed as an unseen-device test.
    \item \textbf{Trajectory generation:} Random endpoint pairs are sampled on the connected corridor graph and joined by shortest paths. Each path is interpolated at $16.7$~Hz with walking speed sampled uniformly from $1.0$ to $1.35$~m/s. Fusion-training and fusion-test walks are generated independently with fixed seeds 1 and 2, respectively. The implementation hashes the resulting binned target trajectories and aborts if an identical trajectory occurs in both sets.
    \item \textbf{PDR measurements:} The simulator computes the geometric path tangent $\theta^{true}_t$ and forms a noisy heading observation
    \begin{equation}
    \hat{\theta}_t=\theta^{true}_t+b_t+\epsilon_t,
    \label{eq:sim_heading}
    \end{equation}
    where $b_t$ is a Gaussian random walk with per-frame standard deviation $0.5^{\circ}/\sqrt{16.7}$ and $\epsilon_t$ is zero-mean white noise with standard deviation $8.8^{\circ}$. Acceleration magnitude is synthesized as a step-frequency sinusoid (frequency sampled from 1.7--2.0~Hz) around gravity with additive Gaussian noise, then passed through the causal step detector of Section~II-D. Every detected step uses the fixed nominal $L_s=0.65$~m.
    \item \textbf{Wi-Fi measurements:} At each scheduled scan time, the nearest surveyed node to the true simulated position is located, one stored RSSI fingerprint from that node is sampled, and the actual trained Wi-Fi heatmap network produces $\mathbf{z}_{wifi}$. The degraded regime changes the scan interval to 5~s and independently drops 40\% of AP entries before inference.
    \item \textbf{Magnetic measurements:} The four rotation-invariant magnetic channels are bilinearly sampled from the survey-derived magnetic map along the simulated path, perturbed using the measured map-noise scale, and passed through the actual 84-frame magnetic CNN. Only causal windows ending at the current fusion time produce $\mathbf{z}_{mag}$ and $\ell_{mag}$.
    \item \textbf{Estimator/ground-truth separation:} KalmanNet receives only binned PDR controls, Wi-Fi fixes and masks, magnetic CNN fixes and masks, and magnetic log-variance. The generated ground-truth positions are used only to construct the simulated sensor observations and as the supervised training/evaluation target; they are never included in the KalmanNet input vector.
\end{itemize}

We generate 250 fusion-training trajectories and 60 independent fusion-test trajectories, each represented by 160 fusion bins. This protocol provides controlled temporal ground truth while preserving a strict separation between the simulator's latent state and the observations supplied to the estimator.

'''
text, count = re.subn(
    r'\\section\{Experimental Setup and Dataset\}\\label\{sec:setup\}.*?(?=\\subsection\{Training Details\})',
    lambda _: setup_intro,
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f'setup subsection replacement count={count}')

old_device = 'To stringently evaluate device generalization, data originating from the Samsung Galaxy S9+ was explicitly held out during the training phase and reserved solely for evaluation.'
new_device = ('Device generalization is evaluated separately from temporal fusion. In the standalone Wi-Fi heatmap phone-split experiment and the real static KNN baseline, Samsung Galaxy S9+ fingerprints are held out from model fitting and used only for evaluation. The fusion benchmark instead uses survey-derived environment resources and evaluates 60 unseen synthetic trajectories; its reported KalmanNet errors should therefore be interpreted as trajectory-generalization results within the surveyed IT Engineering environment, not as held-out-device performance.')
if old_device not in text:
    raise SystemExit('device-generalization paragraph not found')
text = text.replace(old_device, new_device, 1)
paper.write_text(text, encoding='utf-8')

# -----------------------------------------------------------------------------
# Proofreading tracker: mark P1-P3 resolved and record the exact decision.
# -----------------------------------------------------------------------------
todo = Path('paper/reviews/prof_read_ieee_comments_draft.md')
t = todo.read_text(encoding='utf-8')
for key in ('P1.', 'P2.', 'P3.'):
    t = t.replace(f'### [ ] {key}', f'### [x] {key}', 1)
resolution = '''\n### P1-P3 implementation note\n\nResolved in the methodology-consistency pass:\n\n- **P1:** the evaluated fusion method no longer claims to consume MagWi `Orn_z` or a ground-truth-calibrated `phi_h`. The paper now defines PDR using a generic causal heading observation and documents the actual synthetic heading measurement used in the benchmark.\n- **P2:** the paper no longer claims that stride length is estimated from the fusion training trajectories. It documents the actual fixed nominal `L_s = 0.65 m`; no rolling estimator is claimed or implemented.\n- **P3:** the simulator/estimator boundary is now explicit, fusion results are labelled as held-out-trajectory rather than held-out-device evaluation, and the code saves fixed train/test seeds plus an exact binned-trajectory overlap check. The paper also explicitly states that survey-derived Wi-Fi/magnetic environment resources use the available processed survey fingerprints, so those fusion numbers are not presented as unseen-device generalization.\n\n'''
anchor = '## Priority B — architecture consistency / explanation\n'
if anchor not in t:
    raise SystemExit('TODO Priority B anchor not found')
t = t.replace(anchor, resolution + anchor, 1)
todo.write_text(t, encoding='utf-8')
