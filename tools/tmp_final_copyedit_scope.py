from pathlib import Path

MAIN = Path('paper/main.tex')
TRACKER = Path('paper/reviews/prof_read_ieee_comments_draft.md')

text = MAIN.read_text(encoding='utf-8')
tracker = TRACKER.read_text(encoding='utf-8')


def rep(src: str, dst: str, *, where='main', count=1):
    global text, tracker
    target = text if where == 'main' else tracker
    actual = target.count(src)
    if actual != count:
        raise SystemExit(f'{where}: expected {count} occurrence(s), found {actual}: {src[:120]!r}')
    target = target.replace(src, dst, count)
    if where == 'main':
        text = target
    else:
        tracker = target

# Abstract: standard capitalization/English while preserving every reported number.
rep(
    'Indoor localization can combine Wi-Fi, magnetic-field, and inertial sensing, but practical fusion requires a causal mechanism for reconciling sparse spatial measurements with drifting pedestrian dead reckoning (PDR) when measurement availability and reliability vary over time. We propose a decoupled learned state-space estimator in which a Wi-Fi heatmap Multi-Layer Perceptron (MLP) and an 84-frame gravity-referenced magnetic Convolutional Neural Network (CNN) independently produce Cartesian position measurements, while PDR supplies relative motion. A DualKalmanNet forms separate Wi-Fi and magnetic innovations and uses a Gated Recurrent Unit (GRU) to predict independent $2\\times2$ correction gains. Availability masks remove missing modalities, and a training-normalized magnetic log-uncertainty score scales the magnetic correction. Temporal fusion is evaluated on 60 independent map-constrained test trajectories within a fixed surveyed environment. With 1~Hz Wi-Fi, the uncertainty-weighted model obtains 0.494~m mean error (0.437~m median), compared with 0.473~m (0.449~m median) for Wi-Fi-only KalmanNet. Under degraded Wi-Fi (5~s updates with 40\\% AP dropout), it reduces mean error from 1.533~m to 1.154~m, a 24.7\\% reduction, while lowering the P90 error from 2.643~m to 1.612~m.',
    'Indoor localization can combine Wi-Fi, magnetic, and inertial sensing, but practical fusion requires a causal mechanism for reconciling sparse spatial measurements with drifting pedestrian dead reckoning (PDR) when measurement availability and reliability vary over time. We propose a decoupled learned state-space estimator in which a Wi-Fi heatmap multilayer perceptron (MLP) and an 84-frame gravity-referenced magnetic convolutional neural network (CNN) independently produce Cartesian position measurements, while PDR supplies relative motion. DualKalmanNet forms separate Wi-Fi and magnetic innovations and uses a gated recurrent unit (GRU) to predict independent $2\\times2$ correction gains. Availability masks remove missing modalities, and a training-normalized magnetic log-uncertainty score scales the magnetic correction. Temporal fusion is evaluated on 60 independent map-constrained test trajectories within a fixed surveyed environment. With 1~Hz Wi-Fi, the uncertainty-weighted model obtains 0.494~m mean error (0.437~m median), compared with 0.473~m (0.449~m median) for Wi-Fi-only KalmanNet. Under degraded Wi-Fi (5~s updates with 40\\% access-point dropout), it reduces mean error from 1.533~m to 1.154~m, a 24.7\\% reduction, while lowering the P90 error from 2.643~m to 1.612~m.'
)

# Introduction terminology and flow.
rep('magnetic-field', 'magnetic field', count=3)
rep(
    'Learned spatial models provide complementary measurement functions. DeepPositioning combines Wi-Fi and magnetic fingerprints using deep learning~\\cite{deeppos}; MINLOC applies a CNN to magnetic field patterns~\\cite{minloc}; and Zardkoohi \\emph{et al.} use a Bi-LSTM for magnetic-sequence localization~\\cite{bilstmmag}. A bidirectional recurrent estimator uses both past and future sequence context and therefore is not directly usable for streaming causal inference without reformulation. Our objective is not to claim that these models fail because of poor initialization, but to separate spatial measurement extraction from temporal correction and to make the contribution of each available measurement explicit in the state update.',
    'Learned spatial models provide complementary measurement functions. DeepPositioning combines Wi-Fi and magnetic fingerprints using deep learning~\\cite{deeppos}; MINLOC applies a CNN to magnetic field patterns~\\cite{minloc}; and Zardkoohi \\emph{et al.} use bidirectional long short-term memory for magnetic sequence localization~\\cite{bilstmmag}. Because a bidirectional recurrent estimator uses both past and future sequence context, it is not directly suitable for streaming causal inference without reformulation. Our objective is not to attribute prior limitations to initialization, but to separate spatial measurement extraction from temporal correction and make each available measurement explicit in the state update.'
)
rep(
    'In this work, we propose a decoupled causal learned state-space architecture for multimodal indoor localization, where \\emph{multimodal} denotes heterogeneous Wi-Fi, magnetic field, and inertial measurements. First, a Wi-Fi heatmap MLP and gravity-referenced magnetic CNN independently produce Cartesian position measurements; the magnetic CNN also provides a scalar relative-uncertainty score. Second, DualKalmanNet forms separate Cartesian innovations and predicts independent $2\\times2$ Wi-Fi and magnetic gain matrices, while availability masks remove unavailable modalities. Third, a training-normalized magnetic confidence weight explicitly attenuates uncertain magnetic corrections. The temporal fusion experiment evaluates independent map-constrained trajectories within one surveyed environment; device generalization is evaluated separately for the Wi-Fi and static-fingerprint experiments. The architecture can be reused after site-specific surveying and training, but the present fusion results do not constitute building-agnostic or plug-and-play unseen-device validation.',
    'In this work, we propose a decoupled causal learned state-space architecture for multimodal indoor localization, where \\emph{multimodal} denotes heterogeneous Wi-Fi, magnetic, and inertial measurements. First, a Wi-Fi heatmap MLP and a gravity-referenced magnetic CNN independently produce Cartesian position measurements; the magnetic CNN also provides a scalar relative uncertainty score. Second, DualKalmanNet forms separate Cartesian innovations and predicts independent $2\\times2$ Wi-Fi and magnetic gain matrices, while availability masks remove unavailable modalities. Third, a training-normalized magnetic confidence weight explicitly attenuates uncertain magnetic corrections. The temporal fusion experiment evaluates independent map-constrained trajectories within one surveyed environment; device generalization is evaluated separately for the Wi-Fi and static-fingerprint experiments. The architecture can be reused after site-specific surveying and training, but the present fusion results do not constitute building-agnostic or plug-and-play unseen-device validation.'
)

# Measurement definitions and Route A scope statement.
rep(
    'Assume a targeted indoor environment with $N$ identifiable Wi-Fi Access Points (APs) and a Cartesian reference grid containing $M$ spatial cells. We use $n$ for raw inertial/magnetic samples and $t$ for the lower-rate fusion steps. The phone-side quantities are a sparse Wi-Fi RSSI vector $\\mathbf{s}_t\\in\\mathbb{R}^N$, a magnetometer vector $\\mathbf{m}_n\\in\\mathbb{R}^3$, an accelerometer vector $\\mathbf{a}_n\\in\\mathbb{R}^3$, and a causal heading observation $\\hat{\\theta}_n$. The estimator state is $\\mathbf{x}_t=[x_t,y_t]^T\\in\\mathbb{R}^2$. Availability indicators $m_{\\mathrm{wifi},t},m_{\\mathrm{mag},t}\\in\\{0,1\\}$ specify whether a new Wi-Fi or magnetic measurement is available at fusion step $t$.',
    'Assume a targeted indoor environment with $N$ identifiable Wi-Fi access points (APs) and a Cartesian reference grid containing $M$ spatial cells. We use $n$ for raw inertial/magnetic samples and $t$ for the lower-rate fusion steps. The phone-side quantities are a sparse Wi-Fi received signal strength indicator (RSSI) vector $\\mathbf{s}_t\\in\\mathbb{R}^N$, a magnetometer vector $\\mathbf{m}_n\\in\\mathbb{R}^3$, an accelerometer vector $\\mathbf{a}_n\\in\\mathbb{R}^3$, and a causal heading observation $\\hat{\\theta}_n$. The estimator state is $\\mathbf{x}_t=[x_t,y_t]^T\\in\\mathbb{R}^2$. Availability indicators $m_{\\mathrm{wifi},t},m_{\\mathrm{mag},t}\\in\\{0,1\\}$ specify whether a new Wi-Fi or magnetic measurement is available at fusion step $t$.'
)
rep(
    "The per-phone centering defines a survey-specific magnetic feature domain. The present fusion evaluation operates inside that domain and does not establish that an uncalibrated unseen handset can enter it without an additional causal normalization procedure.",
    "The per-phone centering defines a survey-specific magnetic feature domain. The present fusion evaluation operates inside this centered domain. Because no causal alignment procedure is implemented for an uncalibrated handset, the magnetic fusion results are scoped to the surveyed, device-normalized domain rather than unseen-phone magnetic deployment."
)

# Estimator module transitions and terminology.
rep(
    'The estimator forms Wi-Fi and magnetic Cartesian measurements, propagates PDR motion, and combines them with DualKalmanNet. Figure~\\ref{fig:arch_diagram} shows the functional signal flow of one fusion update; the Wi-Fi, magnetic, and PDR measurement models that produce its input signals are detailed in the following subsections.',
    'The estimator combines two absolute spatial measurements with relative PDR motion. The Wi-Fi branch supplies intermittent Cartesian anchors, the magnetic branch supplies a causal sequence-based position measurement and relative uncertainty score, and PDR propagates motion between absolute updates. DualKalmanNet then converts these signals into a single causal state correction. Figure~\\ref{fig:arch_diagram} shows the functional signal flow of one fusion update; the component models are detailed below.'
)
rep('then applies Softmax, producing', 'then applies softmax, producing')
rep('the MLP Softmax output', 'the MLP softmax output')
rep('At inference, the Cartesian Wi-Fi measurement is the Softmax expectation', 'At inference, the Cartesian Wi-Fi measurement is the softmax expectation')
rep(
    '\\caption{Architecture of the Wi-Fi Multi-Layer Perceptron (MLP) environment model. Dense layers sequentially extract features from the RSSI vector to compute an $M$-dimensional probability heatmap.}',
    '\\caption{Architecture of the Wi-Fi multilayer perceptron (MLP) environment model. Dense layers sequentially extract features from the RSSI vector to compute an $M$-dimensional probability heatmap.}'
)
rep(
    'where $\\mathbf{c}_c\\in\\mathbb{R}^2$ is the physical coordinate of grid cell $c$. DualKalmanNet consumes $\\mathbf{z}_{\\mathrm{wifi},t}$ directly; a heatmap covariance is not part of its 13-dimensional GRU input.\n\n\\subsection{Magnetic Sequence Measurement Model}',
    'where $\\mathbf{c}_c\\in\\mathbb{R}^2$ is the physical coordinate of grid cell $c$. DualKalmanNet consumes $\\mathbf{z}_{\\mathrm{wifi},t}$ directly; a heatmap covariance is not part of its 13-dimensional GRU input. Thus, the Wi-Fi branch supplies an intermittent absolute Cartesian anchor to the fusion stage.\n\n\\subsection{Magnetic Sequence Measurement Model}'
)
rep(
    'Let $n_t$ denote the raw-sample index at the end of fusion step $t$. The magnetic input is the causal feature window',
    'The magnetic branch complements the Wi-Fi anchor with a causal sequence-based spatial measurement. Let $n_t$ denote the raw-sample index at the end of fusion step $t$. The magnetic input is the causal feature window'
)
rep(
    '\\caption{Code-accurate architecture of the magnetic sequence CNN for $T=84$. The shared Conv1D encoder reduces the temporal length $84\\to42\\to21$ before adaptive global averaging into a 128-dimensional representation. Separate heads predict the 2D magnetic position fix $\\mathbf{z}_{mag}$ and a scalar log-uncertainty score $\\ell_{mag}$.}',
    '\\caption{Code-accurate architecture of the magnetic sequence CNN for $T=84$. The shared Conv1D encoder reduces the temporal length $84\\to42\\to21$ before adaptive global averaging into a 128-dimensional representation. Separate heads predict the 2-D magnetic position fix $\\mathbf{z}_{\\mathrm{mag}}$ and a scalar log-uncertainty score $\\ell_{\\mathrm{mag}}$.}'
)
rep(
    '$\\ell_{\\mathrm{mag},t}$ is used only as a relative uncertainty indicator; its training objective is given in Section~\\ref{sec:setup}.\n\n\\subsection{Causal PDR Motion Model}',
    '$\\ell_{\\mathrm{mag},t}$ is used only as a relative uncertainty indicator; its training objective is given in Section~\\ref{sec:setup}. Together, the Wi-Fi and magnetic branches provide the absolute spatial measurements used for correction; PDR supplies the relative motion used between those measurements.\n\n\\subsection{Causal PDR Motion Model}'
)
rep(
    'PDR propagates relative motion between absolute updates~\\cite{nnwifipdr,axesmapping}.',
    'PDR complements the learned absolute measurements by propagating relative motion between their updates~\\cite{nnwifipdr,axesmapping}.'
)
rep(
    '\\subsection{Dual-Innovation KalmanNet Fusion}\\label{sec:kalmannet}\n\nFor an Extended Kalman Filter (EKF),',
    '\\subsection{Dual-Innovation KalmanNet Fusion}\\label{sec:kalmannet}\n\nThe preceding modules provide PDR motion, Wi-Fi position, and magnetic position/confidence signals. DualKalmanNet combines these signals in a common Cartesian prediction/correction update. For an extended Kalman filter (EKF),'
)
rep('Our dual extension predicts separate gains for Wi-Fi and magnetic measurements.', 'DualKalmanNet instead predicts separate gains for Wi-Fi and magnetic measurements.')
rep(
    'Setting an availability mask to zero removes that modality\'s correction. A large magnetic uncertainty score additionally reduces the magnetic term through $w_{\\mathrm{mag},t}$.\n\n\\section{Experimental Setup and Training}',
    'Setting an availability mask to zero removes that modality\'s correction. A large magnetic uncertainty score additionally reduces the magnetic term through $w_{\\mathrm{mag},t}$. These equations define the causal inference update; the next section specifies how its measurement models and recurrent gain generator are trained and evaluated.\n\n\\section{Experimental Setup and Training}'
)

# Training/evaluation prose and explicit P11 scope.
rep(
    'We evaluate the proposed fusion architecture in the IT Engineering building of the MagWi Benchmark Dataset~\\cite{magwi}. The surveyed fingerprints define the environment model used to generate Wi-Fi and magnetic measurements, while temporal fusion is evaluated on newly generated map-constrained trajectories. Consequently, the fusion experiment tests generalization to unseen trajectories within a fixed surveyed environment; it is not, by itself, a held-out-smartphone experiment.',
    'We evaluate the proposed fusion architecture in the IT Engineering building of the MagWi Benchmark Dataset~\\cite{magwi}. Surveyed fingerprints define the environment resources used to generate Wi-Fi and magnetic measurements, while temporal fusion is evaluated on newly generated map-constrained trajectories. Consequently, the fusion experiment tests generalization to unseen trajectories within a fixed surveyed environment; it is not a held-out-device experiment. The magnetic evaluation also remains inside the per-phone-centered survey domain defined in Section~\\ref{sec:measurements}; no online alignment of an uncalibrated handset to that domain is evaluated.'
)
rep('The historical training implementation uses the uncertainty-weighted regression objective', 'The implemented training objective is the uncertainty-weighted regression objective')
rep(
    '    \\item \\textbf{Survey environment:} An $\\varepsilon$-graph ($\\varepsilon=1.6$~m) is built from surveyed IT Engineering reference-node coordinates. The processed survey database is also used to construct a pool of real Wi-Fi scans and an interpolated four-channel magnetic map. These environment resources include the available surveyed devices; therefore the fusion results are not claimed as an unseen-device test.',
    '    \\item \\textbf{Survey environment:} An $\\varepsilon$-graph ($\\varepsilon=1.6$~m) is built from surveyed IT Engineering reference-node coordinates. The processed survey database is also used to construct a pool of real Wi-Fi scans and an interpolated four-channel magnetic map after per-phone centering. These environment resources include the available surveyed devices, and the simulator samples directly from the centered magnetic domain; therefore the fusion results are not claimed as an unseen-device test or as validation of causal handset-to-map alignment.'
)
rep('equivalent to assuming that the initial 2D position is known', 'equivalent to assuming that the initial 2-D position is known')
rep('and the actual trained Wi-Fi heatmap network produces', 'and the trained Wi-Fi heatmap network produces')
rep('and passed through the actual 84-frame magnetic CNN.', 'and passed through the trained 84-frame magnetic CNN.')
rep(
    'All neural network components were developed using the PyTorch framework. The MLP and CNN were optimized via Adam ($\\beta_1 = 0.9, \\beta_2 = 0.999$, weight decay $10^{-4}$) with an initial learning rate of $10^{-3}$, dynamically decayed via a ReduceLROnPlateau scheduler (patience 8, factor 0.5). The MLP uses a Dropout rate of $p = 0.3$ and a Gaussian target with spatial standard deviation $\\sigma = 2.0$~m. The MLP was trained for 80 epochs; the CNN for 60 epochs. The 1D-CNN magnetic sequence matcher uses $T=84$ frames (5.0~s at 16.7~Hz), selected from the candidate set $\\{50,84,134,167\\}$ by the development sweep. The KalmanNet GRU was trained for 150 epochs via Adam with a learning rate of $2 \\times 10^{-3}$, weight decay of $10^{-5}$, and MSE loss against ground-truth trajectories. For each signal-availability regime, the reference magnetic log-uncertainty score $\\ell_{\\mathrm{ref}}$ in~(\\ref{eq:mag_ref}) is computed only from the 250 fusion-training trajectories and is then frozen for evaluation.',
    'All neural components are implemented in PyTorch. The MLP and CNN are optimized with Adam ($\\beta_1=0.9$, $\\beta_2=0.999$, weight decay $10^{-4}$) from an initial learning rate of $10^{-3}$, reduced with a ReduceLROnPlateau scheduler (patience 8, factor 0.5). The MLP uses dropout $p=0.3$ and a Gaussian target with spatial standard deviation $\\sigma=2.0$~m. The MLP is trained for 80 epochs and the CNN for 60 epochs. The 1-D CNN magnetic sequence matcher uses $T=84$ frames (5.0~s at 16.7~Hz), selected from the candidate set $\\{50,84,134,167\\}$ by the development sweep. The KalmanNet GRU is trained for 150 epochs with Adam, learning rate $2\\times10^{-3}$, weight decay $10^{-5}$, and mean squared error against ground-truth trajectories. For each signal-availability regime, the reference magnetic log-uncertainty score $\\ell_{\\mathrm{ref}}$ in~(\\ref{eq:mag_ref}) is computed only from the 250 fusion-training trajectories and then frozen for evaluation.'
)
rep(
    'Device generalization is evaluated separately from temporal fusion. In the standalone Wi-Fi heatmap phone-split experiment and the real static KNN baseline, Samsung Galaxy S9+ fingerprints are held out from model fitting and used only for evaluation. The fusion benchmark instead uses survey-derived environment resources and evaluates 60 unseen synthetic trajectories; its reported KalmanNet errors should therefore be interpreted as trajectory-generalization results within the surveyed IT Engineering environment, not as held-out-device performance.',
    'Device generalization is evaluated separately from temporal fusion. In the standalone Wi-Fi heatmap phone-split experiment and the real static KNN baseline, Samsung Galaxy S9+ fingerprints are held out from model fitting and used only for evaluation. The fusion benchmark instead uses survey-derived environment resources and evaluates 60 unseen synthetic trajectories. Its KalmanNet errors therefore measure trajectory generalization within the surveyed IT Engineering environment, not held-out-device performance. In particular, the magnetic fusion pipeline assumes the survey-centered feature domain and does not evaluate causal normalization of a new handset.'
)

# Results wording/terminology without changing any values.
rep('The standalone performance of the Wi-Fi heatmap environment model prior to temporal fusion is shown', 'The standalone performance of the Wi-Fi heatmap environment model before temporal fusion is shown')
rep('For a classical non-neural reference, we additionally evaluate K-nearest-neighbor (KNN) localization', 'For a classical non-neural reference, we additionally evaluate k-nearest-neighbor (KNN) localization')
rep('WiFi-only KalmanNet', 'Wi-Fi-only KalmanNet', count=2)
rep('WiFi+Mag KNN (non-temp.)', 'Wi-Fi+magnetic KNN (non-temp.)', count=2)
rep(
    'Thus relative magnetic uncertainty gives a 24.7\\% mean-error reduction over Wi-Fi-only KalmanNet and lowers the unweighted model\'s P90 from 2.064~m to 1.612~m. The KNN gap shows that PDR propagation and recurrent context matter beyond simply having both absolute estimates available.',
    'Thus relative magnetic uncertainty gives a 24.7\\% mean-error reduction over Wi-Fi-only KalmanNet and lowers the unweighted model\'s P90 from 2.064~m to 1.612~m. In this matched protocol, the KNN gap indicates that PDR propagation and recurrent context contribute beyond simply having both absolute estimates available.'
)
rep(
    'Figure~\\ref{fig:trajectory_current} shows the same degraded regime geometrically. Open-loop PDR drifts through corridor turns and Wi-Fi-only KalmanNet can re-anchor only at sparse scans, whereas the CNN supplies causal magnetic position innovations between scans and relative uncertainty suppresses unreliable corrections. The fused path therefore remains closer to the surveyed route. The displayed walk is a central-performance case rather than the best improvement; its exact selection data are stored with the benchmark output.',
    'Figure~\\ref{fig:trajectory_current} shows the same degraded regime geometrically. Open-loop PDR drifts through corridor turns, and Wi-Fi-only KalmanNet can re-anchor only at sparse scans, whereas the CNN supplies causal magnetic position innovations between scans and relative uncertainty suppresses unreliable corrections. On this representative walk, the fused path therefore remains closer to the surveyed route. The displayed walk is a central-performance case rather than the best improvement; its exact selection data are stored with the benchmark output.'
)

# Conclusion: Route A scope + Route B as future work, with no page compression language.
rep(
    'This paper presented a decoupled learned state-space architecture that separates spatial inference from temporal tracking. A Wi-Fi heatmap MLP and gravity-referenced magnetic sequence CNN provide Cartesian measurements, while DualKalmanNet predicts independent $2\\!\\times\\!2$ gains; availability masks and relative magnetic uncertainty suppress missing or unreliable corrections. Under degraded Wi-Fi, the uncertainty-weighted model reduces mean error from 1.533~m for Wi-Fi-only KalmanNet to 1.154~m (24.7\\%) and reduces the high-error tail. The present fusion results demonstrate trajectory generalization within a surveyed environment and still require building-specific environment resources. Future work will extend the state to three-dimensional, multi-floor positioning, including barometric altitude information.',
    'This paper presented a decoupled learned state-space architecture that separates spatial inference from temporal tracking. A Wi-Fi heatmap MLP and gravity-referenced magnetic sequence CNN provide Cartesian measurements, while DualKalmanNet predicts independent $2\\!\\times\\!2$ gains; availability masks and relative magnetic uncertainty suppress missing or unreliable corrections. Under degraded Wi-Fi, the uncertainty-weighted model reduces mean error from 1.533~m for Wi-Fi-only KalmanNet to 1.154~m (24.7\\%) and reduces the high-error tail. The present fusion results demonstrate trajectory generalization within a surveyed environment and require building-specific environment resources. They also assume access to the survey-centered magnetic feature domain; causal alignment of an uncalibrated handset to that domain is not evaluated. Future work will investigate such device-domain alignment without position ground truth and extend the state to three-dimensional multi-floor positioning with barometric altitude information.'
)

# Tracker: close P11 via Route A, close editorial comments/J15, and preserve Route B as non-blocking future work.
rep('### [ ] P11. Real-device magnetic centering / deployment gap uncovered during P6-P7', '### [x] P11. Real-device magnetic centering / deployment gap uncovered during P6-P7', where='tracker')
rep(
    'The current magnetic-map trainer subtracts each phone\'s mean feature value before node averaging and interpolation. The synthetic CNN/fusion evaluation then samples directly from this centered survey map. A physical unseen phone would need a causal normalization/calibration procedure to map live `magN/magV/magH/dip` features into the same centered domain, but this step is not presently implemented or evaluated. Keep held-out-device claims separate from the magnetic fusion experiment and decide whether to add an online centering strategy in a future experiment or explicitly scope the current paper to the surveyed magnetic domain.',
    'The current magnetic-map trainer subtracts each phone\'s mean feature value before node averaging and interpolation. The synthetic CNN/fusion evaluation then samples directly from this centered survey map. A physical unseen phone would need a causal normalization/calibration procedure to map live `magN/magV/magH/dip` features into the same centered domain, but this step is not presently implemented or evaluated.\n\n**Resolved via Route A:** the manuscript now explicitly scopes magnetic fusion to the surveyed, per-phone-centered feature domain in the preprocessing, experimental-setup, device-generalization, and conclusion text. It does not claim causal alignment or plug-and-play magnetic fusion for an uncalibrated unseen handset. A causal alignment strategy is retained below as a non-blocking future-work item rather than being added ad hoc to the present experiment.',
    where='tracker'
)
rep('### [ ] G4. Improve flow between subsections/modules and perform a grammar pass', '### [x] G4. Improve flow between subsections/modules and perform a grammar pass', where='tracker')
rep(
    'Add short motivation/transition sentences between Wi-Fi, magnetic, PDR, and KalmanNet modules; prefer top-down explanations; remove abrupt jumps; fix capitalization, punctuation, abbreviations, article usage, hyphenation, and awkward phrasing.',
    'Add short motivation/transition sentences between Wi-Fi, magnetic, PDR, and KalmanNet modules; prefer top-down explanations; remove abrupt jumps; fix capitalization, punctuation, abbreviations, article usage, hyphenation, and awkward phrasing.\n\n**Resolved in final copy-edit/scope pass:** module transitions now explain the functional role of each branch (Wi-Fi anchor, magnetic sequence measurement/confidence, PDR propagation, and recurrent correction), while the full manuscript was edited for grammar, punctuation, articles, hyphenation, and sentence flow without changing reported results.',
    where='tracker'
)
rep('### [ ] J1. Capitalization, abbreviations, and standard English usage throughout - especially the abstract', '### [x] J1. Capitalization, abbreviations, and standard English usage throughout - especially the abstract', where='tracker')
rep('**Status:** still applicable as a global copy-edit.', '**Resolved in final copy-edit/scope pass:** standardized common-noun capitalization and first-use abbreviations in the abstract/body (including multilayer perceptron, convolutional neural network, gated recurrent unit, access points, RSSI, softmax, Wi-Fi, and magnetic field) while preserving proper names such as KalmanNet and DualKalmanNet.', where='tracker')
rep('### [ ] J15. Legacy anomaly-map mathematics - preserve as a historical warning, do not reintroduce it', '### [x] J15. Legacy anomaly-map mathematics - preserve as a historical warning, do not reintroduce it', where='tracker')
rep(
    '**Current status:** the active architecture has removed `A_obs`, `A(x)`, and `nabla A`; therefore these exact comments are **superseded**. Their lasting lesson is that every new function/measurement we keep (`z_mag`, `ell_mag`, `w_mag`, heatmap covariance, etc.) must be defined with equal mathematical precision.',
    '**Resolved / superseded:** the active architecture has removed `A_obs`, `A(x)`, and `nabla A`, so the V3 anomaly-map equations are intentionally not reintroduced. The retained warning is historical: every active function/measurement must remain mathematically defined with the precision now used for $\\mathbf{z}_{\\mathrm{mag}}$, $\\ell_{\\mathrm{mag}}$, and $w_{\\mathrm{mag}}$.',
    where='tracker'
)
rep('### [ ] J18. Final grammar/punctuation micro-edits from the annotations', '### [x] J18. Final grammar/punctuation micro-edits from the annotations', where='tracker')
rep('**Status:** applicable as final pass.', '**Resolved in final copy-edit/scope pass:** completed sentence-level article, punctuation, spacing, capitalization, hyphenation, and awkward-phrase cleanup after the structural and mathematical revisions stabilized.', where='tracker')

old_tail = '''---\n\n## Consolidated order for the next session\n\n1. **Validity first:** P1, P2, P3 - reconcile the experiment with the paper and audit leakage/generalization claims.\n2. **Scientific framing:** G1, J2-J6, J16 - establish exactly what problem/contribution is defensible and verify the literature/citations.\n3. **Paper architecture:** G3, J10 - separate measurements/preprocessing, methodology, and training/losses.\n4. **Mathematical definitions:** G2, J11-J14, J17 plus P4-P7 - make every retained module/code path mathematically auditable.\n5. **Figures/presentation:** J7, P8, P9, J8-J9.\n6. **Copy-edit:** G4, J1, J12-J13 where relevant, J18, P10.\n\nThe goal is to finish the next revision as a **methodologically consistent new draft**, not to patch the old V3 sentence-by-sentence.'''
new_tail = '''---\n\n## Current review status\n\nAll manuscript changes tracked from the professor/reviewer comments above are resolved in the current draft. Remaining work below is future expansion rather than a blocker for the present paper.\n\n## Future extensions (non-blocking)\n\n### [ ] F1. Causal unseen-phone magnetic domain alignment (Route B)\n\nDevelop and evaluate a causal procedure that maps live magnetic features from an uncalibrated handset into the survey-centered magnetic feature domain without using position ground truth. A valid study should separate handset bias from spatial magnetic structure (rather than naively subtracting a short-walk mean), specify the required calibration horizon/information, and evaluate the resulting magnetic CNN/fusion pipeline on genuinely held-out devices. This is a potential future research extension, not part of the current reported experiment.'''
rep(old_tail, new_tail, where='tracker')

# Guardrails: metrics and core methodology must remain present.
for token in [
    '0.494~m', '0.473~m', '1.533~m', '1.154~m', '24.7\\%', '1.612~m',
    '250 fusion-training trajectories', '60 independent fusion-test trajectories',
    'T=84', 'L_s=0.65~\\mathrm{m}', '\\boldsymbol{\\phi}_t\\in\\mathbb{R}^{13}',
]:
    if token not in text:
        raise SystemExit(f'guardrail token missing after edit: {token}')

MAIN.write_text(text, encoding='utf-8')
TRACKER.write_text(tracker, encoding='utf-8')
print('Applied final copy-edit, flow, and Route A scope pass.')
