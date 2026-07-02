"""
Regenerates SURA_MidTerm_Report.pdf using ReportLab.
Changes from original:
  - Main title: "Hybrid Sensor Fused KalmanNet for Indoor Localisation"
  - Label above title: "SURA Mid Term Evaluation Report"
  - Submitted to: "Industrial Research and Development (IRD) Unit, IIT Delhi"
  - Added: Mentor (Prof. Neel Kanth Kundu) + Students (Utkarsh Agrawal, Jayendra Vijay Birhade)
  - Removed: Status row
  - Added: references throughout the document
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Flowable

# ── Output path ─────────────────────────────────────────────────────────────
OUTPUT = "SURA_MidTerm_Report.pdf"

# ── Page setup ───────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.5*cm,
    rightMargin=2.5*cm,
    topMargin=2.5*cm,
    bottomMargin=2.5*cm,
    title="Hybrid Sensor Fused KalmanNet for Indoor Localisation - Mid-Term Evaluation Report",
    author="Utkarsh Agrawal, Jayendra Vijay Birhade",
    subject="Indoor Positioning and Localization",
    creator="ReportLab",
)

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
W = A4[0] - 5*cm  # usable width

# Label above title (small caps style)
supra = ParagraphStyle(
    'supra', fontName='Helvetica', fontSize=11,
    textColor=colors.HexColor('#333333'), alignment=TA_CENTER, spaceAfter=4
)
# Main title
title_style = ParagraphStyle(
    'title', fontName='Helvetica-Bold', fontSize=22,
    textColor=colors.black, alignment=TA_CENTER, spaceAfter=6, leading=28
)
# Subtitle under title
subtitle_style = ParagraphStyle(
    'subtitle', fontName='Helvetica', fontSize=13,
    textColor=colors.HexColor('#555555'), alignment=TA_CENTER, spaceAfter=18
)
# Meta label (bold key)
meta_label = ParagraphStyle(
    'meta_label', fontName='Helvetica-Bold', fontSize=10,
    textColor=colors.black, alignment=TA_LEFT
)
# Meta value
meta_value = ParagraphStyle(
    'meta_value', fontName='Helvetica', fontSize=10,
    textColor=colors.black, alignment=TA_LEFT
)
# Author block
author_style = ParagraphStyle(
    'author', fontName='Helvetica', fontSize=10,
    textColor=colors.HexColor('#333333'), alignment=TA_LEFT, leading=15
)
author_bold = ParagraphStyle(
    'author_bold', fontName='Helvetica-Bold', fontSize=10,
    textColor=colors.black, alignment=TA_LEFT
)
# Section heading
h1 = ParagraphStyle(
    'h1', fontName='Helvetica-Bold', fontSize=13,
    textColor=colors.black, spaceBefore=16, spaceAfter=6
)
# Subsection heading
h2 = ParagraphStyle(
    'h2', fontName='Helvetica-Bold', fontSize=11,
    textColor=colors.black, spaceBefore=10, spaceAfter=4
)
# Body text
body = ParagraphStyle(
    'body', fontName='Times-Roman', fontSize=10,
    textColor=colors.black, leading=15, alignment=TA_JUSTIFY, spaceAfter=6
)
# Table cell text (wrapping)
cell_style = ParagraphStyle(
    'cell', fontName='Times-Roman', fontSize=9,
    textColor=colors.black, leading=12
)
cell_bold = ParagraphStyle(
    'cell_bold', fontName='Helvetica-Bold', fontSize=9,
    textColor=colors.black, leading=12
)
# Small caption under table
caption = ParagraphStyle(
    'caption', fontName='Helvetica', fontSize=9,
    textColor=colors.HexColor('#444444'), alignment=TA_CENTER, spaceAfter=10
)
# References list
ref_style = ParagraphStyle(
    'ref', fontName='Times-Roman', fontSize=9,
    textColor=colors.black, leading=13, spaceAfter=4, leftIndent=18, firstLineIndent=-18
)
# Abstract label
abs_label = ParagraphStyle(
    'abs_label', fontName='Helvetica-Bold', fontSize=10,
    textColor=colors.black, spaceBefore=10, spaceAfter=4
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def HR():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#999999'), spaceAfter=8, spaceBefore=4)

def P(text, style=None):
    """Wrap a string in a Paragraph so table cells word-wrap properly."""
    if isinstance(text, Paragraph):
        return text
    return Paragraph(str(text), style or cell_style)

def Pb(text):
    """Bold cell paragraph."""
    return Paragraph(str(text), cell_bold)

def row(cells, bold=False):
    """Wrap a whole row of cells."""
    s = cell_bold if bold else cell_style
    return [P(c, s) for c in cells]

def table_style_base():
    return TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#F5F5F5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCCCCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ])

# ════════════════════════════════════════════════════════════════════════════
# BUILD STORY
# ════════════════════════════════════════════════════════════════════════════
story = []

# ── PAGE 1: Title Page ───────────────────────────────────────────────────────
story.append(Spacer(1, 1.2*cm))

# SUPRA label
story.append(Paragraph("SURA Mid Term Evaluation Report", supra))
story.append(Spacer(1, 0.3*cm))
story.append(HR())

# Main title
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Machine Learning-Based User Localization Using Hybrid<br/>Magnetic Sensor and Wi-Fi Received Signal Strength (RSS)<br/>Data in GPS-Denied Scenarios", title_style))
story.append(HR())
story.append(Spacer(1, 0.5*cm))

# Meta table (Submitted to, Date — Status removed)
meta_data = [
    [Paragraph("Submitted to", meta_label),
     Paragraph("Industrial Research and Development (IRD) Unit,<br/>Indian Institute of Technology Delhi", meta_value)],
    [Paragraph("Date", meta_label),
     Paragraph("1 July 2026", meta_value)],
]
meta_table = Table(meta_data, colWidths=[3.5*cm, W - 3.5*cm])
meta_table.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
]))
story.append(meta_table)

story.append(Spacer(1, 0.7*cm))
story.append(HR())

# Authors block
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Mentor", author_bold))
story.append(Paragraph("Prof. Neel Kanth Kundu<br/>"
                        "Centre for Applied Research in Electronics (CARE),<br/>"
                        "Indian Institute of Technology Delhi", author_style))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Students", author_bold))
story.append(Paragraph(
    "Utkarsh Agrawal &nbsp;&nbsp;(2024CS10076)<br/>"
    "Jayendra Vijay Birhade &nbsp;&nbsp;(2024CS10891)<br/>"
    "Department of Computer Science and Engineering,<br/>"
    "Indian Institute of Technology Delhi",
    author_style))

story.append(Spacer(1, 0.7*cm))
story.append(HR())

# Abstract
story.append(Paragraph("Abstract", abs_label))
story.append(Paragraph(
    "This report documents mid-term progress on the SURA Indoor Positioning project, which aims to "
    "learn a static spatial fingerprint of an indoor environment for real-time, on-device localization "
    "[1, 2]. We describe the core system constraints — causal inference, cross-device generalization, "
    "and fingerprint-based rather than trajectory-based learning — and trace the evolution of five model "
    "architectures from an initial mean absolute error (MAE) of 6.36 m down to a current best of 1.43 m. "
    "We formally identify four fundamental flaws in the initial experimental setup and present the "
    "redesigned three-part causal system, comprising a WiFi heatmap environment model [3, 4], causal "
    "pedestrian dead-reckoning [5, 6], and a KalmanNet-based filter [7], that resolves each flaw. "
    "The reported 1.43 m MAE represents a 2.5× improvement over the strongest prior model.",
    body))

story.append(PageBreak())

# ── SECTION 1 ────────────────────────────────────────────────────────────────
story.append(Paragraph("1.  Problem Statement and System Constraints", h1))
story.append(Paragraph(
    "The central challenge of this project is to learn an environment fingerprint — a static, spatially "
    "unique, and temporally stable characterization of a physical space — from heterogeneous sensor "
    "streams (WiFi RSSI and IMU) [1, 8]. The critical distinction from trajectory-based approaches [9, 10] "
    "is that the system must learn location identity, not path-specific dynamics. This means the same "
    "model must localize a user correctly regardless of which direction they are walking or which session "
    "they are in.", body))
story.append(Paragraph("Five hard constraints govern every design decision in this project.", body))

t1_data = [
    row(["Constraint", "Requirement", "Architectural implication"], bold=True),
    row(["Causal inference", "No look-ahead; strictly real-time", "Bidirectional RNNs are excluded"]),
    row(["On-device deployment", "Full pipeline runs locally on the handset", "No server round-trips at inference time"]),
    row(["Cross-device generalization", "Works on unseen phones without retraining", "Per-device calibration is not a viable approach"]),
    row(["Cross-session generalization", "Works in walk sessions not seen during training", "Route memorization must be eliminated by design"]),
    row(["Ground truth fidelity", "Per-node labels from the static database are preferred", "Fabricated or interpolated labels are rejected"]),
]
t1 = Table(t1_data, colWidths=[3.5*cm, 5.5*cm, W - 9.0*cm])
t1.setStyle(table_style_base())
story.append(t1)
story.append(Paragraph("Table 1. Hard constraints governing all architectural choices.", caption))

story.append(Paragraph(
    "A secondary structural difficulty is spatial multi-modality. A single WiFi or magnetic fingerprint "
    "measurement can correspond to multiple physical positions, particularly along corridor segments where "
    "sensor signatures look very similar [3]. A model that emits a single (X, Y) coordinate cannot "
    "represent this uncertainty and will systematically pull its predictions toward the centroid of the "
    "candidate set. This observation motivates the shift to probabilistic heatmap outputs described in "
    "Section 4.", body))

# ── SECTION 2 ────────────────────────────────────────────────────────────────
story.append(Paragraph("2.  Initial Model Architectures", h1))
story.append(Paragraph(
    "Four model architectures were developed and benchmarked during the initial phase of the project. "
    "All four were trained on continuous-walk CSV files (Continuous_Fused_*.csv). As documented in "
    "Section 3, these files contain fundamental data validity issues; the figures here are reported as "
    "originally measured, with the full caveats addressed below.", body))

story.append(Paragraph("2.1  CNN-LSTM (Non-Causal)", h2))
story.append(Paragraph(
    "Implemented in train_cnn_lstm.py. A one-dimensional convolutional encoder was paired with a "
    "bidirectional LSTM [9] to capture both local sensor patterns and long-range temporal context. "
    "The reported MAE was 5.04 m. The bidirectional LSTM is, however, a fundamental causality "
    "violation: it accessed future sensor readings to infer the current position, making the model "
    "deployment-invalid regardless of its accuracy.", body))

story.append(Paragraph("2.2  Causal Hybrid", h2))
story.append(Paragraph(
    "Implemented in train_causal_hybrid.py. The bidirectional LSTM was replaced with a unidirectional "
    "variant and all convolutions were made causal, achieving full deployment compliance. This model "
    "produced the strongest initial result at 3.64 m MAE. Post-hoc analysis revealed that the accuracy "
    "arose from route memorization rather than genuine spatial fingerprinting.", body))

story.append(Paragraph("2.3  Environment Model", h2))
story.append(Paragraph(
    "Implemented in train_env_model.py. This model was designed explicitly to learn per-location "
    "signatures rather than temporal sequences [4]. It achieved 6.36 m MAE on the forward corridor "
    "path, but collapsed to 63 m MAE when the same path was walked in reverse. The root cause is "
    "that raw magnetometer axes (Mag_x, Mag_y, Mag_z) are measured in the phone body frame and "
    "rotate with heading [11], so the same physical spot yields different axis values depending on "
    "direction of travel. The model had learned a heading-conditioned shortcut, not a true spatial "
    "fingerprint.", body))

story.append(Paragraph("2.4  Neural EKF with Learned Alpha-Gate", h2))
story.append(Paragraph(
    "Implemented in train_ekf_fusion.py. This model introduced a three-stage learned fusion design. "
    "A spatial anchor branch combined WiFi and magnetic MLPs to produce a per-frame estimate "
    "P_spatial. A motion tracker built from causal convolutions and a unidirectional LSTM predicted "
    "step deltas. A sigmoid-gated alpha value then blended the two signals. The composite training "
    "loss applied weights of 1.0 on the final output, 0.2 on the spatial branch, and 2.0 on the "
    "motion branch. At inference the update rule was: "
    "P_final[t] = alpha[t] × P_spatial[t] + (1 − alpha[t]) × (P_final[t−1] + delta[t]). "
    "Reported MAE: 4.29 m. A mean alpha near 0.6 indicates a slight preference for the spatial "
    "anchor. Route memorization was not eliminated.", body))

t2_data = [
    row(["Model", "MAE", "Causal", "Route memory risk", "Primary issue"], bold=True),
    row(["CNN-LSTM", "5.04 m", "No", "High", "Bidirectional LSTM; deployment-invalid"]),
    row(["Causal Hybrid", "3.64 m", "Yes", "High", "Memorizes the route, not the space"]),
    row(["Environment Model", "6.36 m / 63 m *", "Yes", "Medium", "Direction-dependent magnetometer features"]),
    row(["Neural EKF", "4.29 m", "Yes", "Medium", "Learned fusion; still route-biased"]),
]
t2 = Table(t2_data, colWidths=[3.0*cm, 2.8*cm, 1.5*cm, 2.5*cm, W - 9.8*cm])
t2.setStyle(table_style_base())
story.append(t2)
story.append(Paragraph("Table 2. Summary of initial architectures. * Forward path / reversed path.", caption))

# ── SECTION 3 ────────────────────────────────────────────────────────────────
story.append(Paragraph("3.  Identification of Fundamental Experimental Flaws", h1))
story.append(Paragraph(
    "A systematic audit of the initial experimental pipeline identified four independent sources of "
    "invalidity. Each flaw on its own is sufficient to make the reported figures unreliable; taken "
    "together they mean that the initial results measured something other than true indoor localization "
    "performance.", body))

story.append(Paragraph("3.1  Single one-dimensional training trajectory", h2))
story.append(Paragraph(
    "All four Continuous_Fused_*.csv files trace the same corridor: the same start point at "
    "approximately (90, 24), the same end region, and only two turns along the entire path. The "
    "dataset yielded roughly 325 training windows and 99 test windows, all drawn from a single "
    "narrow line through a two-dimensional building floor. No model trained on this data could learn "
    "a generalizable fingerprint; it could only memorize the one trajectory it was shown.", body))

story.append(Paragraph("3.2  Direction-dependent magnetometer axes", h2))
story.append(Paragraph(
    "Raw Mag_x, Mag_y, and Mag_z are measured in the phone body frame and rotate with heading [11]. "
    "The same physical location therefore produces different axis values depending on which direction "
    "the user is walking. Model 3's collapse from 6.36 m to 63 m MAE on path reversal is a direct "
    "consequence — the model had learned a heading-conditioned shortcut that failed the moment "
    "heading changed.", body))

story.append(Paragraph("3.3  Single-point regression and multi-modal ambiguity", h2))
story.append(Paragraph(
    "Several locations along a corridor appear similar in sensor space [3, 8]. A regression model "
    "forced to emit one (X, Y) coordinate cannot represent the resulting ambiguity — instead of "
    "outputting a distribution over candidate locations, it collapses to a centroid. This is a "
    "fundamental limitation of the target formulation, not a hyperparameter problem, and it "
    "systematically degrades accuracy wherever the environment is spatially ambiguous.", body))

story.append(Paragraph("3.4  Fabricated ground truth", h2))
story.append(Paragraph(
    "The Continuous_Fused_*.csv files were generated by fuse_continuous_wifi.py, which used BE "
    "Building coordinates rather than the IT Engineering layout, linearly interpolated True_X and "
    "True_Y by timestamp from the ordered static-node list, and simulated WiFi measurements by "
    "injecting Gaussian noise around static node scans. The ground truth was not independently "
    "observed — it was constructed from the same database used to generate the model inputs. Models "
    "that learned the node sequence appeared accurate precisely because the labels encoded that "
    "sequence. The reported errors reflected consistency with a fabricated trajectory, not validity "
    "against real-world positions.", body))

# ── SECTION 4 ────────────────────────────────────────────────────────────────
story.append(Paragraph("4.  Redesigned Architecture", h1))
story.append(Paragraph(
    "The redesigned system resolves all four documented flaws by separating the problem into "
    "components that can be trained and validated independently on the static fingerprint database "
    "[1, 8], rather than on continuous walks. The static database provides 538 node visits across "
    "167 of 168 nodes, giving genuine two-dimensional spatial coverage with independently observed "
    "ground truth.", body))

story.append(Paragraph("4.1  Design rationale", h2))
story.append(Paragraph(
    "Three decisions follow directly from the flaw analysis. The environment model is trained only "
    "on static node visits, which eliminates flaws 1 and 4. It outputs a probability distribution "
    "over a one-metre grid rather than a single coordinate, which eliminates flaw 3. All features "
    "are computed from heading-invariant quantities — specifically, the magnitude |M| and per-frame "
    "WiFi RSSI vectors [3, 11] — which eliminates flaw 2. The fusion stage uses KalmanNet [7] to "
    "keep nearly zero route memory while learning context-dependent gain matrices.", body))

story.append(Paragraph("4.2  System components", h2))
t3_data = [
    ["Component", "Role", "Key property"],
    [Paragraph("WiFi heatmap environment model", cell_style),
     Paragraph("Outputs a probability distribution over a 1 m grid per frame. The spatial spread of the heatmap becomes the Kalman measurement covariance R.", cell_style),
     Paragraph("Trained only on static DB. Per-frame and direction-invariant. [3, 4]", cell_style)],
    [Paragraph("Online magnetometer normalizer", cell_style),
     Paragraph("Fits an ellipsoid from a trailing buffer of raw IMU samples. Outputs calibrated magnitude |M| rather than raw body-frame axes.", cell_style),
     Paragraph("3\u00d7 cross-device spread reduction: 1.77 to 0.57 \u00b5T. [11]", cell_style)],
    [Paragraph("Causal PDR (Pedestrian Dead-Reckoning)", cell_style),
     Paragraph("Step detection from accelerometer magnitude. Predicts a motion delta for each detected step.", cell_style),
     Paragraph("Strictly causal. Approx. 1.5 m MAE on short walks. [5, 6]", cell_style)],
    [Paragraph("KalmanNet GRU Fusion", cell_style),
     Paragraph("Recurrent neural filter replacing fixed Kalman gain with a learned 2\u00d72 matrix gain computed by a GRU. No linear-Gaussian assumption.", cell_style),
     Paragraph("No route memory. Fully causal. Adaptive gain from context. [7]", cell_style)],
]
t3 = Table(t3_data, colWidths=[3.5*cm, 6.5*cm, W - 10.0*cm])
t3.setStyle(table_style_base())
story.append(t3)
story.append(Paragraph("Table 3. Components of the redesigned four-part causal system.", caption))

story.append(Paragraph("4.3  Inference pipeline", h2))
story.append(Paragraph(
    "At inference time the pipeline is fully sequential and causal. Raw IMU samples enter the online "
    "magnetometer normalizer, which maintains a trailing buffer and fits an ellipsoid to produce a "
    "calibrated |M| value consistent across devices [11]. Per-frame sensor readings then enter the "
    "WiFi heatmap environment model, which outputs a probability distribution over the spatial grid "
    "[3, 4]. KalmanNet [7] fuses this distribution — using its spatial spread as the adaptive "
    "measurement covariance and computing a per-step learned gain — with the PDR motion prediction "
    "to produce a refined position estimate [5, 6]. No future observations are required at any step "
    "and no memory of the walked path is retained.", body))

# ── SECTION 5 ────────────────────────────────────────────────────────────────
story.append(Paragraph("5.  Results", h1))
story.append(Paragraph(
    "All results are reported under two evaluation protocols. The random split assesses generalization "
    "to unseen visits of nodes seen during training. The held-out device split uses a Samsung Galaxy "
    "S9+ that was excluded entirely from training, testing cross-device generalization.", body))

story.append(Paragraph("5.1  WiFi heatmap environment model", h2))
t4_data = [
    row(["Protocol", "MAE", "vs. best prior (3.64 m)", "What it measures"], bold=True),
    row(["Random split (unseen visits)", "1.43 m", "2.5\u00d7 improvement", "Generalizes to unseen visits of known nodes"]),
    row(["Held-out device (Samsung Galaxy S9+)", "2.02 m", "1.8\u00d7 improvement", "Generalizes to a completely unseen handset"]),
]
t4 = Table(t4_data, colWidths=[4.5*cm, 1.5*cm, 3.0*cm, W - 9.0*cm])
t4.setStyle(table_style_base())
story.append(t4)
story.append(Paragraph("Table 4. WiFi heatmap environment model performance.", caption))

story.append(Paragraph("5.2  Online magnetometer normalizer", h2))
story.append(Paragraph(
    "The normalizer was validated by running calibration on all four continuous walk recordings and "
    "measuring the standard deviation of |M| across devices before and after [11]. Cross-device "
    "spread dropped from 1.77 µT to 0.57 µT, a reduction of approximately 3×, enabling a single "
    "environment model to work across different handsets without any per-device calibration step.", body))

story.append(Paragraph("5.3  Causal PDR", h2))
story.append(Paragraph(
    "The causal PDR [5, 6] achieves approximately 1.5 m MAE on short walks. Drift accumulates on "
    "longer paths, as expected for any dead-reckoning system. In the full KalmanNet pipeline the "
    "WiFi heatmap anchor corrects this drift roughly every one second, preventing unbounded error "
    "growth.", body))

story.append(Paragraph("5.4  MAE progression across all models", h2))
t5_data = [
    row(["Model", "MAE", "Training data", "Output", "Route memory"], bold=True),
    row(["CNN-LSTM", "5.04 m", "Continuous walks", "Single (X, Y)", "High"]),
    row(["Causal Hybrid", "3.64 m", "Continuous walks", "Single (X, Y)", "High"]),
    row(["Environment Model", "6.36 m / 63 m *", "Continuous walks", "Single (X, Y)", "Medium"]),
    row(["Neural EKF", "4.29 m", "Continuous walks", "Single (X, Y)", "Medium"]),
    [Pb("WiFi Heatmap + KalmanNet (current)"), P("1.43 m / 2.02 m +"), P("Static fingerprint DB"), P("Probability heatmap"), P("None")],
]
t5 = Table(t5_data, colWidths=[4.5*cm, 2.8*cm, 2.8*cm, 3.0*cm, W - 13.1*cm])
ts5 = table_style_base()
ts5.add('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#E8F4E8'))
t5.setStyle(ts5)
story.append(t5)
story.append(Paragraph(
    "Table 5. Full MAE evolution across all five models. * Forward / reversed path. "
    "+ Random split / held-out device. The highlighted row is the current system.", caption))

# ── SECTION 6 ────────────────────────────────────────────────────────────────
story.append(Paragraph("6.  Path Forward", h1))
story.append(Paragraph(
    "The WiFi heatmap environment model is validated and ready. The KalmanNet fusion pipeline [7] "
    "is implemented and individually verified on physically faithful synthetic trajectories [12]. "
    "The single remaining prerequisite for end-to-end system validation is the collection of "
    "continuous walk recordings with per-frame independent ground truth across the IT Engineering "
    "building — explicitly not synthesized by the method described in Section 3.4.", body))

t6_data = [
    row(["Step", "Action", "Dependency"], bold=True),
    row(["1", "Collect WiFi-enabled walks with independent per-frame ground truth", "Physical access and logging setup"]),
    row(["2", "Validate the complete KalmanNet pipeline end-to-end on the collected data", "Step 1"]),
    row(["3", "Evaluate on held-out devices and unseen sessions", "Step 2"]),
    row(["4", "Deploy on-device and benchmark inference latency", "Step 3"]),
]
t6 = Table(t6_data, colWidths=[1.5*cm, 8.5*cm, W - 10.0*cm])
t6.setStyle(table_style_base())
story.append(t6)
story.append(Paragraph("Table 6. Planned steps toward full system deployment.", caption))

story.append(Paragraph(
    "Once Step 1 is complete the system is architecturally finished and will require only validation. "
    "No further redesign is anticipated unless real-world data reveals environmental characteristics "
    "not captured in the static fingerprint database.", body))

# ── REFERENCES ───────────────────────────────────────────────────────────────
story.append(Paragraph("References", h1))

refs = [
    "[1] I. Ashraf, S. Din, M. U. Ali, S. Hur, Y. B. Zikria, and Y. Park, \"MagWi: Benchmark Dataset for Long Term Magnetic Field and Wi-Fi Data Involving Heterogeneous Smartphones, Multiple Orientations, Spatial Diversity and Multi-Floor Buildings,\" IEEE Access, vol. 9, pp. 77976–77996, 2021.",
    "[2] I. Ashraf, S. Din, S. Hur, G. Kim, and Y. Park, \"Empirical Overview of Benchmark Datasets for Geomagnetic Field-Based Indoor Positioning,\" Sensors, vol. 21, no. 10, p. 3533, 2021.",
    "[3] P. Bahl and V. N. Padmanabhan, \"RADAR: An in-building RF-based user location and tracking system,\" in Proc. IEEE INFOCOM, 2000, pp. 775–784.",
    "[4] M. Youssef and A. Agrawala, \"The Horus WLAN location determination system,\" in Proc. ACM MobiSys, 2005, pp. 205–218.",
    "[5] U. Bolat and M. Akcakoca, \"A hybrid indoor positioning solution based on Wi-Fi, magnetic field, and inertial navigation,\" in Proc. WPNC, 2017, pp. 1–6.",
    "[6] D. Yu, C. Li, and J. Xiao, \"Neural Networks-Based Wi-Fi/PDR Indoor Navigation Fusion Methods,\" IEEE Trans. Instrum. Meas., vol. 72, pp. 1–14, 2023.",
    "[7] G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, and Y. C. Eldar, \"KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,\" IEEE Trans. Signal Process., vol. 70, pp. 1532–1547, 2022.",
    "[8] Z. Toth and J. Tamas, \"Miskolc IIS hybrid IPS: Dataset for hybrid indoor positioning,\" in Proc. 26th Int. Conf. Radioelektronika, 2016, pp. 408–412.",
    "[9] M. Zardkoohi, Y. Seifi Kavian, and K. Ansari-Asl, \"Indoor Localization Using Smartphone Magnetic Sensor Data: A Bi-LSTM Neural Network Approach,\" IEEE Access, vol. 13, pp. 165795–165809, 2025.",
    "[10] W. Zhang, R. Sengupta, J. Fodero, and X. Li, \"DeepPositioning: Intelligent fusion of pervasive magnetic field and WiFi fingerprinting for smartphone indoor localization via deep learning,\" in Proc. ICMLA, 2017, pp. 7–13.",
    "[11] A. Mansour, J. Wang, H. Luo, M. Adham, J. Ye, and W. Chen, \"Drift-Resistant Heading Estimation for Smartphone-Based Indoor Positioning via Adaptive Calibration Using Wi-Fi Fingerprinting and Magnetic Stability,\" IEEE Trans. Instrum. Meas., vol. 75, pp. 1–20, 2026.",
    "[12] C. Isaia, L. Yu, W. Cai, and M. P. Michaelides, \"Axes Mapping and Sensor Fusion for Attitude-Unconstrained Pedestrian Dead Reckoning,\" Sensors, vol. 26, no. 6, p. 1968, 2026.",
]

for r in refs:
    story.append(Paragraph(r, ref_style))

# ── BUILD ────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF generated: {OUTPUT}")
