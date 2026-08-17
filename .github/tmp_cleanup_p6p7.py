from pathlib import Path
import re

paper_path = Path('paper/main.tex')
text = paper_path.read_text(encoding='utf-8')
text = text.replace('the relative-variance weight suppresses corrections', 'the relative-uncertainty weight suppresses corrections')
text = text.replace('the CNN-predicted magnetic variance supplies a relative confidence weight', 'the CNN-predicted magnetic uncertainty score supplies a relative confidence weight')
text = text.replace('Relative magnetic-variance weighting reduces mean error', 'Relative magnetic-uncertainty weighting reduces mean error')
paper_path.write_text(text, encoding='utf-8')

review_path = Path('paper/reviews/prof_read_ieee_comments_draft.md')
review = review_path.read_text(encoding='utf-8')
pattern = re.compile(r"### \[x\] P6\. Explain the CNN uncertainty output precisely \(observation 7\).*?(?=### \[x\] P7\.)", re.S)
replacement = '''### [x] P6. Explain the CNN uncertainty output precisely (observation 7)\n\n**Resolved finding.** The second head outputs one learned scalar `ell_mag` from the shared 128-D CNN representation. The active implementation defines a positive scale `q_mag = exp(ell_mag)` and trains with\n\n`0.5 * ||position_error||^2 / q_mag + 0.5 * ell_mag`\n\n(with a numerical floor on the denominator). Because this one scalar weights the **summed 2-D radial squared error** while the normalization penalty is only `0.5 * ell_mag`, the exact objective is **not** the full negative log-likelihood of a 2-D isotropic Gaussian. We therefore do not call `q_mag` a calibrated Cartesian variance or covariance. It is a learned relative uncertainty/difficulty scale. The existing calibration benchmark shows useful confidence ordering but a conservative absolute scale, so final fusion uses only the training-normalized score difference `ell_mag - ell_ref`.\n\n'''
review, count = pattern.subn(replacement, review, count=1)
if count != 1:
    raise SystemExit(f'P6 tracker replacement count={count}')
review = review.replace('- variance head: 128->32->1 with ReLU;\n- joint heteroscedastic NLL training.', '- scalar uncertainty head: 128->32->1 with ReLU;\n- joint uncertainty-weighted radial regression training.')
review = review.replace('split the median definition and `sigma_ref^2 = exp(ell_ref)` across lines', 'split the median definition and `q_ref = exp(ell_ref)` across lines')
review_path.write_text(review, encoding='utf-8')

# Preserve one additional numerical caveat in the detailed architecture audit.
doc_path = Path('docs/architecture/magnetic_sequence_cnn.md')
doc = doc_path.read_text(encoding='utf-8')
needle = 'This objective encourages the uncertainty head to assign a larger scale to examples with larger position residuals while the `+0.5*ell_mag` term penalizes making the scale arbitrarily large. The position and uncertainty heads share the CNN encoder, so the scale is predicted from the same 84-frame magnetic representation used for localization.\n'
extra = needle + '\nImplementation caveat: the historical code floors `exp(ell_mag)` only in the residual denominator, while the additive `0.5*ell_mag` term itself is not floored. The learned scores in the current calibration runs are far above that floor, but a future probabilistic retraining should use a consistently bounded parameterization if calibrated likelihood semantics are desired.\n'
if needle in doc and 'Implementation caveat:' not in doc:
    doc = doc.replace(needle, extra)
doc_path.write_text(doc, encoding='utf-8')
