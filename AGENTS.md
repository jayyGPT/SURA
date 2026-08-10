# Repository working rules

These rules apply to human and agent-assisted changes.

## Canonical locations

- Reusable Python code: `src/sura/`
- Runnable data scripts: `scripts/data/`
- Runnable training scripts: `scripts/train/`
- Verified result summaries: `experiments/results/`
- Manuscript source: `paper/`
- Historical material: `archive/`

Do not create a second active copy of a model or paper file elsewhere.

## Research integrity

- Never insert an unverified metric into the paper.
- Every headline result must be traceable to a configuration, seed, code revision, and result file.
- Do not describe a component in the paper unless the evaluated implementation uses it.
- Label synthetic-trajectory results as synthetic.
- Preserve the Wi-Fi-only baseline when modifying dual fusion.

## Code and data

- Prefer direct scripts over adding another command framework.
- Default raw dataset: `data/raw/magwi/`.
- Generated processed data and training outputs stay ignored.
- Keep preprocessing identical between training and inference.
- Add or update tests for shapes, causality, masks, and numerical finiteness.
- Do not edit archived implementations unless reconstructing historical work.

## Paper

- Main document: `paper/main.tex`
- Bibliography: `paper/Ref.bib`
- Compile with `latexmk -pdf -outdir=build main.tex` from `paper/`.

## CNN magnetic fusion milestone

Replace scalar anomaly-gradient fusion with the magnetic sequence CNN's 2D position estimate
and predicted uncertainty. Keep the anomaly implementation only as a comparison baseline during
the transition.
