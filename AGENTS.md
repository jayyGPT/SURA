# Repository working rules

These rules apply to human and agent-assisted changes.

## Canonical locations

- Reusable Python code: `src/sura/`
- User-facing commands: `sura` / `python -m sura`
- Checked-in configurations: `configs/`
- Local datasets: `data/` or `SURA_DATA_ROOT`
- Raw run artifacts and checkpoints: ignored `experiments/runs/`
- Verified result summaries: `experiments/results/`
- Manuscript source: `paper/`
- Historical material: `archive/`

Do not create a second active copy of a model, training loop, preprocessing pipeline, or paper
file elsewhere. Thin workflow wrappers may live under `scripts/`, but they must import the
package and must not duplicate implementation.

## Research integrity

- Never insert an unverified metric into the paper.
- Every headline result must be traceable to a configuration, seed, code revision, and
  machine-readable result file.
- Do not describe a model component in the paper unless the evaluated implementation actually
  uses it.
- Synthetic-trajectory results must always be labelled as synthetic and must not be presented
  as measured real-walk ground truth.
- Preserve the Wi-Fi-only baseline when modifying dual fusion so the contribution of the
  magnetic channel remains measurable.

## Data and commands

- Resolve data through a command argument, `SURA_DATA_ROOT`, or `data/`; never hard-code a
  machine-specific absolute path.
- Raw data belongs below `data/raw/`, processed fingerprints below `data/processed/`, and
  generated run outputs below `experiments/runs/`.
- Add user-facing workflows to `sura.cli` and document them in `COMMANDS.md`.
- Every training command must provide a `--dry-run` preflight and write its effective
  configuration, seed, Git commit, metrics, and outputs to the run directory.

## Paper changes

- Main document: `paper/main.tex`
- Bibliography: `paper/Ref.bib`
- Compile with `python -m sura paper build` before committing paper changes.
- Treat `paper/reviews/` as reviewer input and `paper/notes/` as internal project notes.
- Do not commit LaTeX auxiliary files from `paper/build/`.

## Code changes

- New model behavior must be implemented on a feature branch.
- Keep preprocessing identical between training and inference.
- Prefer configuration values over hard-coded constants and paths.
- Add or update smoke tests for commands, tensor shapes, causality, masks, and numerical
  finiteness.
- Do not edit archived implementations unless a task explicitly concerns historical
  reconstruction.

## CNN magnetic fusion milestone

The active research direction is to replace scalar anomaly-gradient fusion with the magnetic
sequence CNN's two-dimensional position estimate and predicted uncertainty. The anomaly
implementation remains available only as a reproducibility baseline during that transition.
