# IEEE manuscript workspace

`main.tex` is the canonical paper source and `Ref.bib` is its bibliography. Historical drafts and generated build products are preserved under `archive/legacy_publication/`.

## Build

```bash
latexmk -pdf -outdir=build main.tex
```

or:

```bash
make
```

The generated PDF and auxiliary files remain under ignored `paper/build/`.

## Directory roles

- `reviews/`: professor annotations and extracted feedback.
- `notes/`: revision requests, figure audit, and previous code-paper audits.
- `figures/`: supplementary or replacement publication figures not currently referenced by `main.tex`.
- `template/`: IEEE class documentation.

The four PNG files currently referenced by `main.tex` remain beside the main source so the relocated manuscript compiles without changing scientific content. When the CNN-based DualKalmanNet update is implemented, figures and section files will be modularized in the same paper revision.

## Important current limitation

The existing manuscript describes the magnetic sequence CNN but the legacy fusion equations use a scalar anomaly-gradient correction. This is documented in `notes/architecture_consistency.md`. Do not polish or strengthen the affected claims before the CNN-output fusion experiment is complete.

For Overleaf Git synchronization, set the project's main document to:

```text
paper/main.tex
```
