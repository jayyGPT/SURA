# IEEE paper

The current manuscript is:

```text
paper/main.tex
```

The bibliography is `Ref.bib`.

## Build

From this folder:

```bash
latexmk -pdf -outdir=build main.tex
```

The generated PDF is `build/main.pdf`. Build files are ignored by Git.

## Other folders

- `reviews/` — professor comments and feedback
- `notes/` — revision notes and previous paper/code checks
- `figures/` — paper figures and replacements
- `template/` — IEEE template material

The current paper still contains the old magnetic-anomaly fusion description. We will update the
method, equations, figures, and results only after the new DualKalmanNet is implemented and tested
with the magnetic CNN's 2-D position output and uncertainty.

For Overleaf, use `paper/main.tex` as the main document.
