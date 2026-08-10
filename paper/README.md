# IEEE paper

The current manuscript is:

```text
paper/main.tex
```

The bibliography is `Ref.bib`.

## Build

From the `paper/` folder:

```powershell
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

Run twice so cross-references and labels resolve correctly. The generated PDF is `build/main.pdf`.

> **Note:** The README originally listed `latexmk -pdf -outdir=build main.tex` but `latexmk` requires Perl which is not installed. Use `pdflatex` directly as above. For Overleaf, no build command is needed — just use `paper/main.tex` as the main document.


## Other folders

- `reviews/` — professor comments and feedback
- `notes/` — revision notes and previous paper/code checks
- `figures/` — paper figures and replacements
- `template/` — IEEE template material

The current paper still contains the old magnetic-anomaly fusion description. We will update the
method, equations, figures, and results only after the new DualKalmanNet is implemented and tested
with the magnetic CNN's 2-D position output and uncertainty.

For Overleaf, use `paper/main.tex` as the main document.
