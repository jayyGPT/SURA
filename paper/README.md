# IEEE paper

Active manuscript: `main.tex`; bibliography: `Ref.bib`.

A normal TeX installation can build with:

```bash
latexmk -pdf main.tex
```

The manuscript includes the final CDF directly from `../benchmarks/final_protocol/current_results/final_cdf.png` so the plotted result and canonical metrics share the same result directory.

`reviews/` contains the faculty-review tracker plus the final evidence, numerical-traceability, and cleanup audits.
