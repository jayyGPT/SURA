# KNN baselines

Classical fingerprinting baselines retained for comparison with learned environment models.

```text
knn/
├── src/              baseline implementations and utilities
├── legacy_results/   historical plots; not validated against the canonical pipeline
└── notes/             original proof-of-concept report
```

The code was moved and renamed for clarity but has not yet been ported to the canonical data/configuration interface. Do not copy its historical numbers into the manuscript. Rerun each baseline on the same split and evaluation protocol as the proposed model, then store reviewed outputs under `experiments/results/`.
