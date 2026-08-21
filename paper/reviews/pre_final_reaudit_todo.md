# Pre-Final Re-Audit TODO

This checklist records the issues reproduced in the evidence-first re-audit of the current paper/code state. These items are **not resolved merely because they are documented here**. Anything marked as a blocker must be fixed and revalidated before declaring the manuscript final.

The earlier professor/reviewer tracker remains the historical record of those review rounds. This file is the active pre-final checklist and is intended to absorb the additional partner-review issues before implementation.

## Priority A — publication blockers

### [ ] R1. Replace the seed-2 evaluation set with a genuinely untouched final test set

**Finding.** Fusion training uses seed 1 and the reported 60-walk evaluation uses seed 2. However, the same seed-2 protocol was also used for magnetic-uncertainty analysis and the Wi-Fi-delta ablation that informed final design choices. Therefore seed 2 is best treated as a development/evaluation set, not an untouched final test set.

**Required action.** Freeze the current architecture/hyperparameters, designate seed 2 as development, generate a fresh untouched final set (for example seed 3), and evaluate it once without further design tuning. Rerun all paper-facing temporal baselines/variants on that same final set.

### [ ] R2. Make the simulated PDR heading strictly causal

**Finding.** `synthesize_walk()` currently forms the latent heading with `np.gradient(...)` and a centered 7-sample `np.convolve(..., mode="same")`. Interior gradients and centered smoothing use future trajectory samples, contradicting the manuscript's strict causal interpretation.

**Required action.** Replace the heading backbone with backward-only displacement, e.g. `atan2(y_n-y_{n-1}, x_n-x_{n-1})`, followed by no smoothing or a one-sided causal filter. Add a regression test that changing future trajectory samples cannot alter earlier heading samples. Regenerate all temporal fusion results affected by heading.

## Priority B — statistical/protocol correctness

### [ ] R3. Define and correct median/P90/CDF semantics

**Finding.** The active fusion evaluation first computes one MAE per walk and then reports median/P90/CDF over the 60 walk-level MAEs. The manuscript wording reads like ordinary pointwise localization-error percentiles. These are materially different statistics; in full Wi-Fi, the pointwise median comparison even changes direction relative to the median of walk MAEs.

**Required action.** Choose and state the convention explicitly. Preferred final reporting: mean and CI over per-walk MAE for trajectory-level uncertainty, while median/P90/max/CDF use all pointwise localization errors. Regenerate affected table values, discussion text, and CDF figures from the frozen final test set.

### [ ] R4. Define what the reported ± / 95% CI actually measures

**Finding.** The current half-width is `1.96 * SD(per-walk MAE) / sqrt(60)`. It measures variation across the 60 synthetic walks for one trained model and does not include retraining/model-seed variability.

**Required action.** State the CI definition explicitly in the paper. Do not imply it is a full model uncertainty interval. If practical, add repeated fixed-seed retraining to quantify training variability; otherwise list retraining variability as unmeasured.

### [ ] R5. Remove aggregate test information from trajectory-KNN preprocessing

**Finding.** The trajectory-KNN feature construction computes a magnetic log-uncertainty fill/reference statistic separately on the test dataset. This leaks an aggregate test statistic into test preprocessing, even though a mask is also supplied.

**Required action.** Compute all fill/reference/scaling statistics on training data only and pass them unchanged into development/final transformations. Rerun trajectory-KNN metrics and figures.

### [ ] R6. Describe the synthetic path graph as a survey-node proximity graph, not a verified corridor/floor-plan model

**Finding.** Graph edges are created from surveyed coordinates within a Euclidean epsilon threshold. No wall geometry, obstacle test, corridor labels, or manually validated edge map is encoded. The exact trajectory hash guard also detects only identical binned trajectories, not shared/reversed/near-duplicate routes.

**Required action.** Replace stronger wording such as "corridor graph", "map-constrained corridor path", or "surveyed route" where it implies obstacle-aware geometry. Use precise wording such as "survey-node proximity graph" / "survey-graph-constrained synthetic trajectories" and describe what the overlap guard does and does not guarantee.

## Priority C — manuscript/code consistency and provenance

### [ ] R7. Correct the Wi-Fi training scheduler statement

**Finding.** The paper says both the Wi-Fi MLP and magnetic CNN use `ReduceLROnPlateau`. The magnetic trainer does; the active and historical Wi-Fi heatmap trainers do not.

**Required action.** Split the optimizer/training-details sentence so the scheduler applies only to the magnetic CNN unless a reproduced Wi-Fi experiment actually uses one.

### [ ] R8. Reproduce the standalone Wi-Fi table with complete provenance

**Finding.** The manuscript's 1.43 m random-split and 2.02 m S9+ held-out values exist in historical records, but `benchmarks/results.yaml` labels them `legacy_pending_reproduction`. The repository does not currently provide a clean chain from exact source commit + split indices + checkpoint + predictions to those table numbers.

**Required action.** Reproduce both Wi-Fi protocols under a documented frozen implementation, preserve split indices/checkpoint/predictions/metrics/source commit, and update Table I only from those canonical artifacts. Do not claim that the historical checkpoint was test-selected unless direct evidence establishes that; the re-audit did not establish that claim.

### [ ] R9. Qualify the statement that S9+ is "used only for evaluation"

**Finding.** S9+ rows are excluded from Wi-Fi parameter fitting/static-KNN fitting, but the processed environment AP vocabulary is built globally before the phone split and therefore can include AP identities observed by S9+. This may be acceptable as fixed survey metadata, but "used only for evaluation" is literally stronger than the implementation.

**Required action.** Either rebuild the vocabulary from training phones only for the held-out-phone experiment or explicitly state that S9+ fingerprint rows are excluded from parameter fitting while the common surveyed-environment AP vocabulary is fixed globally. Keep this claim separate from the fusion experiment, which is not an unseen-device evaluation.

### [ ] R10. Treat magnetic seed 200 as development unless an untouched standalone magnetic final set is added

**Finding.** The magnetic sequence trainer uses seed 200 for scheduler control, epoch selection, and window-length comparison. Calling it development is scientifically fine; presenting it as an untouched final test would not be.

**Required action.** Keep seed 200 explicitly labelled development. For any paper-facing standalone magnetic generalization number, add a frozen unseen final set (e.g. seed 201) and evaluate the selected 84-frame model once.

### [ ] R11. Remove unused bibliography entries after the final citation pass

**Finding.** The re-audit identified currently unused bibliography entries including `driftresistant`, `miskolc`, `rizk2023globloc`, and `wang2024gnn`.

**Required action.** Recheck citations after all partner-review edits and remove entries that remain unused. Do not delete a reference that becomes necessary for a new substantiated claim.

## Priority D — repository integrity / cleanup

### [ ] R12. Revert the unfulfilled PR #18 temporary transport/audit files

**Finding.** PR #18 was merged even though its correction workflow failed on a payload checksum mismatch. Its actual diff added only temporary workflow/helper/base64 transport files and did not apply the claimed scientific corrections or cleanup.

**Required action.** Delete the merged temporary files (`tmp-pre-final-audit`, `tmp-protocol-corrections`, `tmp_apply_protocol_corrections.py`, `tmp_pre_final_audit.py`, and `tools/tmp_payload/*`) unless any are explicitly needed for a new reproducible workflow. Do not preserve the misleading PR #18 state as evidence that corrections were completed.

### [ ] R13. Perform repository cleanup only after corrected results are canonical

**Finding.** `archive/`, stale benchmark narratives/results, `agentConvoHist.md`, local reference copies, historical project documents, and duplicate/intermediate material remain in the active repository. Some archived checkpoints are still referenced by the active fusion code, so deleting first would break reproducibility.

**Required action.** First promote every genuinely required checkpoint/artifact into clearly named active locations with provenance/hashes and update active paths. Then remove only proven stale/duplicate material. The cleanup PR should have an obvious deletion-heavy diff and a written retained/deleted inventory; deleted history remains recoverable through Git.

## Required final validation before checking these items off

- [ ] Freeze all architecture and hyperparameter choices before creating the final test set.
- [ ] Re-run all affected paper-facing experiments on the corrected protocol.
- [ ] Verify every numerical claim in the manuscript against canonical machine-readable artifacts.
- [ ] Re-audit every labelled equation against the code after protocol edits.
- [ ] Compile and visually inspect the complete paper.
- [ ] Verify no temporary audit/transport files remain.
- [ ] Verify repository README, benchmark documentation, checkpoint paths, and manuscript all describe the same active system.
- [ ] Keep page-count compression deferred unless explicitly requested.

## Non-blocking future extension retained from the earlier tracker

### [ ] F1. Causal unseen-phone magnetic domain alignment

Develop and evaluate a causal procedure that maps an uncalibrated handset's live magnetic features into the survey-centered feature domain without position ground truth, separating handset bias from genuine spatial magnetic structure. This remains future work rather than a blocker for the current scoped study.
