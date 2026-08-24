# Pre-final re-audit status

This is the final disposition of the R1-R13 evidence-first audit. An item is marked resolved only where the active code, manuscript, and canonical artifact agree.

- [x] **R1 - untouched final test.** Seed 2 is development. Seed 3 was retired after the database-registration issue was found. The corrected frozen protocol uses seed 4 exactly once for the final 60-trajectory result set.
- [x] **R2 - causal simulated heading.** `causal_path_heading()` uses current/previous positions only; future-path changes do not alter earlier headings. No centered smoothing or `np.gradient` remains.
- [x] **R3 - metric semantics/P90 baselines.** Mean/CI are trajectory-level; median/P90/max/CDF are pointwise. The paper names each compared model explicitly and uses the corrected pointwise statistics.
- [x] **R4 - CI scope.** The paper states `1.96 * SD(per-trajectory mean error) / sqrt(60)` for one fixed model and explicitly excludes retraining variability.
- [x] **R5 - KNN preprocessing leakage.** Magnetic log-uncertainty fill/reference values are fitted on training trajectories and then reused unchanged for final data.
- [x] **R6 - graph wording.** The paper and result metadata describe a Euclidean survey-node proximity graph and explicitly state that no wall/obstacle geometry is used.
- [x] **R7 - scheduler wording.** Wi-Fi uses Adam at a fixed learning rate; only the magnetic CNN additionally uses `ReduceLROnPlateau`.
- [x] **R8 - weak standalone Wi-Fi table.** The old weakly-provenanced standalone table was removed. The selected Wi-Fi checkpoint now has a canonical corrected-pairing development run and preserved artifacts.
- [x] **R9 - S9+/AP-vocabulary overclaim.** The paper no longer presents the fusion result as held-out-device evaluation. The active Wi-Fi checkpoint is trained on the common corrected survey database; no S9+-only headline result is claimed.
- [x] **R10 - magnetic seed 200.** Seed 200 is explicitly development and no standalone magnetic final-test number is reported.
- [x] **R11 - bibliography closure.** Unused entries were removed after the final citation pass.
- [x] **R12 - failed PR #18 transport files.** The cleaned active tree contains no temporary audit workflow, base64 payload, or patch-transport helper.
- [x] **R13 - repository cleanup.** Required checkpoints/results are promoted into active locations; superseded experiment trees, old reports, local literature copies, notes, caches, and temporary helpers are excluded from the cleaned tree.

## Additional database-registration issue found during implementation

The historical builder paired Wi-Fi to magnetic visits by basename, which is ambiguous across phones. The corrected builder requires an exact normalized mode/scenario/phone/user match plus the timestamped Wi-Fi filename. The magnetic static coordinate is the canonical map coordinate; raw Wi-Fi coordinates are retained separately in `nodes.csv` and summarized in `pairing_audit.json`. This correction was frozen before the selected Wi-Fi checkpoint and final seed-4 fusion evaluation were produced.

## Remaining non-blocking scope limitation

Causal alignment of an uncalibrated unseen handset into the per-phone-centered magnetic survey domain remains future work. The present paper does not claim that capability.
