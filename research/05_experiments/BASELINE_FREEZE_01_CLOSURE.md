# BASELINE-FREEZE-01 Final Closure

## Status

`COMPLETE`

BASELINE-FREEZE-01 is closed after archive ingestion, canonical-run reconciliation,
600-epoch convergence review, and standardized validation-only checkpoint revalidation.

- 11/11 execution/session archives ingested.
- 6/6 canonical baseline runs completed at 100 epochs.
- 600 canonical epoch rows frozen.
- 6/6 frozen `best.pt` checkpoints revalidated on validation only.
- 54 per-class rows frozen.
- `foreignbody` has zero validation support and is recorded as N/A, not AP=0.
- no training occurred during freeze analysis.
- no test predictions or metrics were generated.

`EXPERIMENTS.csv` stores the training-time selection metrics from the canonical
`results.csv` row maximizing validation mAP50-95. Standardized FP32 checkpoint
revalidation is retained separately and does not replace selection metrics.

The standardized full-output archive is kept outside Git at
`artifacts_external/baseline_freeze_01/standardized_validation/` with SHA256
`15004b6c14219f654d7b4e8e34e62b1b483b5ba127a15f8d208826ebbefa2d14`.

The canonical `BASE-B-ORG-SCR-S42` 100-epoch run remains immutable. Because its best
epoch is 100 and late-window mAP50-95 trends remain positive, the next separately
governed diagnostic is `BASE-B-ORG-SCR-S42-E200-CAL`.

Split-A experiments remain reporting-only comparators. Split-B test remains sealed.
