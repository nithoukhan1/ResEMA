# BASELINE-FREEZE-01A Review

## Status

`CORE_ARTIFACT_INGESTION_PASS`

The uploaded review package was independently cross-checked after local ingestion.

- review package SHA256: `748aa6292e4923f1566df671ca82556554ed3c6b623d34a903d04b09e47de158`
- review package bytes: `18,087`
- registered ZIP/session archives: `11`
- unique ZIP/session archives: `11`
- canonical final experiments expected: `6`
- canonical final experiments passed: `6`
- resume-parent sessions preserved: `5`
- training started during freeze: `false`
- test predictions generated during freeze: `false`
- test metrics generated during freeze: `false`

Cross-file checks passed:

- `SHA256SUMS.txt` agrees with `ZIP_ARCHIVES.csv` for all 11 ZIPs.
- `CANONICAL_FINAL_SUMMARY.csv` agrees with matching canonical rows in `SESSION_SUMMARY.csv`.
- all six canonical experiments have unique experiment IDs.
- all six canonical runs are classified `CANONICAL_FINAL_PASS`.
- S2/A2 lineage is preserved for B-ORG and A-AUG.
- S3/A3 lineage is preserved for B-AUG.
- resume preflight/runtime records are paired for every resumed final run.
- every canonical run records a closed test firewall.

## Canonical validation-selected metrics

| Experiment | Init | Best epoch | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE-B-ORG-SCR-S42 | scratch | 100 | 0.69276 | 0.63409 | 0.662128 | 0.63611 | 0.40360 |
| BASE-B-ORG-PT-S42 | pretrained | 44 | 0.68900 | 0.61491 | 0.649850 | 0.65301 | 0.41704 |
| BASE-A-AUG-PT-S42 | pretrained | 49 | 0.63660 | 0.63661 | 0.636605 | 0.62635 | 0.41036 |
| BASE-A-AUG-SCR-S42 | scratch | 64 | 0.65639 | 0.58853 | 0.620610 | 0.61498 | 0.39072 |
| BASE-B-AUG-PT-S42 | pretrained | 33 | 0.66991 | 0.65989 | 0.664862 | 0.67162 | 0.43133 |
| BASE-B-AUG-SCR-S42 | scratch | 94 | 0.70871 | 0.62963 | 0.666834 | 0.64174 | 0.40492 |

These are the rows that maximize validation mAP50-95 in each canonical `results.csv`; P/R/F1/mAP50 are taken from the same selected row. F1 is derived as `2PR/(P+R)`.

## Remaining BASELINE-FREEZE-01A work

The artifact ingestion pass is not yet the final BASELINE-FREEZE-01 closure.

Before repository synchronization:

1. generate full-trajectory convergence diagnostics from all six canonical `results.csv` files;
2. resolve the `BASE-B-ORG-SCR-S42` epoch-100 boundary question from the full curve rather than the single best-epoch fact;
3. freeze standardized validation-only per-class P/R/AP50/AP50-95 and support for all six `best.pt` checkpoints;
4. then update `EXPERIMENTS.csv`, `ARTIFACTS.csv`, six compact records, `CURRENT.md`, and `DECISIONS.md`.

Four canonical final archives contain a final-validation-summary sidecar. The two B-AUG final archives do not. This is recorded as an evidence-shape difference, not a canonical-run failure.

Split-B test remains sealed.
