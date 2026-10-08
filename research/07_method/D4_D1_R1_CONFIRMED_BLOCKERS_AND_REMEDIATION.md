# D4-D1R — Confirmed findings and corrective patch

## Authority
- Parent: `c9370aa730977e462cad721a8b1f058a2a4fa85d`.
- Branch: `research/vgra-impl-01`.
- D4-D1 R1 read-only audit confirmed BOTH integration blockers with a clean worktree.

## Blocker A — End-of-training best-checkpoint validation
`BaseTrainer.final_eval()` passes `validator(model=best_path)`, without a trainer.
The VGRA validator intentionally rejects this metadata-free API.

**Remediation:** The VGRA trainer now owns `final_eval()`, restores selected BEST
VGRA model through repository `load_checkpoint`, verifies TRAIN-state weights,
and calls its validator with BOTH `trainer=self` and `model=selected`.
The validator explicitly accepts this trusted model override while preserving
pair-aware prediction, native decoding/NMS and metrics. Native source untouched.

No silent native-only end evaluation or substitution with latest EMA is allowed.
Checkpoint save/load consistency remains for D4-D2/D3 independent verification.

## Blocker B — Unreadable frozen B-VAL member
`VGRAYOLODataset` now loads the frozen C2 assignment rows BEFORE the inherited
image discovery/YOLO annotation cache. Its `get_img_files()` removes the single
`EXCLUDED_UNREADABLE` image before any YOLO label scan. It fails closed if any
operational image is missing, unknown or duplicated. Post-label-scan cardinality
must match the entire intended operational split.

`bind_vgra_assignments()` also rejects any excluded image if upstream filtering
is bypassed. The readable companion remains a true single-view fallback.
C2 manifest data and C4 pair-sampler mathematics remain unchanged.

## Additional related safeguards
- A missing B-VAL path cannot silently fall back to sealed B-TEST.
- CPU non-distributed `world_size=0` is accepted in validation; DDP still fails.
- Independent synthetic tests are appended and all prior CPU regressions rerun.

## Open, not waived
- D4-D1 independent re-verification NOT YET COMPLETE.
- C5C R1 all-zero image gradient failure NOT RESOLVED.
- Signed-beta semantic polarity NOT RESOLVED.
- D4-D2 checkpoint identity, D4-D3 integrated numerical verification LOCKED.
- No GPU training, raw image/label access, or B-TEST access authorized.

## R1 documentation-only safe stop
- D4-D1R R1 script SHA256:
  `592EBE67642AD5F49A76CBF9159273FDA1CCD814BE8B9B3E118996BA18E1C372`.
- R1 passed all 75 CPU tests before failing a next-action tracker
  precondition: the master tracker has a Markdown heading plus a code line,
  whereas the execution tracker uses an inline label.
- R1 made NO commit/push; `PRECOMMIT_ROLLBACK_CLEAN=TRUE`.
- R2 retains identical VGRA module repairs and test cases; it checks both
  tracker layouts before source mutation and preserves each tracker format.

## R2 preflight safe stop
- R2 script SHA256:
  `987346565A7E54B76669BF16886771460DCF313444CF9B382A620D0DCDEBED8E`.
- R2 stopped before source mutation while checking the raw SHA256 of
  `ultralytics/data/vgra.py`, after R1 had restored tracked files.
- `PREFLIGHT_ABORT_WITH_NO_MODIFICATIONS=TRUE`; R2 made NO commit or push.
- R3 tolerates LF/CRLF worktree representation ONLY when normalized canonical
  source bytes and the independent pinned Git blob ID both match.
- R3 also corrects the doubled-escape newlines in R2 tracker preconditions.
- VGRA replacement source and all seven synthetic tests are still
  byte-identical to the R1/R2 embedded implementations.

## R3 documentation-only safe stop
- R3 script SHA256:
  `3286BD30B265A72AA1BFF3774008EBBBE2D1847EC4DE1163BAC710A95024DB37`.
- The frozen Git blob and canonical line-ending checks passed.
- The focused CPU regression suite passed **75/75** tests.
- R3 then failed in the tracker documentation update with
  `ValueError: illegal newline value: 
` because the Python source passed a
  literal backslash-n (`newline="\n"`) rather than LF (`newline="
"`).
- R3 made NO commit/push and reported `PRECOMMIT_ROLLBACK_CLEAN=TRUE`.
- R4 preserves the same frozen source repairs and synthetic test code and
  corrects only the transaction newline argument, adding this failure record.

## Next
`D4-D1B_VGRA_INDEPENDENT_REVERIFICATION`
