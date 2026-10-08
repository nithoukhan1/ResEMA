# D4-D1 — Formal Independent Static / Remediation Acceptance

## Repository provenance
- Branch: `research/vgra-impl-01`
- Repaired-source parent: `9499fe88bd8d6724aada58f8dd2c62b5274f6add`
- Independent verifier: `D4_D1B_INDEPENDENT_READONLY_REVERIFY_R1.py`
- Verifier SHA256: `f3612dce4519457835ae2ebb1c27fdf1e4d8209831262195337c95016ee92868`
- Captured independent verifier stdout/stderr: `research/07_method/D4_D1B_INDEPENDENT_REVERIFY_EVIDENCE.txt`
- Captured evidence SHA256: `7c37c98a9ff25eb148286002e2edbf0ea1c2b19dd389e5ef4281ee1c48b5759b`

## Accepted gate: `D4-D1=COMPLETE_INDEPENDENT_CPU_STATIC_VERIFIED`
Independent replay of the D4-D1B verification passed on the frozen D1R parent.
The complete 75-test CPU suite also passed within that independently executed verifier.

Accepted:
1. The final-evaluation best checkpoint is routed through the pair-aware
   validator, rather than native-only single-view validation.
2. The frozen one-image VAL unreadable exclusion is enforced before label scanning;
   its readable companion is retained as a single-view fallback.
3. The frozen VAL manifest accounts for 3,050 original entries,
   1 excluded member and 3,049 operational entries: 1,401 pairs and 247 singles.
4. A size-16 pair-preserving VAL sampler yields 191 batches; companions stay intact.
5. The explicit B-VAL-versus-B-TEST firewall and CPU world-size guard remain active.
6. The native box/DFL firewall, pair-only residuals and four-term synthetic loss
   retain their earlier passing focused CPU regressions.

**Scientific limitation:** The synthetic dummy image fixture prints detection
P/R/mAP values of zero. These are not real GRAZPEDWRI findings and are not
research-quality metrics or evidence of actual model accuracy.

## Unresolved risks deliberately carried forward
- **Signed-beta sign/interpretation:** OPEN, including whether negative beta
  inverts the intended assist-versus-suppress explanation; governed resolution
  required before source freeze/training.
- **All-zero-input gradient non-finiteness:** OPEN; D4-D3 must reproduce,
  localize failing parameters and determine whether numerical repair is required.
- **Pretrained SCConv-Early checkpoint transfer and zero-rho identity:** NOT
  independently proven; D4-D2 is the next authorized verification task.
- **Persistent sampler/InfiniteDataLoader, complete synthetic epoch,
  save/reload and EMA state:** NOT verified; D4-D3.
- **MV-00 vs MV-01 fully identical pair-safe training recipe:** NOT VERIFIED;
  must be demonstrated before experiment source freeze D4-E.
- **Actual TRAIN visibility counts/weights:** NOT computed from real labels.
- **Full D4-D verification and final architectural freeze:** NOT complete.
- **Standalone final B-TEST evaluation:** NOT authorized.

## Scope of permission
D4-D2 *read-only/pretrained checkpoint verification* is next. This is NOT
authorization for model training, B-TEST access, hyperparameter sweeps,
architecture expansion, or final source freeze.

## Failure history preserved
- D4-D1 formal acceptance R1 independently reran D1B and 75 tests successfully,
  but stopped before changing any repository files: its excessively strict
  start/end-anchored pytest summary parser missed a decorated console line.
  `PREFLIGHT_ABORT_NO_REPOSITORY_MODIFICATIONS=TRUE`. R2 retains every
  research gate and requires a normalized, independently observed 75-test
  pytest summary along with all original D1B PASS markers.
- D4-D1 first independent diagnostic found two production integration blockers.
- D4-D1R R1 had 75 passing CPU tests, then documentation wording mismatch.
- D4-D1R R2 stopped before edits on Windows raw checkout SHA mismatch.
- D4-D1R R3 passed 75 tests, then failed on literal-backslash newline argument.
- D4-D1R R4 closed repairs and pushed commit `9499fe88bd8d6724aada58f8dd2c62b5274f6add`.
- D4-D1B independently verified both repairs fixed and passed 75 CPU tests.
- This transaction reruns D4-D1B, records stdout/stderr SHA and closes ONLY D1.

## Next
`D4-D2_VGRA_PRETRAINED_TRANSFER_AND_ZERO_GATE_IDENTITY`
