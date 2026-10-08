# D4-C1 VGRA Core Mathematical Implementation

## Status
`COMPLETE / CPU-UNIT-VERIFIED`

Branch: `research/vgra-impl-01`
Parent: `e309d1c99d1e20677b08d342619c3c8ce1b168f0`

## R1 safe-stop history
`D4_C1_IMPLEMENT_VGRA_CORE_R1.py` successfully compiled the core source and passed all
17 focused CPU tests, then stopped before commit while updating documentation because
two tracker strings did not exactly match the frozen D4-C0 wording.

The pre-commit rollback reported:
`PRECOMMIT_ROLLBACK_CLEAN=TRUE`

Therefore R1 produced no repository mutation or commit. R2 changes tracker matching
and records this history; the VGRA mathematics and tests are unchanged.

## Scope
Implemented only the standalone frozen VGRA V1 mathematical primitives in:
`ultralytics/nn/modules/vgra.py`

No generic Ultralytics files were modified. No dataset, Detect, trainer, criterion,
validator, raw-data, inference, GPU-training or B-test work occurred.

## Implemented primitives
- visibility-state mapping and weighting;
- target-specific AP/LAT coefficients;
- per-view semantic descriptor;
- paired four-state visibility predictor;
- low-rank class-conditioned compatibility;
- bounded zero-initialized residual;
- standalone paired VGRA core.

## Frozen bindings
- states: neither / AP-only / LAT-only / both;
- k_AP = q11 - q01;
- k_LAT = q11 - q10;
- stop-gradient gate;
- descriptor rank 32;
- pair hidden 128;
- cross-view rank 16;
- beta_max 2.0;
- rho init 0.

## Reference core parameter count
For channels 128/256/512 and nc=9:
`102567`

This is below the frozen +300,000 cap. Final integrated total remains a D4-D2 gate.

## Tests
The focused CPU suite verifies state mapping, state weights, gate signs, shapes,
compatibility bounds, rho=0 identity, residual bounds, stop-gradient behavior,
visibility-loss trainability, finite backward and exact reference parameter count.
Existing TPSC tests are re-run.

## Next
`D4-C2_VGRA_PAIR_AND_VISIBILITY_MANIFEST`
