# Current Project State

## Active branch

`research/baseline-refresh`

## Active phase

`BASELINE-FREEZE-01 COMPLETE -> EPOCH-CALIBRATION FOLLOW-UP`

All six seed-42 YOLO11s baseline experiments are complete and frozen.

| Experiment | Condition | Init | Best epoch | val mAP50-95 |
|---|---|---|---:|---:|
| BASE-B-ORG-PT-S42 | Split-B original | pretrained | 44 | 0.41704 |
| BASE-B-ORG-SCR-S42 | Split-B original | scratch | 100 | 0.40360 |
| BASE-B-AUG-PT-S42 | Split-B historical augmented | pretrained | 33 | 0.43133 |
| BASE-B-AUG-SCR-S42 | Split-B historical augmented | scratch | 94 | 0.40492 |
| BASE-A-AUG-PT-S42 | Split-A augmented | pretrained | 49 | 0.41036 |
| BASE-A-AUG-SCR-S42 | Split-A augmented | scratch | 64 | 0.39072 |

These are training-time selection metrics. Standardized FP32 validation-only
checkpoint metrics are stored separately.

Actual canonical execution lineages:
- B-ORG and A-AUG: `S2=46f40838c1c24a8ced77a2b868dfa0f7f1037f9c` -> `A2=9fe475175d3963a083d7afc29426f07d86c1887d`
- B-AUG: `S3=18e75338ae116beccf3f5e4a0481efede601026b` -> `A3=fe95e2d51c4d545111ae2fa7be70c8e1c8e77487`

BASELINE-FREEZE-01 evidence:
- 11/11 execution archives;
- 6/6 canonical finals;
- 600 convergence rows;
- 6/6 standardized validation-only passes;
- 54 per-class rows;
- zero-support `foreignbody` reported as N/A;
- heavy checkpoints/plots retained outside Git;
- Split-B test access: NONE.

## Current task

Register and execute `BASE-B-ORG-SCR-S42-E200-CAL` as a separate longer-budget
scratch calibration. The frozen 100-epoch baseline is not overwritten or resumed.

After calibration, return to the C3k2 + DySample + ResEMA line with a
transfer-preserving corrected C3k2/SC design, then controlled Split-B pretrained module
ablations and multi-seed finalist confirmation.

Split-A remains reporting-only. Split-B test remains sealed.

## Parallel architecture-correction workstream

A separate non-training branch is active:

`research/arch-corr-01b`

Purpose:
- forensically audit the historical C3k2_SC / ResEMA implementations;
- preserve the validated DySample operator;
- implement transfer-preserving Self-Calibrated C3k2 candidates;
- implement a canonical EMA candidate without shifting native YOLO layer indices;
- verify native state, official checkpoint transfer, zero-gate identity, gradients and parameter counts before any architecture training.

Architecture training remains **NOT AUTHORIZED** until:
1. dedicated GitHub runtime audit passes on the current architecture head;
2. exact-head local verification reproduces the same contracts;
3. ARCH-CORR-01 verification evidence and tracker are frozen in Git.

The E200 calibration continues independently on `research/baseline-refresh`.
