# EPOCH-CAL-01 — Split-B Original Scratch 200-Epoch Calibration

## Objective

Register one explicit follow-up to the frozen 100-epoch baseline:

`BASE-B-ORG-SCR-S42-E200-CAL`

The parent `BASE-B-ORG-SCR-S42` remains immutable and complete.

The calibration asks whether a **fresh** 200-epoch version of the same baseline
training recipe obtains a materially different validation optimum.

## Why this is a separate experiment

The 100-epoch run selected epoch 100 and retained positive late-window validation
mAP50-95 slopes. The frozen baseline protocol therefore triggered a longer-budget
diagnostic.

The diagnostic is **not** allowed to resume the completed 100-epoch checkpoint.

## Important scheduler semantics

"200 epochs" is not a literal pure extension of the first 100 epochs.

The frozen recipe uses `cos_lr=true`, and Ultralytics constructs the cosine schedule
from the configured total epoch count. Therefore a fresh 200-epoch run has a
200-epoch cosine horizon.

Also, `close_mosaic=10` means mosaic remains active until the final ten epochs; the
absolute closure point therefore shifts later under a 200-epoch budget.

`patience` is changed from 100 to 200 only to preserve the parent's practical
no-early-stop-within-budget behavior.

These effects are part of the budget calibration and must be acknowledged when
interpreting 100 vs 200.

## Frozen comparison

Preserved:
- YOLO11s
- Split-B original
- scratch initialization
- seed 42
- deterministic mode
- imgsz 1024
- global batch 16
- SGD and all optimizer hyperparameters
- loss weights
- augmentation settings
- validation protocol
- two-T4 runtime contract
- test firewall

Explicit changes:
- epochs: 100 -> 200
- patience: 100 -> 200
- output project directory: calibration-specific path

## Selection

Primary metric remains validation mAP50-95.

The calibration is validation-only for development. Split-B test remains sealed.

After completion, compare:
- best validation mAP50-95;
- best epoch;
- peak stability / rolling means;
- late convergence;
- standardized best.pt validation if needed.

No test outcome may determine the epoch budget.

## Resume

The 200-epoch run may span multiple Kaggle sessions. Resume must continue only from
the exact calibration partial `last.pt` with optimizer/scaler/EMA preserved and the
same execution commit. It must never use the finalized 100-epoch baseline as a parent.
