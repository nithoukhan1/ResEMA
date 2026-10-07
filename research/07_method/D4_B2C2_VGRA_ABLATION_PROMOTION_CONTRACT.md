# D4-B2C2 VGRA Ablation and Promotion Contract

## Purpose

Freeze the minimum experiment logic before implementation/training.

## Stage 1 — implementation controls

No GPU training until all D4-D structural tests pass.

Required numerical controls:
1. Early base path unchanged when VGRA disabled.
2. VGRA paired path with `rho=0` equals base logits/boxes.
3. Missing-view fallback equals base path.
4. Box/DFL outputs are invariant to VGRA residual activation.
5. Visibility target mapping `00/10/01/11` is unit tested.
6. Target-specific coefficient signs are unit tested.
7. Residual bound never exceeds `|beta_l|`.
8. Parameter cap passes.

## Stage 2 — first controlled training comparison

Use the same:
- B-TRAIN;
- B-VAL;
- pretrained source;
- seed;
- image size;
- optimizer/schedule;
- epoch budget;
- evaluator.

First two D4 runs:

### D4-MV-00 — Early paired-pipeline control

- same pair-aware data loader;
- pair metadata available;
- VGRA modules disabled;
- each image trained with ordinary Early detection loss;
- establishes whether pair-aware batching/data plumbing changes the baseline.

### D4-MV-01 — Early + full VGRA

- exact D4-B2C2 specification;
- lambda_vis=0.25;
- r_d=32;
- d_pair=128;
- r_x=16;
- beta_max=2.0;
- rho initialized 0.

No hyperparameter sweep before these two complete.

## Stage 3 — first-screen decision

Compare D4-MV-01 to D4-MV-00.

### PROMOTE_TO_ABLATION

Require all:
1. strict mAP50-95 delta >= +0.0050;
2. F1 at confidence 0.25 >= control;
3. recall at confidence 0.25 >= control - 0.0020;
4. fracture AP/mAP component does not regress materially;
5. no catastrophic high-confidence TP loss analogous to the Early D3 failure;
6. parameter cap satisfied.

### HOLD_FOR_DIAGNOSTIC

Any of:
- strict mAP50-95 delta in [+0.0020,+0.0050);
- smaller AP gain with clear TP/FN improvement and no material FP/efficiency regression;
- fracture-specific improvement with neutral overall result.

A HOLD result authorizes diagnostics, not module shopping.

### REJECT_PRIMARY

Any of:
- strict mAP50-95 delta <= 0;
- meaningful recall/TP regression without compensating benefit;
- pair pipeline is unstable;
- visibility branch collapses to dominant states;
- parameter/compute contract violated.

## Stage 4 — attribution ablations if promoted/held

Minimum planned ablations:

### D4-MV-02 — visibility auxiliary only
- retain visibility predictor and `L_vis`;
- force residual beta=0;
- isolates representation effect of visibility supervision.

### D4-MV-03 — residual without four-state target semantics
- replace VGRA target-specific gate with a simple always-on/shared-assist control;
- tests whether gains come merely from adding companion information.

### D4-MV-04 — full VGRA
- canonical candidate.

Optional only if needed:
### D4-MV-05 — positive-assist-only
- use shared-state positive assistance;
- remove exclusive-state negative residual;
- determines whether suppressive exclusive-state term is helpful.

### D4-MV-06 — YOLO11s + VGRA without Early
- only after the VGRA mechanism itself is supported;
- tests dependence on the Early backbone adapter.

## Diagnostics required after a promising result

Reuse D3-style evidence where applicable:
- thresholded P/R/F1;
- TP/FP/FN;
- FP taxonomy;
- class-level changes;
- patient-level changes;
- bootstrap intervals;
- paired AP/LAT error states;
- fracture-specific paired rescue behavior;
- visibility-state confusion matrix;
- calibration of predicted four-state probabilities.

## Architecture freeze rule

D4-B2C2 freezes the candidate specification for implementation.

It does NOT freeze the global final paper architecture.

`GLOBAL_FINAL_PAPER_ARCHITECTURE_FROZEN = FALSE`

## Test firewall

Split-B test remains inaccessible throughout D4 development.

`B_TEST_ACCESS = NONE`
