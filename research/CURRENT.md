# Current Project State

## Active branch

`research/baseline-refresh`

## Active phase

Baseline data and initialization freeze before fresh YOLO11s reverification.

## Framework closure

The baseline-refresh research framework is remotely closed and CI-verified.

- framework commit: `0a7b6304cc4bd2f064618f873304bd849f82e3c0`
- GitHub Research Integrity workflow: success
- GitHub workflow run ID: `35687683929`
- local framework validation before push: 33/33 research tests passed
- model source changed during framework setup: no

## Completed

- historical V1/V2/V3 reconstruction
- historical implementation audit
- C3k2_SC topology/semantic defect confirmation
- DySample reference audit
- ResEMA/EMA reassessment
- Split-B provenance/freeze
- EDA-00
- EDA-01
- EDA-02 to EDA-10
- baseline training-parameter review
- six-run baseline design
- Kaggle Save-Version/resume governance
- active experiment/artifact tracking framework
- baseline-refresh branch remote closure
- Research Integrity CI activation for `research/**`

## Current task

`DATA-01` - freeze exact active dataset bindings for:

1. Split A augmented
2. Split B original
3. Split B augmented

The freeze must record the exact Kaggle source/reference, runtime path, YAML identity, image/label counts, split membership evidence and relevant hashes without using test labels for development.

## Next

1. `DATA-01` - dataset bindings
2. `INIT-01` - freeze official YOLO11s pretrained checkpoint identity and SHA256
3. `TRAIN-01` - implement reusable governed baseline trainer
4. `RESUME-01` - implement Kaggle Save-Version/resume helper
5. launch `BASE-B-ORG-PT-S42`
6. register validation/per-class/convergence evidence
7. continue the remaining five seed-42 baseline conditions
8. perform fresh validation-only model diagnostics
9. freeze architecture and loss
10. implement the final method

## Active baseline experiments

- `BASE-B-ORG-PT-S42`
- `BASE-B-ORG-SCR-S42`
- `BASE-B-AUG-PT-S42`
- `BASE-B-AUG-SCR-S42`
- `BASE-A-AUG-PT-S42`
- `BASE-A-AUG-SCR-S42`

All remain `NOT_STARTED` until their required data/init/code bindings are committed.

## Test policy

Neither Split A test nor Split B test is used for architecture, loss, training-recipe, epoch-budget or model-selection decisions.
