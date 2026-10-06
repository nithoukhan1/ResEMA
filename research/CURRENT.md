# Current Project State

## Active branch

`research/combination-screen-01`

## Active phase

`CORRECTED MODULE FAMILY CLOSED -> RESIDUAL-ERROR + NOVELTY DECISION GATE`

## Frozen baseline reference

Primary controlled pretrained reference:

`BASE-B-ORG-PT-S42`

- validation mAP50-95: 0.41704
- test access: NONE

## Corrected module-family results

| Candidate | validation mAP50-95 | Delta vs baseline |
|---|---:|---:|
| SCConv-Early | 0.43130 | +0.01426 |
| SCConv-4Stage | 0.42587 | +0.00883 |
| DySample | 0.41525 | -0.00179 |
| Canonical EMA | 0.42796 | +0.01092 |

Family winner:

`YOLO11s + SCConv-Early`

## Combination result

`YOLO11s + SCConv-Early + Canonical EMA`

- best epoch: 47
- precision: 0.61407
- recall: 0.63296
- F1: 0.6233719272
- mAP50: 0.64357
- mAP50-95: 0.41839
- delta vs baseline: +0.00135
- delta vs SCConv-Early: -0.01291
- strict complementarity: FALSE
- decision: DO_NOT_PROMOTE_COMBINATION_PRIMARY_METRIC

Execution archive:

`research/experiment-execution-archive`

Archive HEAD:

`3acd95ea046b9027c857f9068bb276ef4b2aa0ee`

## Controlling scientific decision

`CORRECTED_MODULE_FAMILY_SCREEN_CLOSED=TRUE`

`SELECTED_FAMILY_CANDIDATE=YOLO11s + SCConv-Early`

`GLOBAL_FINAL_PAPER_ARCHITECTURE_FROZEN=FALSE`

`REPLACEMENT_RESEARCH_PENDING_RESIDUAL_DIAGNOSIS=TRUE`

`NEW_GPU_TRAINING_AUTHORIZED=FALSE`

`TEST_ACCESS=NONE`

Canonical EMA remains an individually positive corrected ablation.
The combination result does not establish a causal explanation for its
negative interaction with SCConv-Early.

## Current task

A12-D1 standardized validation diagnostic preflight is complete and preserved.

Source-frozen D1 implementation:

`e2367057e3a4ffabcb5cde1c2ff569df264625ef`

Runtime preflight result:

- status: PASS;
- six canonical `best.pt` checkpoints resolved exactly once;
- all six checkpoints loaded structurally without validation inference;
- corrected-module signatures and parameter counts verified;
- data binding: `DATA01:B-ORG:v1`;
- frozen validation membership: 3,050 images / 914 patients;
- operational readable validation images: 3,049;
- runtime YAML contains train and val only;
- validation inference: NONE;
- prediction: NONE;
- training: NONE;
- Split-B test access: NONE.

Preserved runtime-preflight evidence:

- `A12_D1_PREFLIGHT.json` SHA256:
  `d33ead0f712aa432e4afdd67aa89f6a4481acc543ae0486b8424a95761909de8`;
- runtime YAML SHA256:
  `b2114397eefc1ac377352fb4e0429df0fe6fc954a307e4b0df89139b4bed82c4`;
- preservation manifest SHA256:
  `217d67b401afbeca76033a7eef381649ef36e4f0f3b829c76b47216a8cc6039d`;
- preservation ZIP SHA256:
  `f9f47b7816caec3556fd1700aee79f8e716a8efa866bb9030e571a947b4d9de8`.

Next:

`GOV_CHRONOLOGY_FORWARD_CORRECTION`

After that governance correction, proceed to the separately implemented and
source-frozen A12-D2 standardized validation execution gate.

A12-D2 remains unauthorized at this state.

Required comparison:

- frozen YOLO11s baseline;
- SCConv-Early;
- SCConv-4Stage as a placement/intensity control;
- DySample;
- Canonical EMA;
- SCConv-Early + Canonical EMA.

Required questions:

- what errors does SCConv-Early fix?
- why does SCConv-4Stage change the precision-recall behavior?
- why does DySample underperform the baseline in this frozen condition?
- does DySample fail by class, object size, confidence, localization,
  cross-scale alignment, or because upsampling is not the bottleneck?
- what errors does Canonical EMA fix?
- which fixes disappear in the SCConv-Early + Canonical EMA combination?
- what residual errors remain after SCConv-Early?
- should the next mechanism target upsampling, feature refinement,
  localization, rare-class behavior, or another demonstrated bottleneck?

Do not choose a replacement module before this diagnostic.

Do not start new GPU training.

Split-B test remains sealed.
