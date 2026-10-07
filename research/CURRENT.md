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

`PROJECT_MIGRATION_AFTER_A12_D2_CLOSURE`

A12-D2 standardized six-model validation is complete, fully preserved,
registered and closed.

Governed evidence:

- execution checkout: `791566cb04c91258eadbbb8ee6120edac9cc4c0f`;
- frozen source commit: `758f0643e2bceb8d996d8e9603fb836d9319d8eb`;
- frozen runner SHA256: `9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250`;
- authorization SHA256: `8cbb1e999d3ffb2e7e4b2e30d9882c5510c076d116468609962cd06e36b3dcb1`;
- canonical preservation archive SHA256:
  `419ac71b1e42b791168a5ea24cf87d71b8a3d1eba9333d183a73009733056b93`;
- preservation review SHA256:
  `9805009b0fb757bcd3e84ef7ec28b471cced7da0b9a700d3ad56cac6932a84bd`;
- global artifact manifest: 18,407 verified rows;
- data binding: `DATA01:B-ORG:v1`;
- frozen validation: 3,050 images / 914 patients;
- operational validation: 3,049 readable images;
- GT boxes: 7,113 frozen / 7,110 operational / 3 on unreadable image;
- training: NONE;
- Split-B test access: NONE.

Standardized diagnostic revalidation mAP50-95:

- Baseline: 0.41765881
- SCConv-Early: 0.42588415
- SCConv-4Stage: 0.42452888
- DySample: 0.41427802
- Canonical EMA: 0.42663323
- SCConv-Early + Canonical EMA: 0.41308765

This standardized diagnostic metric family does not replace the historical
training-time selection metric family and does not by itself change the
selected family candidate.

The one governed D2 authorization is consumed by the completed execution.
Any rerun requires a new explicit authorization transaction.

Offline diagnostics have not yet been executed.
New GPU training remains unauthorized.
Split-B test remains sealed.

After the migration/archive handoff, the next scientific transaction is:

`A12-D3_OFFLINE_DIAGNOSTIC_IMPLEMENTATION_AND_FREEZE`

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

## A12-D3 offline diagnostic source freeze — 2026-10-07

Status:

`SOURCE_FROZEN_PENDING_EXECUTION_AUTHORIZATION`

Scientific source-freeze commit:

`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

Frozen D3 implementation:
- primary matcher: global same-class prediction/GT candidate pairs ordered by descending IoU;
- primary match threshold: IoU >= 0.50;
- frozen confidence thresholds: 0.05 / 0.10 / 0.25 / 0.50;
- frozen unmatched-prediction taxonomy: duplicate -> class_confusion -> localization -> background -> other_overlap;
- frozen comparison lattice: 8 predefined contrasts;
- patient bootstrap: seed 42 / 10,000 replicates / two-sided 95% percentile interval;
- engine synthetic regressions: 13/13 PASS;
- runner synthetic regressions: 20/20 PASS;
- D2 preservation archive read during implementation/source freeze: NONE;
- checkpoint loading: NONE;
- validation rerun: NONE;
- training: NONE;
- Split-B test access: NONE.

Source-freeze record:

`research/06_diagnostics/A12_D3_SOURCE_FREEZE.json`

SHA256:

`24b34050764342d0ac2c2398ffdb80aedc59a4c151206765b203d01361ba4e19`

Execution is **not authorized** by this source freeze.

Next governed transaction:

`A12_D3_OFFLINE_DIAGNOSTIC_EXECUTION_AUTHORIZATION`
