# CORRECTED-MODULE-FAMILY-CLOSURE-01

## Status

`CORRECTED_MODULE_FAMILY_SCREEN_CLOSED=TRUE`

`SELECTED_FAMILY_CANDIDATE=YOLO11s + SCConv-Early`

`GLOBAL_FINAL_PAPER_ARCHITECTURE_FROZEN=FALSE`

`REPLACEMENT_RESEARCH_PENDING_RESIDUAL_DIAGNOSIS=TRUE`

`NEW_GPU_TRAINING_AUTHORIZED=FALSE`

`TEST_ACCESS=NONE`

## Frozen development condition

- Split-B Original
- `DATA01:B-ORG:v1`
- pretrained `yolo11s.pt`
- seed 42
- 100 epochs
- image size 1024
- global batch 16
- SGD
- primary selection metric: validation mAP50-95
- Split-B test access: NONE

## Corrected single-module screen

| Candidate | mAP50-95 | Delta vs baseline | Disposition |
|---|---:|---:|---|
| YOLO11s baseline | 0.41704 | - | reference |
| SCConv-Early | 0.43130 | +0.01426 | selected family candidate |
| SCConv-4Stage | 0.42587 | +0.00883 | positive, not selected |
| DySample | 0.41525 | -0.00179 | do not promote |
| Canonical EMA | 0.42796 | +0.01092 | positive single module, not family winner |

SCConv-Early is the strongest corrected candidate on the frozen primary metric.

## Controlled combination screen

Candidate:

`YOLO11s + SCConv-Early + Canonical EMA`

Authoritative result:

- best epoch: 47
- precision: 0.61407
- recall: 0.63296
- F1: 0.6233719272
- mAP50: 0.64357
- mAP50-95: 0.41839
- delta vs baseline: +0.00135
- delta vs SCConv-Early: -0.01291
- delta vs Canonical EMA: -0.00957

The predeclared promotion requirement was strict improvement over
SCConv-Early mAP50-95 = 0.43130.

Therefore:

`STRICT_COMPLEMENTARITY=FALSE`

`SCIENTIFIC_DECISION=DO_NOT_PROMOTE_COMBINATION_PRIMARY_METRIC`

## Interpretation guardrail

The result does not mean Canonical EMA failed as an isolated module.
Canonical EMA was individually positive.

The evidence establishes only that Canonical EMA did not provide positive
primary-metric complementarity when combined with SCConv-Early under the
frozen development condition.

No causal mechanism such as feature competition, over-suppression,
redundant recalibration, or destructive interaction is frozen at this stage.

Those remain hypotheses until prediction-level residual analysis is
performed.

## Scientific consequence

The corrected SCConv/DySample/EMA family is now closed.

This is NOT a global architecture freeze.

SCConv-Early is carried forward as the family reference because it is the
strongest corrected candidate, but the final paper architecture remains open.

The next phase is the Residual-Error + Novelty Decision Gate.

That phase must determine:

1. what errors the baseline makes;
2. which errors SCConv-Early fixes;
3. which errors Canonical EMA fixes;
4. which gains are lost by the combination;
5. what errors remain after SCConv-Early;
6. whether those residual errors justify a new replacement mechanism.

No new GPU training is authorized by this closure.
Split-B test remains sealed.

## Expanded Residual Diagnostic Scope

The next diagnostic covers the complete corrected module family rather
than only the SCConv-Early + Canonical EMA interaction.

Required models:

- native YOLO11s baseline;
- SCConv-Early;
- SCConv-4Stage as a placement/intensity control;
- DySample;
- Canonical EMA;
- SCConv-Early + Canonical EMA.

### DySample observation

Under the frozen Split-B Original / pretrained / seed42 / E100 condition:

| Metric | Baseline | DySample | Delta |
|---|---:|---:|---:|
| Precision | 0.68900 | 0.68413 | -0.00487 |
| Recall | 0.61491 | 0.60415 | -0.01076 |
| F1 | 0.649850 | 0.641657 | -0.008193 |
| mAP50 | 0.65301 | 0.64235 | -0.01066 |
| mAP50-95 | 0.41704 | 0.41525 | -0.00179 |

DySample is therefore not promoted from the seed-42 corrected screen.

However, the small negative mAP50-95 delta is not accepted as proof that
DySample is intrinsically harmful.

The Residual-Error + Novelty Decision Gate must determine whether the
observed behavior is associated with:

- particular classes;
- small/subtle object scales;
- confidence ranking;
- false-positive or false-negative behavior;
- localization / IoU behavior;
- cross-scale feature alignment;
- dynamic sampling behavior;
- or the absence of a meaningful upsampling bottleneck.

No alternative upsampler or attention mechanism is authorized before this
diagnostic.

If the problem is genuinely feature reconstruction / scale alignment, the
replacement search must remain in the upsampling or guided-fusion family
first.

If the evidence instead shows that upsampling is not the limiting factor,
the project may retire the upsampling branch rather than force a
replacement.

### Historical combination evidence

No clean corrected SCConv-Early + DySample experiment has yet been run.

No clean corrected SCConv-Early + DySample + Canonical EMA experiment has
yet been run.

Historical V1/V2/V3 SC+Dy and three-module results remain useful only for
hypothesis generation because later implementation and transfer confounds
prevent clean causal attribution.
