# D4 LPQ Scientific Rationale

## Evidence basis

This phase follows the governed A12-D3 execution and the read-only A12-D3-E1
evidence extraction.

Evidence package SHA256:
`d75afd61c0549c6ae805a57b364e926bc7750bfb46fec520e69002976b147cb9`

## What the corrected architecture screen established

Standardized D2 mAP50-95:
- YOLO11s baseline: 0.41765881
- SCConv-Early: 0.42588415
- SCConv-4Stage: 0.42452888
- DySample: 0.41427802
- Canonical EMA: 0.42663323
- SCConv-Early + Canonical EMA: 0.41308765

The single modules can shift performance, but the combination does not establish
complementarity.

## What D3 added

At confidence 0.25, SCConv-Early versus baseline:
- precision delta: approximately +0.0055;
- recall delta: approximately -0.0001;
- F1 delta: approximately +0.0027;
- TP delta: -1;
- FP delta: -44;
- FN delta: +1.

Error changes at the same operating point included:
- background FP reduction;
- localization FP reduction;
- duplicate FP increase.

Same-GT object transitions showed approximately balanced recovery and loss:
- baseline FN -> Early TP: 100;
- baseline TP -> Early FN: 101.

Therefore the main Early benefit is not a robust net increase in lesion recovery.

At confidence 0.50 Early loses substantial TP relative to baseline, showing
confidence-sensitive sensitivity degradation.

## Scientific interpretation

The evidence supports a residual problem at the prediction-quality / candidate-competition
level:

1. useful false-positive suppression should be retained;
2. true lesions should not become under-confident or be lost at stricter operating points;
3. localization confidence should better reflect localization reliability;
4. multiple high-confidence candidates for one lesion should be separated more clearly.

This does not prove that a specific LPQ formulation is correct.

## Working D4 hypothesis

`DAR` explores whether information already present in YOLO11's localization-distribution
output can support a better reliability signal.

`GDS` explores whether one-to-many supervision can be retained while enforcing clearer
score separation among candidates assigned to the same GT.

Both remain hypothesis-level until D4-B closes.

## Why not immediately add another attention or upsampling module

D3 did not demonstrate that upsampling is the dominant unresolved bottleneck.
SCConv/EMA/DySample combinations were not reliably complementary.
Therefore D4 begins at the demonstrated residual rather than at a module catalog.

An attention mechanism may be reconsidered only after LPQ diagnostics demonstrate a
separate residual feature-representation problem.
