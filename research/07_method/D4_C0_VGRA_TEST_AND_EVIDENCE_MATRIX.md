# D4-C0 VGRA Test and Evidence Matrix

## C1 core
Shapes; state mapping; AP/LAT sign truth table; stop-gradient; bounded residual; rho=0 identity; finite gradients.

## C2 pairing
Full B-TRAIN accounting; governed B-VAL accounting; pair counts; patient isolation; no duplicates; manifest hashes; no B-test.

## C3 head
Classification shape unchanged; box tensor unchanged; VGRA-off and zero-gate identity; companion permutation tests; singles unchanged.

## C4 batching
Pairs co-batched; singles allowed; deterministic seeded shuffle; no cross-study mosaic/mixup; labels remain correctly indexed after flatten/collate.

## C5 stack
Native detection loss retained; visibility CE only for valid pairs; lambda=0.25 exact; state weights TRAIN-only; validator produces normal image predictions; single-view fallback exact.

## D4-D hard gates
Early transfer; additive VGRA state only; total params <= 9,870,093; frozen numerical identity tolerance; box/DFL invariance; save/load; missing-view exact fallback.

## D4-F scientific metrics
Primary mAP50-95. Secondary mAP50/mAP75/P/R/F1. Mechanism evidence: visibility confusion/calibration, paired rescue/suppression, fracture-specific effects, D3-compatible FP taxonomy.

PROMOTE: delta mAP50-95 >= +0.0050 vs MV-00, F1@.25 >= control, recall@.25 >= control-0.0020, no material fracture regression, no severe high-confidence TP collapse, efficiency contract respected. HOLD: smaller positive result with meaningful mechanism benefit. REJECT: nonpositive strict AP without compensating benefit, material sensitivity loss, visibility collapse, instability, or budget violation.
