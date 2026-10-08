# D4-C5A VGRA Visibility Supervision Contract

## Status

`COMPLETE / SYNTHETIC_CPU_VERIFIED`

Parent:
`a79f53ae5e332b7dbe2036114f8edde02a64a198`

## Reason C5 is divided

A complete C5 transaction would simultaneously touch target semantics, paired
runtime model dispatch, native loss aggregation, trainer and validator.
It is divided into:
- D4-C5A: target construction, TRAIN-only state counts, weighted loss;
- D4-C5B: paired/single model runtime dispatch and native loss integration;
- D4-C5C: trainer/validator integration and synthetic end-to-end checks.

No D4-C5 completion or GPU-training authorization is claimed by C5A.

## Added modules

- `ultralytics/models/yolo/detect/vgra_targets.py`
- `research/tests/test_d4_c5a_vgra_targets.py`

No existing Ultralytics source file was modified.

## Label provenance

The nine class-presence bits are computed from ordinary YOLO
`cls`/`batch_idx` ground-truth annotations in the collated batch.

The target formula is the frozen D4-B2C2 rule:

`state = AP_present + 2 * LAT_present`

Values:
- 0: neither;
- 1: AP-only;
- 2: LAT-only;
- 3: both.

These represent **class-level annotation presence**, not cross-view matching
of individual lesions.

Single-view images do not generate a visibility target or auxiliary loss.

The batch companion contract requires reciprocal indices, an AP/LAT pair,
matching nonempty pair IDs, and explicit single-view sentinels.

## Train-only weights

`train_state_counts_from_dataset()` operates on in-memory `.labels` of an
explicit `vgra_split="train"` dataset and rejects a validation dataset.

It derives class/state frequencies from exact TRAIN pairs only and uses
the already-frozen normalized inverse-square-root weighting.

**No real dataset is instantiated and no real labels are opened in C5A.**
Actual B-TRAIN frequency values/hashes will be generated and independently
verified during C5B/C5C training-stack preparation.

## Frozen loss

`L_vis = mean(pair,class)[ w[class,state] * CE(logits, state) ]`

`lambda_vis=0.25`

No empty-pair hallucination. No validation-derived training weights.
The native detection loss is not modified in C5A.

## OPEN SCIENTIFIC SIGN GATE — must close before training

The frozen V1 strength equation is:

`beta_l = 2*tanh(rho_l)`

It has exact zero initialization and bounded magnitude, but beta can
become NEGATIVE.

For negative beta, shared-view `k>0` can suppress, and view-exclusive
`k<0` can reinforce. Therefore the sign of the *effective residual* is
not guaranteed to follow the intended positive-assist/negative-suppress
narrative.

This issue is NOT silently patched: V1 code and mathematical freeze stay
unchanged. D4-D must test the sign behavior and obtain a governed explicit
decision before the source freeze/training authorization. Potential options:
accept a signed corrective residual with appropriately limited interpretation,
or formally revise the candidate mechanism under a new documented spec.

## Tests

Synthetic CPU tests include:
- class presence from YOLO GT only;
- two-pair + single visibility targets;
- all four states;
- reciprocal pair validation;
- missing/invalid partner rejection;
- fractional/out-of-range class rejection;
- weighted CE exact reference;
- loss backpropagation;
- differentiable zero loss for zero paired samples;
- TRAIN-only state count/weight normalization and VAL exclusion;
- negative-beta semantic inversion regression check.

The existing C1/C2/C3/C4 and TPSC regressions run again.

## Firewalls

- B-TEST access: NONE.
- Raw images: NONE.
- Raw YOLO label files: NONE.
- GPU training: NOT AUTHORIZED.
- Trainer/validator integrated: FALSE.
- Native detection loss modified: FALSE.

## Next

`D4-C5B_VGRA_PAIRED_RUNTIME_AND_LOSS_INTEGRATION`
