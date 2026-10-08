# D4-C5C VGRA Pair-Aware Trainer / Validator Integration

## Status

`COMPLETE / SYNTHETIC CPU VERIFIED`

Branch: `research/vgra-impl-01`
Parent: `81d44f4c6d6fdea101c6feaf3f42141ed52b99ae`

## Scope and code hygiene

Added:
- `ultralytics/models/yolo/detect/vgra_trainer.py`
- `ultralytics/models/yolo/detect/vgra_validator.py`
- `research/tests/test_d4_c5c_vgra_trainer_validator.py`

No existing Ultralytics source file changed.
No actual patient images or YOLO label files were read.
No real dataset training or clinical validation performed.

## Trainer architecture

Dedicated `VGRADetectionTrainer(DetectionTrainer)` provides:
- dedicated `VGRADetectionModel` construction;
- mandatory frozen C2 combined assignment manifest SHA;
- strict nine-class B-TRAIN and B-VAL split handling;
- pair-safe dataset construction (rect=False);
- indivisible AP/LAT batch sampler;
- source-checked visibility counts from TRAIN only;
- persistent TRAIN-only visibility-weight model buffers;
- four loss labels (box/class/DFL/vis);
- dedicated VGRA validation.

The inherited stock train loop remains unchanged and is **not authorized**
for dataset execution until D4-D and D4-E contracts close.

Unsupported options fail closed:
- compile=True;
- distributed training;
- fraction other than 1.0;
- multi-scale augmentation;
- auto batch or batch < 2;
- single-class remapping.

## Native training forward

The unmodified stock loop passes the whole `batch` into the model.
C5B `VGRADetectionModel` dispatches paired/single logic via
`forward_vgra_batch()` and returns four-component loss vector.
The stock trainer applies `.sum()` before backward.

## Pair-aware validation — critical policy

Stock `BaseValidator` invokes `model(batch["img"])`, discarding pair
metadata and inadvertently evaluating only the native single-view path.

VGRA cannot use that path.

`VGRADetectionValidator` instead:
1. uses the existing B-VAL pair-aware dataloader;
2. preprocesses the full batch with retained companion metadata;
3. calls `forward_vgra_batch(batch)` on the actual evaluation/EMA model;
4. uses native `Detect._inference` for standard box decode and sigmoid scores;
5. computes four-component VAL loss from that SAME paired raw output;
6. applies the unchanged standard YOLO NMS and detection metrics;
7. restores ordinary float weights after validation;
8. returns regular detection metrics plus four labeled VAL losses.

An unsupported standalone validation call **raises an error** instead
of silently returning single-view results. This standalone interface
requires a separate governed follow-up before final B-TEST evaluation.

## R1 safe-stop / R2 correction (no loss of negative evidence)

R1 source compilation passed, but the focused suite returned `66 passed, 2 failed`:

1. An all-zero synthetic image batch produced non-finite parameter gradients
   during the synthetic optimizer step. The forward loss was finite, but the
   specific autograd operation/parameters were not identified by R1.
2. The evaluation-mode native Detect method returned `(decoded, raw)`, while
   the test incorrectly indexed that tuple as a raw-prediction dictionary.

R1 stopped **before commit**, and its rollback reported
`PRECOMMIT_ROLLBACK_CLEAN=TRUE`.

R2 changes ONLY the synthetic test fixture/diagnostics: a fixed-seed,
nonconstant synthetic image batch, exact eval `(decoded, raw)` extraction,
per-parameter NaN/Inf gradient details on failure, and a mandatory post-step
finite-parameter check. No VGRA model, trainer or validator source logic is
changed between R1 and R2.

The all-zero-image gradient edge case remains an **OPEN D4-D3 NUMERICAL GATE**;
R2 success must not be interpreted as proving all-zero-input stability.
If realistic nonconstant synthetic inputs still produce non-finite gradients,
R2 must safe-stop for diagnosis rather than altering the assertion.

## CPU verification

Synthetic no-patient-data tests require:
- trainer/validator subclass contract;
- unsupported configuration fail-closed;
- native Early+VGRA model and trainable parameter-count identity;
- a synthetic mixed AP/LAT+single optimizer step;
- paired score changes and unpaired native-score/box invariance;
- actual VGRA decoding path;
- dedicated pair-aware synthetic B-VAL metrics + 4 loss items;
- rejection of missing TRAIN-derived visibility weights.

All earlier C1–C5B and historical TPSC regression tests rerun.

## OPEN GATES

1. The R1 all-zero synthetic-input non-finite-gradient edge case remains
   OPEN for D4-D3 numerical diagnosis; R2 verifies realistic seeded
   synthetic-input gradient and post-step parameter finiteness.
2. Signed-beta assist/suppress polarity remains unresolved.
3. Checkpoint transfer from pretrained Early to VGRA requires D4-D2 proof.
4. Fully governed synthetic epoch, EMA/save-load, data-source binding,
   and experiment framework need D4-D3 validation.
5. D4-E source freeze and D4-F GPU training remain unauthorized.
6. B-TEST remains entirely sealed.

## Next

`D4-C6_VGRA_IMPLEMENTATION_CLOSURE`
