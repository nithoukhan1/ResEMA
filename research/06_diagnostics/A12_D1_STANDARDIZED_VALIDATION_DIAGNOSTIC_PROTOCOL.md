# A12-D1 Standardized Validation Diagnostic Protocol

## Status

`IMPLEMENTATION_CANDIDATE_PENDING_SOURCE_FREEZE`

## Purpose

A12-D1 is the pre-execution gate for a common validation-only diagnostic
evaluation of the six frozen Split-B Original pretrained checkpoints.

The goal is not to choose another module by aggregate validation score alone.
The goal is to generate one common set of outputs from which the project can
diagnose why the corrected modules differ and what residual bottleneck remains.

## Six-model diagnostic set

1. `BASE-B-ORG-PT-S42` — frozen YOLO11s baseline
2. `BORG-PT-S42-SCCONV-EARLY-E100` — SCConv-Early
3. `BORG-PT-S42-SCCONV-4STAGE-E100` — SCConv-4Stage
4. `BORG-PT-S42-DYSAMPLE-E100` — DySample
5. `BORG-PT-S42-CANONICAL-EMA-E100` — Canonical EMA
6. `BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100` —
   SCConv-Early + Canonical EMA

All six use the same frozen primary development condition:

- data binding: `DATA01:B-ORG:v1`
- initialization family: pretrained
- training seed: 42
- selection split: Split-B validation
- Split-B test access: NONE

## Metric-family distinction

Training-time selection metrics remain frozen historical evidence.

A12-D2 will produce a second metric family:

`standardized diagnostic revalidation metrics`

These are computed from the six frozen `best.pt` checkpoints under one current
evaluation checkout and one common validation protocol.

They do not replace the historical training-time selection metrics.

Any numerical difference between the training-time selected row and the
standardized revalidation result must be preserved and reported.

## Common runtime contract

Reuse the already governed BASELINE-FREEZE-01 standardized validation runtime:

- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- Ultralytics: 8.4.7
- expected GPU inventory: Tesla T4, Tesla T4
- evaluation device: GPU 0
- imgsz: 1024
- batch: 16
- workers: 4
- rect: false
- validation confidence: Ultralytics default (`None -> 0.001`)
- NMS IoU: 0.7
- max_det: 300
- half: false
- plots: true
- save_json: false
- seed: 42
- deterministic: true
- split: val

This is controlled reproducibility, not a claim of cross-hardware bitwise
determinism.

## Single evaluation environment

All six checkpoints must be loaded by one source-frozen current repository
checkout.

The preflight must reject:

- an unexpected Git HEAD;
- a dirty evaluation checkout;
- an Ultralytics import outside that checkout;
- runtime version drift;
- GPU-inventory drift;
- checkpoint SHA drift;
- parameter-count drift;
- corrected-module signature drift;
- dataset-binding drift;
- any runtime YAML containing a `test` key.

Historical training source commits remain recorded separately and are not
rewritten.

## Checkpoint discovery

The canonical checkpoint identity is the SHA256 of `best.pt` registered in
`research/01_provenance/ARTIFACTS.csv`.

A12-D1 discovers checkpoints under `/kaggle/input` by exact SHA256, not by
filename or dataset title alone.

Every required SHA256 must resolve exactly once.

## Dataset firewall

Only `DATA01:B-ORG:v1` is permitted.

Frozen validation membership:

- images: 3,050
- patients: 914
- membership SHA256:
  `f36040d4a798cbda113909907ba21e3dac1bb91b268124ec51c4fb2899e4a816`

Known operational unreadable validation image:

`1502_0635264266_05_WRI-R2_M015.png`

Expected operational readable validation images:

`3049`

The frozen membership remains 3,050.

Runtime dataset discovery must prune test-like directories.

The runtime YAML may contain `train` and `val` only. It must not contain
`test`.

No test prediction, metric, label-content analysis, error analysis, or model
selection is allowed.

## A12-D1 preflight scope

A12-D1 may:

- verify repository/runtime identity;
- verify the six checkpoint hashes;
- load each checkpoint without validation inference;
- verify `nc=9`;
- verify parameter counts;
- verify expected corrected-module signatures;
- verify train/validation membership identity;
- generate a train+validation-only runtime YAML;
- write an A12-D1 preflight manifest.

A12-D1 must not:

- train;
- run model validation;
- run model prediction;
- access Split-B test outcomes;
- modify model weights;
- authorize a replacement architecture.

A successful D1 preflight still requires review before A12-D2 execution.

## A12-D2 planned validation outputs

For each of the six checkpoints, A12-D2 will run one standardized
validation-only pass and preserve:

- aggregate precision;
- aggregate recall;
- F1 from aggregate P/R;
- mAP50;
- mAP50-95;
- per-class precision/recall/F1/AP50/AP75/AP50-95;
- class support;
- confusion matrix;
- normalized confusion matrix;
- P/R/F1/PR curves;
- post-NMS prediction text files with confidence;
- validation artifact manifest;
- runtime/provenance summary.

`foreignbody` has zero Split-B validation support and must remain N/A rather
than being reported as AP=0.

## Prediction-export contract

A12-D2 will preserve post-NMS validation detections at the Ultralytics
validation default confidence floor (`0.001`) using `save_txt=True` and
`save_conf=True`.

This low-confidence export is intentional. It permits all later diagnostic
thresholds to be applied offline to the same frozen prediction set without
rerunning a model.

No diagnostic conclusion may be based on a prediction threshold that was
chosen after seeing which threshold makes a particular model look best.

## Frozen offline diagnostic thresholds

Confidence thresholds:

- 0.05
- 0.10
- 0.25
- 0.50

Localization thresholds:

- IoU 0.50
- IoU 0.75

Ground-truth normalized box-area strata:

- small: area < 0.01 of image area
- medium: 0.01 <= area < 0.05
- large: area >= 0.05

Continuous normalized box area must also be retained so the fixed strata do
not replace the underlying geometry.

## Frozen one-to-one matching rule

Within each image and class:

1. form prediction/ground-truth IoU pairs;
2. sort candidate pairs by descending IoU;
3. greedily assign one-to-one matches at IoU >= 0.50;
4. assigned pairs are true positives;
5. unmatched ground truths are false negatives.

Unmatched predictions are then classified in this order:

1. duplicate — same-class IoU >= 0.50 to an already matched ground truth;
2. class confusion — different-class ground truth IoU >= 0.50;
3. localization error — same-class ground truth IoU >= 0.10 and < 0.50;
4. background false positive — maximum IoU with any ground truth < 0.10;
5. other-overlap false positive — remaining unmatched predictions.

The taxonomy is diagnostic. It does not replace Ultralytics' official AP
calculation.

## Frozen comparative questions

### SCConv-Early vs baseline

Identify:

- GT objects converted from FN to TP;
- GT objects converted from TP to FN;
- class-level precision/recall shifts;
- size-stratified changes;
- localization-IoU changes;
- patient concentration of gains/losses.

### SCConv-4Stage vs SCConv-Early

Determine whether extending SCConv to all four backbone stages changes:

- recall versus precision;
- specific classes;
- object-size strata;
- localization;
- false-positive composition.

### DySample vs baseline

Determine whether the negative aggregate result is concentrated in:

- specific classes;
- small/medium/large objects;
- confidence/ranking;
- localization;
- duplicate predictions;
- background false positives;
- patient-specific failures.

Outcome-level evidence may show that DySample is especially weak for a scale
or localization regime. It does not, by itself, prove an internal
cross-scale-alignment mechanism.

A feature-level DySample investigation is justified only if the outcome-level
diagnostic points to a scale/localization bottleneck.

### Canonical EMA vs baseline

Determine which classes/cases/geometry regimes account for its positive
single-module result.

### SCConv-Early + Canonical EMA

Compare against both single modules and identify which previously corrected
objects/errors are lost or retained in the combination.

Do not describe the interaction as causal architectural interference unless
a separate mechanism-level test demonstrates that claim.

### Residual SCConv-Early errors

The residual profile determines the next research branch.

Only after this profile is frozen may the project decide whether the next
mechanism should target:

- upsampling;
- feature fusion/refinement;
- localization;
- rare-class behavior;
- confidence calibration;
- another demonstrated bottleneck.

## Patient-level linkage

Use the repository-authoritative:

`research/04_data/split_b/split_B_val.csv`

for `filestem -> patient_id` mapping.

Do not infer patient identity from filenames when authoritative metadata is
available.

## Advancement rule

A12-D1 PASS means only:

`READY_FOR_REVIEW_BEFORE_A12_D2`

It does not authorize new training.

It does not unseal Split-B test.

It does not select a replacement module.
