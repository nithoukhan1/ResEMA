# A12-D3 Output Schema and Patient-Uncertainty Contract — Source Frozen

Status: `SOURCE_FROZEN_PENDING_EXECUTION_AUTHORIZATION`

## Scope

This contract extends the frozen A12-D3 matching/taxonomy contract without changing its scientific rules. C2 defines how already-loaded preserved D2 evidence will be summarized and how patient-level descriptive uncertainty will be computed after source freeze. C2 does not authorize D2 archive execution, checkpoint loading, validation reruns, training, test access, architecture promotion, or threshold tuning.

## Frozen model lattice

Exactly six preserved D2 models are analyzed at each frozen confidence threshold `0.05`, `0.10`, `0.25`, `0.50`:

1. `BASE-B-ORG-PT-S42` — YOLO11s baseline.
2. `BORG-PT-S42-SCCONV-EARLY-E100` — SCConv-Early.
3. `BORG-PT-S42-SCCONV-4STAGE-E100` — SCConv-4Stage.
4. `BORG-PT-S42-DYSAMPLE-E100` — DySample.
5. `BORG-PT-S42-CANONICAL-EMA-E100` — Canonical EMA.
6. `BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100` — SCConv-Early + Canonical EMA.

Class order is frozen as:
`boneanomaly, bonelesion, foreignbody, fracture, metal, periostealreaction, pronatorsign, softtissue, text`.


## Frozen comparison lattice

The minimum Chat12 comparison set is source-frozen before any D3 result is inspected. Every pairwise delta is reported as **comparison minus reference**:

1. baseline -> SCConv-Early;
2. baseline -> SCConv-4Stage;
3. baseline -> DySample;
4. baseline -> Canonical EMA;
5. baseline -> SCConv-Early + Canonical EMA;
6. SCConv-Early -> Canonical EMA;
7. SCConv-Early -> SCConv-4Stage;
8. SCConv-Early -> SCConv-Early + Canonical EMA.

No additional pairwise contrast may be elevated to a primary D3 decision contrast after viewing outcomes without a new governed amendment.

## Object-event outputs

For every model × confidence-threshold pair, D3 writes two auditable object-event tables:

- `events/<EXPERIMENT_ID>__CONF_<XX>__PREDICTION_EVENTS.csv`
- `events/<EXPERIMENT_ID>__CONF_<XX>__GT_EVENTS.csv`

Prediction events preserve TP or exactly one unmatched-prediction taxonomy label. GT events preserve TP/FN state, GT class, frozen GT area/size, patient binding, matched prediction index/confidence, match IoU and the IoU>=0.75 reporting flag.

Object-event files are evidence tables, not replacement AP/mAP metrics.

## Aggregate outputs

The execution will produce:

- `A12_D3_GLOBAL_SUMMARY.csv`
- `A12_D3_PER_CLASS_SUMMARY.csv`
- `A12_D3_GT_SIZE_SUMMARY.csv`
- `A12_D3_FP_TAXONOMY_SUMMARY.csv`
- `A12_D3_FP_PREDICTION_SIZE_SUMMARY.csv`
- `A12_D3_CONFIDENCE_SUMMARY.csv`
- `A12_D3_PATIENT_METRICS.csv`
- `A12_D3_PAIRED_PATIENT_CHANGES.csv`
- `A12_D3_PAIRED_PATIENT_BOOTSTRAP.csv`
- `A12_D3_EXECUTION_MANIFEST.json`

All summaries are fixed-confidence diagnostic operating-point statistics. They do not replace historical training-time AP/mAP or standardized D2 AP/mAP.

## Global summary

For each model × threshold, report:
- TP, FP, FN;
- precision, recall, F1;
- active prediction count and GT support;
- TP count with matched IoU >= 0.75 and its rate among TP;
- mean and median IoU among TP;
- counts for duplicate, class-confusion, localization, background and other-overlap FP categories.

## Per-class summary

For each of the nine frozen classes, report support, predictions, TP/FP/FN, fixed-threshold precision/recall/F1, IoU>=0.75 TP count/rate, and taxonomy counts.

`foreignbody` has zero frozen validation GT support. Its row remains `NO_VALIDATION_SUPPORT`; support-dependent precision/recall/F1 are emitted as N/A/blank rather than imputed as zero. Prediction/error counts may still be reported.

## Size summaries

Recall-oriented size analysis always uses frozen **GT size**:
- small `<0.01`;
- medium `[0.01,0.05)`;
- large `>=0.05`.

`A12_D3_GT_SIZE_SUMMARY.csv` reports GT support, TP/FN, recall and IoU>=0.75 TP rate by GT size.

FP size is a different concept and is kept separate in `A12_D3_FP_PREDICTION_SIZE_SUMMARY.csv`, using each unmatched prediction's own normalized area. Prediction size is never silently substituted for GT size.


## Confidence distributions

`A12_D3_CONFIDENCE_SUMMARY.csv` is emitted for every model x frozen threshold and for these predeclared event groups: `all_predictions`, `tp`, `all_fp`, `duplicate`, `class_confusion`, `localization`, `background`, and `other_overlap`. It reports count, minimum, 25th percentile, median, mean, 75th percentile, and maximum confidence. Empty groups remain empty/N/A rather than being assigned fabricated values.

The summary is descriptive and threshold-conditioned; it is not a calibration study and must not be used to tune a new confidence threshold.

## Patient-level metrics

The patient identifier comes only from the frozen validation image index. Every one of the 914 frozen validation patients is represented for every model × threshold, including patients with zero TP/FP/FN for a specific operating point.

Per-patient rows contain TP, FP, FN, precision, recall, F1, IoU>=0.75 TP count and IoU>=0.75 TP rate.

## Paired patient changes

`A12_D3_PAIRED_PATIENT_CHANGES.csv` contains one row for every frozen comparison x confidence threshold x patient. It preserves reference/comparison TP, FP, FN, precision, recall, F1 and IoU>=0.75 TP counts together with signed comparison-minus-reference deltas. This is the direct patient-level change table required by the Chat12 diagnostic plan.

## Paired patient bootstrap

The uncertainty procedure is frozen before D3 execution:

- resampling unit: patient;
- patient population: all 914 frozen validation patients;
- seed: `42`;
- replicates: `10,000`;
- interval: two-sided 95% percentile interval;
- comparisons: exactly the eight source-frozen pairwise contrasts above;
- delta direction: comparison minus reference;
- thresholds: all four frozen confidence thresholds;
- metrics: precision, recall, F1 and TP-IoU>=0.75 rate;
- pairing: the identical patient resample is used for baseline and candidate within every bootstrap replicate;
- within each sampled patient, all of that patient's operational-readable validation images and object events move together;
- bootstrap output is descriptive uncertainty evidence, not a new architecture-selection metric family and not a hypothesis-test p-value.

The implementation uses NumPy `default_rng(seed=42)` and records the runtime NumPy version during source/execution provenance. C2 does not claim cross-version bitwise bootstrap identity unless the runtime version is also fixed at the source-freeze/execution gate.

## Execution firewalls

Before source freeze, C2 tests operate only on synthetic in-memory fixtures. They do not open the preserved D2 ZIP. Actual D2 execution requires a later governed transaction after code review and source freeze.

At every stage:
- checkpoint loading = none;
- model inference = none;
- validation rerun = none;
- new training = none;
- Split-B test access = none;
- no threshold is selected or tuned from D3 outcomes;
- no architecture is promoted from C2 implementation tests.
