# A12-D3 Matching and Error-Taxonomy Contract — Source Frozen

Status: `SOURCE_FROZEN_PENDING_EXECUTION_AUTHORIZATION`

## Purpose

This file freezes the object-level matching and unmatched-prediction taxonomy that will be used by `A12-D3_OFFLINE_DIAGNOSTIC_IMPLEMENTATION_AND_FREEZE`. It is an offline evidence-analysis contract only. It does not authorize model inference, checkpoint loading, validation reruns, training, threshold tuning, or Split-B test access.

## Frozen evidence scope

D3 consumes only already-preserved A12-D2 outputs:
- `VALIDATION_IMAGE_INDEX.csv`;
- `VALIDATION_GROUND_TRUTH.csv`;
- six model-specific `PREDICTIONS.csv` files;
- standardized aggregate/per-class tables as contextual evidence.

Operational diagnostic scope remains:
- frozen validation membership: 3,050 images / 7,113 GT boxes;
- operational readable scope: 3,049 images / 914 patients / 7,110 GT boxes;
- known unreadable image and its 3 GT boxes are excluded from D3 denominators;
- patient identity comes only from the frozen validation image index.

## Frozen thresholds

Confidence thresholds: `0.05`, `0.10`, `0.25`, `0.50`.

Object matching:
- primary same-class match IoU: `>= 0.50`;
- stricter localization reporting: matched IoU `>= 0.75`;
- localization-error floor: same-class IoU `>= 0.10` and `< 0.50`.

GT normalized-area bins:
- small: `< 0.01`;
- medium: `[0.01, 0.05)`;
- large: `>= 0.05`.

Continuous normalized area is retained.

## Deterministic one-to-one matcher

At each frozen confidence threshold, independently for each image and class:
1. discard predictions below the threshold;
2. form every same-class prediction/ground-truth IoU pair;
3. sort candidate pairs by IoU descending;
4. greedily assign one-to-one pairs at IoU `>= 0.50`, skipping any pair whose prediction or GT is already assigned;
5. assigned pairs are true positives;
6. GT boxes left unmatched after the pass are false negatives;
7. retain matched IoU, prediction index, GT box index, class, confidence, area and patient binding.

### Deterministic equality tie-breaks added at implementation freeze

The controlling repository protocol freezes descending-IoU pair order but does not specify exact-equality tie-breaks. To make artifact regeneration deterministic without changing the scientific rule, equal-IoU candidate pairs are ordered by:
- `prediction_index` ascending;
- frozen GT `box_index` ascending;
- stable prediction-list position;
- stable GT-list position.

Prediction confidence is **not** a primary-match ordering key. It is preserved as evidence and is used only for threshold inclusion and later confidence-distribution analysis. These equality tie-breaks are implementation details only and must not be tuned from outcomes.

## Frozen unmatched-prediction taxonomy

After primary one-to-one matching, every unmatched prediction is assigned exactly one category using this priority:
1. `duplicate`: same-class IoU `>= 0.50` to a GT already matched by a higher-priority prediction;
2. `class_confusion`: different-class GT IoU `>= 0.50`;
3. `localization`: same-class GT IoU `>= 0.10` and `< 0.50`;
4. `background`: maximum IoU with any GT `< 0.10`;
5. `other_overlap`: all remaining unmatched predictions.

False negatives are operational GT boxes left unmatched after the one-to-one pass.

The taxonomy order is normative. Later categories must never override an earlier satisfied category.

## Size interpretation

For TP/FN analyses, size is always the frozen **GT** size bin.

For unmatched predictions, two different concepts must remain separate:
- prediction-size bin: computed from the prediction's normalized area;
- reference-GT size bin: recorded only when the taxonomy category has a reference GT (`duplicate`, `class_confusion`, `localization`, or `other_overlap`).

Background predictions have no reference-GT size bin. This prevents prediction area from being silently substituted for GT size in recall-oriented analyses.

## IoU>=0.75 reporting

The primary assignment remains the IoU `>=0.50` matcher. IoU `>=0.75` is a stricter localization-quality flag on those primary matches; it does not trigger a second, differently ordered matching pass.

## Diagnostic metric interpretation

Fixed-threshold TP/FP/FN, precision, recall and F1 produced by D3 are diagnostic operating-point statistics. They do **not** replace the historical training-time metric family or the standardized D2 AP/mAP family.

`foreignbody` has zero validation GT support. D3 may report prediction/error counts for that class, but support-dependent recall/F1/AP must remain `NO_VALIDATION_SUPPORT` / N/A rather than imputed as zero.

## Patient-level uncertainty freeze

Before any D3 execution, the uncertainty procedure is frozen as:
- independence/resampling unit: patient;
- bootstrap seed: `42`;
- bootstrap replicates: `10,000`;
- interval: two-sided 95% percentile interval;
- all model contrasts are paired on the same patient resample;
- bootstrap intervals are descriptive uncertainty evidence, not a post-hoc architecture-selection loophole.

No result-dependent threshold, resampling seed, replicate count, class grouping, or comparison may be introduced after execution.

## Firewalls

During C1/C2 implementation and tests:
- no preserved D2 archive is opened by the diagnostic engine tests;
- no checkpoints are loaded;
- no `model.val()` or equivalent inference is called;
- no GPU operation is required;
- no new training occurs;
- no Split-B test path is accessed;
- no architecture-selection decision is made.

This contract is source-frozen by the A12-D3 scientific source-freeze commit and remains pending a separate execution authorization.
