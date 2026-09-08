# Evaluation, Statistics and Generalization Plan V8

## Development primary endpoint
**Represented-class mAP50-95** on frozen operational Split-B validation.
- 8 represented held-out classes.
- foreignbody is explicitly N/A, never treated as AP=0.

## Required development secondary metrics
1. represented-class mAP50
2. Clinical-7 mAP50 and mAP50-95
3. Core-6 mAP50 and mAP50-95
4. per-class AP50 and AP50-95
5. fracture AP50 and AP50-95
6. precision and recall as secondary operating-point metrics
7. best epoch and best-to-final degradation
8. model parameters

## Diagnostic metrics
### Localization
- AP50
- AP75
- higher-IoU profile when technically valid
- matched true-positive IoU distribution

### Geometry
- performance by normalized box area
- aspect ratio bins
- class x geometry bins

### Projection
- AP/PA vs LAT vs other/unknown
- per-projection mAP and class AP

### Axis head
If APCF is implemented:
- mean absolute angular error in degrees
- median angular error
- 90th percentile angular error
- percentage within 5° and 10°

### Long-tail
- unique positive patients/class
- sampled exposures/patient/epoch
- maximum exposure ratio
- tail-class AP
- leave-one-positive-patient-out sensitivity for rare classes where feasible

## Clinical detection metrics for final locked model
In addition to mAP:
- fracture FROC: sensitivity versus false positives per image
- predeclared reporting points such as 0.25, 0.5, 1, and 2 FP/image if supported by prediction density
- fracture image-level PR curve
- image-level fracture sensitivity/precision/F1 at a threshold selected only on validation
- optional study-level analysis if study grouping is verified before test

mAP remains the primary benchmark metric; FROC is a clinically interpretable complementary metric.

## Statistical uncertainty
### Final Split-B test
Use patient-clustered bootstrap:
- resample patients with replacement;
- include all images belonging to each sampled patient;
- recompute model metrics;
- 2,000 bootstrap replicates by default;
- report 95% percentile or BCa CI if implementation is validated.

For baseline-vs-proposed comparison:
- paired patient bootstrap using identical sampled patient sets;
- report 95% CI for delta mAP50-95, delta mAP50, and fracture AP.

Rare-class CIs will be wide and must be reported honestly.

## Cross-validation
3-fold grouped patient-level CV after method lock:
- same folds for baseline and proposed;
- report each fold;
- mean ± SD;
- paired fold deltas;
- avoid significance claims based only on n=3 folds.

## Calibration
Optional but desirable after model lock:
- reliability diagram for fracture confidence;
- detection ECE only with a clearly documented matching rule;
- calibration is secondary, not a model-selection target unless predeclared.

## External generalization
FracAtlas:
- zero-shot only after model lock;
- no fine-tuning;
- map only compatible fracture class;
- clearly state mixed anatomical regions and domain shift;
- report mAP50/mAP50-95 where labels permit plus fracture FROC;
- do not compare its number directly with Split-B.

## Literature-comparability track
Optional after method lock:
- use only an exact, publicly available split manifest;
- no redesign/tuning from its result;
- report as contextual comparability, separate from the primary Split-B evidence.
