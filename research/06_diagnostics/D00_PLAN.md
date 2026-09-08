# 08 — D00 Diagnostic Plan

## Purpose
D00 is the evidence gate before new architecture coding.

## Existing checkpoint inputs
- A0 seed42
- C0 seed43
- A1 seed42
- C1 seed43
- A3 seed42
- C2 seed43

## D00-A — Transfer coverage
For each architecture:
- build target model
- compare with official yolo11s.pt
- count matching tensors
- count matching parameters
- report percentage coverage
- list unmatched layers

Output:
`transfer_coverage.csv`

## D00-B — Axis feasibility
- locate original axis annotations
- verify two-point-line semantics
- missing/invalid count
- parser tests
- orientation distribution by split/projection

Outputs:
`axis_inventory.csv`
`axis_distribution.png`

## D00-C — Projection audit
- exact projection codes from authoritative metadata
- counts by train/val/test
- patient counts by projection
- class x projection support
- model performance by projection

Outputs:
`projection_dictionary.yaml`
`projection_counts.csv`
`projection_performance.csv`

## D00-D — Paired-study feasibility
- group by patient/exam/study/laterality
- identify AP/PA + LAT pairs
- pair completeness
- class support
- split distribution
- missing-view rate

Output:
`paired_study_inventory.csv`

## D00-E — Localization/geometry
- AP50/AP75/high-IoU profile where valid
- matched IoU distribution
- normalized box area
- aspect ratio
- class x geometry
- size-stratified analysis

## D00-F — Rare-patient sensitivity
For rare classes:
- unique positive patients
- detections per patient
- AP contribution by patient
- leave-one-positive-patient-out sensitivity where feasible

## Final output
`DIAGNOSTIC_CONCLUSIONS.md`

It must answer:
1. Is transfer disruption real and how large?
2. Is fine detail/localization a top bottleneck?
3. Is projection conditioning justified?
4. Is axis annotation usable?
5. Is paired-view fusion feasible?
6. How severe is patient-level long-tail concentration?
7. Which mechanism should be implemented first?
