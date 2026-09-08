# 06 — Experiment Governance

## IDs
- Dxx diagnostics
- Mxx mechanism development
- Rxx confirmation
- CVxx robustness
- Txx final test
- Exx external validation

Never reuse an ID.

## Before-run registration
Required:
- hypothesis
- parent model
- exactly one intended scientific change
- dataset/split
- seed
- primary metric
- secondary metrics
- pass/fail rule
- code commit SHA

## Primary endpoint
Development primary: `mAP50-95` on frozen operational Split-B validation.

## Secondary metrics
- mAP50
- Clinical-7
- Core-6
- per-class AP
- fracture AP
- target-specific metric
- convergence
- params

## Screening rule
A mechanism normally passes if:
A. overall mAP50-95 improves >=0.5 percentage points;
OR
B. a prespecified targeted metric improves >=1.0 pp and overall mAP50-95 decreases no more than 0.2 pp.

Guards:
- no major fracture AP collapse;
- no conclusion driven by one bonelesion patient;
- no hidden recipe change;
- complexity justified.

## Strong final-method target
Prefer:
- >=1.0 pp reproducible gain
- stronger SOTA-oriented target: ~1.5–2.0 pp
- positive Core-6
- robust rare-class behavior

## Repeats
One seed for screening.
Second seed only for shortlisted candidates.

## Test policy
Split-B test is never used for development choices.
Final test evaluates exactly frozen baseline and frozen proposed method.

## Negative results
Negative experiments remain in the registry.
