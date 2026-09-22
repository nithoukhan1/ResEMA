# 06 - Experiment Governance

## Experiment IDs

Use clear, non-reused scientific IDs.

Active baseline reverification:
- `BASE-<SPLIT>-<DATA>-<INIT>-S<SEED>`
- SPLIT: `A` or `B`
- DATA: `ORG` or `AUG`
- INIT: `PT` or `SCR`

Examples:
- `BASE-B-ORG-PT-S42`
- `BASE-B-AUG-SCR-S42`

Existing mechanism/diagnostic families remain valid:
- `Dxx` diagnostics
- `Mxx` mechanism development
- `Rxx` confirmation
- `CVxx` robustness
- `Txx` final test
- `Exx` external validation

Never reuse an experiment ID.

## Before-run registration

Required:
- scientific question or hypothesis;
- parent/control model;
- intended scientific change or controlled condition;
- dataset/split;
- seed;
- initialization;
- primary metric;
- secondary metrics;
- decision rule;
- code commit SHA.

Baseline factorial conditions may differ in split, training representation, or initialization by design. These dimensions must be explicit in the experiment ID and registry.

## Primary endpoint

Development primary:
`mAP50-95` on the applicable frozen validation partition.

Split-B original is the primary development environment unless a registered experiment explicitly targets another training representation or split.

## Secondary metrics

Report where applicable:
- precision;
- recall;
- F1;
- mAP50;
- mAP50-95;
- Clinical-7;
- Core-6;
- per-class AP;
- fracture AP;
- target-specific metric;
- convergence;
- parameters.

## Mechanism screening rule

For later architecture/loss mechanism screening, predeclare the exact pass/fail rule before training.

Historical guidance:
- overall mAP50-95 gain >= 0.5 percentage points; OR
- a prespecified targeted metric gain >= 1.0 pp with overall mAP50-95 decrease <= 0.2 pp.

Guardrails:
- no major fracture AP collapse;
- no conclusion driven by one rare patient;
- no hidden recipe change;
- complexity must be justified.

The baseline-reverification phase does not use this mechanism-pass rule. Its purpose is to measure controlled baseline conditions and convergence.

## Repeats

Initial baseline matrix:
- seed 42 for all six registered baseline conditions.

Additional seeds are added only after the initial matrix and convergence analysis identify the primary confirmation conditions.

## Test policy

Split-A and Split-B test partitions are not used for development choices.
Final test evaluation occurs only after method lock.

## Negative results

Negative and null experiments remain registered.
