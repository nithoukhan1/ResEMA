# GRAZPEDWRI-DX Publication Research Workspace

This directory is the authoritative research record for the publication-oriented detection project.

## Current state

D3 residual diagnostics are complete, preserved and closed.

Current active method-development branch:

`research/lpq-method-01`

D3-E2 scientific decision:

`PATH_B_SELECTED`

Working method concept:

`LPQ = Lesion-Preserving Quality`

Current method status:

`CONCEPT_REDESIGN_REQUIRED_AFTER_D4_B1_PRIOR_ART_GATE`

No new GPU training is authorized.
Split-B test remains sealed.

## Start here

1. `07_method/D4_LPQ_CONTEXT_INDEX.md`
2. `07_method/D4_LPQ_MASTER_TRACKER.md`
3. `07_method/D4_LPQ_MASTER_PLAN.md`
4. `07_method/D4_LPQ_SCIENTIFIC_RATIONALE.md`
5. `CURRENT.md`
6. `DECISIONS.md`
7. `PROJECT_LOG.md`
8. `05_experiments/EXPERIMENTS.csv`
9. `01_provenance/ARTIFACTS.csv`

## Active D4 workflow

D4-A transition/context freeze
-> D4-B1 prior-art collision gate
-> D4-B2 revised hypothesis + mathematical/novelty freeze
-> D4-C implementation
-> D4-D structural/synthetic verification
-> D4-E source freeze + experiment contract
-> D4-F first LPQ training
-> D4-G diagnostics
-> D4-H ablations
-> optional D4-I attention gate
-> D4-J final architecture freeze
-> robustness/comparators
-> sealed final test.

## Dataset roles

- Split-B TRAIN: candidate training
- Split-B VAL: architecture development and selection
- Split-B TEST: sealed final evaluation only

No inner development split is planned.

## Workflow

Local VS Code -> Git -> GitHub immutable commit -> governed Kaggle exact commit ->
Save Version / resume if needed -> external checkpoint storage -> compact result
registration in Git -> documented scientific decision.

## Historical work

The baseline refresh, corrected SCConv/DySample/EMA architecture audit, module-family
screen, combination screen, D2 standardized validation and D3 residual diagnostics
remain preserved as historical scientific evidence.

`07_method/METHOD_BLUEPRINT_V7.md` is historical/paused and is not the active architecture specification.

## D4 current checkpoint — multi-view feasibility

D4-B2A found STRONG AP/LAT pair coverage in Split-B TRAIN and VAL.

Multi-view is now a serious candidate but remains unimplemented.

Next canonical gate:
`D4-B2B_MULTIVIEW_OBJECT_LEVEL_COMPLEMENTARITY_AND_ERROR_FEASIBILITY`

Split-B TEST remains sealed.

## D4 current checkpoint — multi-view complementarity

D4-B2A: STRONG pair availability.
D4-B2B: MODERATE annotation/model-error complementarity.

Multi-view remains a serious candidate but no architecture is selected.

Next canonical gate:
`D4-B2C_MULTIVIEW_METHOD_NOVELTY_AND_MATHEMATICAL_DESIGN`

Split-B TEST remains sealed.

## D4 current checkpoint — VGRA working candidate

After D4-B2A/B2B established strong pair availability and moderate complementarity,
D4-B2C1 narrowed the prior-art landscape.

Working design candidate:

`VGRA = Visibility-State-Gated Cross-View Residual Assistance`

Status:
`WORKING CANDIDATE ONLY`

Next:
`D4-B2C2_VGRA_MATHEMATICAL_ARCHITECTURE_AND_PROMOTION_FREEZE`

No implementation/training/test access is authorized.

## Active D4 method candidate

`VGRA = Visibility-State-Gated Cross-View Residual Assistance`

Status:
`CANDIDATE V1 SPEC FROZEN FOR IMPLEMENTATION`

Implementation is authorized.
Training is not.

Start with:
`07_method/D4_B2C2_VGRA_MATHEMATICAL_SPEC.md`

## VGRA implementation workflow
Frozen design: `research/lpq-method-01 @ a5d260ca83236c774e1920ce729e08eaacfcf5d4`. Active implementation: `research/vgra-impl-01`. Start with D4-C0 governance/master plan/tracker. Training remains locked.

## VGRA implementation status — D4-C1

Standalone VGRA mathematical primitives:
`IMPLEMENTED / CPU-UNIT-VERIFIED`

Not yet integrated with Detect, pair-aware data loading or trainer/validator.

Next:
`D4-C2_VGRA_PAIR_AND_VISIBILITY_MANIFEST`

## VGRA implementation status — D4-C2

Pairing universe:
`FROZEN / DETERMINISTIC`

Visibility target schema:
`FROZEN`

Actual training visibility states:
`NOT YET GENERATED`

Next:
`D4-C3_VGRA_HEAD_MODEL_INTEGRATION`

## VGRA implementation status — D4-C3

VGRA head/model integration:
`IMPLEMENTED / CPU-VERIFIED`

Candidate parameters:
`9672660`

Next:
`D4-C4_VGRA_PAIR_AWARE_DATA_AND_BATCHING`

## VGRA implementation status — D4-C4

Pair-aware dataset/batching:
`IMPLEMENTED / CPU-VERIFIED / MANIFEST-AUDITED`

Cross-study composition:
`DISABLED`

Next:
`D4-C5_VGRA_CRITERION_TRAINER_VALIDATOR_INTEGRATION`

## VGRA implementation status — D4-C5A

Target/visibility loss utility: `IMPLEMENTED / SYNTHETIC CPU TESTED`.

Next: `D4-C5B_VGRA_PAIRED_RUNTIME_AND_LOSS_INTEGRATION`.

C5C trainer/validator, D4-D verification, source freeze and training remain locked.

## VGRA status — D4-C5B

VGRA mixed paired/single runtime and native detection loss:
`IMPLEMENTED / SYNTHETIC CPU VERIFIED`

Trainer and validator integration:
`NEXT / D4-C5C`

Training is still locked.

## VGRA status — D4-C5C

Dedicated pair-aware trainer and validator:
`IMPLEMENTED / SYNTHETIC CPU VERIFIED`.

Next: `D4-C6_VGRA_IMPLEMENTATION_CLOSURE`.
No GPU training permitted.

## VGRA implementation status — D4-C6

`IMPLEMENTED / SOURCE INVENTORIED / CPU REGRESSION PASS`

Independent D4-D verification not yet complete.
GPU training and B-TEST remain unauthorized.
Next: `D4-D1_VGRA_INDEPENDENT_STATIC_UNIT_VERIFICATION`.

## VGRA D4-D1R remediation state

Read-only audit revealed final-eval and VAL-exclusion blockers. Corrective patch complete;
D4-D1 independent re-verification NEXT. Training and test remain sealed.

## D4-D1 formal independent acceptance — 2026-10-08

Independent D4-D1 static/structural verification: `COMPLETE / CPU VERIFIED`.
D4-D2 checkpoint transfer/identity: NEXT; D4-D3 integrated numerical
verification: LOCKED. Signed-beta/all-zero-gradient gates are OPEN.
GPU training and B-TEST remain forbidden.

Next: `D4-D2_VGRA_PRETRAINED_TRANSFER_AND_ZERO_GATE_IDENTITY`

## D4-D2 formal pretrained-transfer acceptance — 2026-10-08

D4-D2A/B/C rerun as immutable SHA-pinned independent synthetic CPU checks.
Canonical pretrained SCConv-Early checkpoint matched its frozen archive;
537/537 Early tensors transferred unchanged; zero-rho class/box/DFL identity
passed in TRAIN and EVAL, and synthetic TRAIN buffers/EMA/best-checkpoint
restoration plus gradient reachability passed.

- Evidence: `research/07_method/D4_D2_INDEPENDENT_REPLAY_EVIDENCE.txt` (SHA256 `ee23f2f7ae0bcb9775bf9b7e36aef8b77b911a926ea324233cca7c8988b2ee83`).
- Formal decision: `research/07_method/D4_D2_FORMAL_ACCEPTANCE.md`; gate: `research/07_method/D4_D2_GATE_RECORD.json`.
- D4-D3 all-zero gradient numerical issue, persistent dataloader/complete
  synthetic epoch, signed-beta interpretation and MV00/MV01 recipe parity: OPEN.
- No GPU training, source freeze, model architecture change or B-TEST access.

Next: `D4-D3_VGRA_INTEGRATED_SYNTHETIC_NUMERICAL_VERIFICATION`
