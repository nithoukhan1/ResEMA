# D4 LPQ Context Index

## Purpose

This is the canonical starting point for the D4 method-development phase.
A future chat or collaborator should read this file first instead of reconstructing
the project from conversation history.

## Authoritative transition

- source branch: `research/combination-screen-01`
- source HEAD: `227918fa1c2a496023e39a992bfaedd953bcc8f9`
- D4 branch: `research/lpq-method-01`
- D3 execution/preservation: CLOSED
- D3 rerun authorized: FALSE
- D3-E2 decision: PATH B
- final paper architecture frozen: FALSE
- new GPU training authorized: FALSE
- Split-B test access: NONE

## Why D4 exists

D3 showed that SCConv-Early is useful but insufficient as the final paper method.

At the central fixed confidence operating point (0.25), Early versus baseline:
- net TP change: -1;
- FP change: -44;
- FN change: +1;
- precision improved modestly;
- recall was essentially unchanged;
- duplicate FP increased;
- object-level recovered/lost detections were approximately balanced.

The central residual problem is therefore not simply "insufficient feature extraction".
The working residual is:

`retain useful FP suppression while improving true-lesion score reliability,
true-positive preservation and duplicate-candidate separation`.

## Working method hypothesis

Temporary project name:

`LPQ = Lesion-Preserving Quality`

Working components:
- `DAR = Distribution-Aware Reliability`;
- `GDS = Groupwise Duplicate Separation`.

These are CONCEPTS ONLY at D4-A.
No equations, source implementation, loss coefficients or final architecture are frozen yet.

## Development protocol

- B-TRAIN: architecture training;
- B-VAL: architecture development/selection;
- B-TEST: sealed final evaluation only;
- no B-inner split is planned;
- all architecture changes must stop before B-test access.

## Canonical evidence

A12-D3 preservation archive:
`8c3545fd772b56984d87090819ba265bc98dbfdf15c456b4abb6f0a02fd513af`

A12-D3-E1 package:
`d75afd61c0549c6ae805a57b364e926bc7750bfb46fec520e69002976b147cb9`

A12-D3-E1 manifest:
`34d79b264f7a26e051ab78947d59a3680afdf03356cd76ac47cf4d36460dd36e`

A12-D3-E1 scientific evidence report:
`02abfeb20c59aa16e3a2251d2dd48d55ab3b2a914146adad7ea98d3003bc84db`

## Read in this order

1. `research/07_method/D4_LPQ_CONTEXT_INDEX.md`
2. `research/07_method/D4_LPQ_MASTER_TRACKER.md`
3. `research/07_method/D4_LPQ_MASTER_PLAN.md`
4. `research/07_method/D4_LPQ_SCIENTIFIC_RATIONALE.md`
5. `research/07_method/D4_LPQ_TRANSITION_RECORD.json`
6. `research/CURRENT.md`
7. `research/DECISIONS.md`
8. `research/PROJECT_LOG.md`
9. `research/01_provenance/ARTIFACTS.csv`

## Immediate next transaction

`D4-B_LPQ_MATHEMATICAL_SPECIFICATION_NOVELTY_AND_PROMOTION_FREEZE`

D4-B must occur before LPQ implementation or training.

## D4-B1 novelty correction

The first LPQ solution sketch did not survive prior-art review.

- Original DAR: rejected as novelty core because GFLV2 already derives localization
  quality from learned box distributions.
- Generic class/localization-quality fusion: already covered by GFL, VFNet, TOOD/TAL
  and the repository's TaskAlignedAssigner.
- Original GDS score-gap concept: rejected as novelty core because hybrid/dual
  assignment and duplicate ranking/score-gap mechanisms already exist.

The D3 residual problem remains valid.

Read next:
- `D4_B1_PRIOR_ART_COLLISION_LEDGER.md`
- `D4_LPQ_REDESIGN_CONSTRAINTS.md`

Next transaction:
`D4-B2_REVISED_METHOD_HYPOTHESIS_PRIOR_ART_AND_MATHEMATICAL_FREEZE`

## D4-B2A multi-view feasibility result

A read-only Split-B TRAIN/VAL metadata audit found STRONG multi-view coverage.

TRAIN:
- AP/LAT-capable side-specific study groups: 6502 (85.6653%)
- patients with >=1 pair: 3998 (93.7617%)

VAL:
- AP/LAT-capable side-specific study groups: 1402 (86.4365%)
- patients with >=1 pair: 857 (93.7637%)

This establishes data-structure feasibility only.

No architecture is selected and no implementation/training is authorized.

Next:
`D4-B2B_MULTIVIEW_OBJECT_LEVEL_COMPLEMENTARITY_AND_ERROR_FEASIBILITY`

## D4-B2B complementarity result

The preserved paired-view audit returned:

`COMPLEMENTARITY_EVIDENCE_TIER=MODERATE`

Key values:
- annotation view-exclusive fraction: 11.4813%
- mean AP/LAT class-set Jaccard: 0.914408
- baseline one-view-only TP fraction @0.25: 3.7280%
- Early one-view-only TP fraction @0.25: 4.3425%
- fracture baseline one-view-only TP fraction @0.25: 5.9293%
- fracture Early one-view-only TP fraction @0.25: 7.4116%

Interpretation:
paired views are mostly similar but not redundant. Complementary detection behavior is
modest overall and stronger for fracture. This supports studying selective/conditional
cross-view assistance rather than assuming unconditional heavy fusion.

Next:
`D4-B2C_MULTIVIEW_METHOD_NOVELTY_AND_MATHEMATICAL_DESIGN`

## D4-B2C1 novelty landscape

The multi-view design space was narrowed before implementation.

Rejected as novelty cores:
- generic always-on dual-view feature fusion;
- generic uncertainty/conflict-gated fusion;
- generic cross-view mutual distillation.

Working candidate promoted to mathematical design:

`VGRA = Visibility-State-Gated Cross-View Residual Assistance`

VGRA remains a hypothesis only.
It explicitly models per-class projection visibility states (absent, AP-only, LAT-only,
both) and uses them to control residual cross-view semantic assistance while keeping
box regression view-specific.

Next:
`D4-B2C2_VGRA_MATHEMATICAL_ARCHITECTURE_AND_PROMOTION_FREEZE`

## D4-B2C2 VGRA V1 candidate freeze

VGRA candidate V1 is mathematically frozen for implementation.

Core:
- shared Early detector for AP/LAT;
- four-state per-class visibility supervision;
- target-specific gate:
  - AP: q11 - q01;
  - LAT: q11 - q10;
- low-rank target-local/companion-semantic compatibility;
- bounded zero-initialized classification-logit residual;
- box/DFL branches remain strictly view-specific;
- exact single-view fallback.

Candidate implementation is now authorized.
GPU training is still forbidden.

Next:
`D4-C_VGRA_IMPLEMENTATION_SCAFFOLD_AND_UNIT_TESTS`
