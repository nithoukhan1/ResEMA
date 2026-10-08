# D4 LPQ Master Tracker

## Overall state

| Item | State |
|---|---|
| D3 execution | COMPLETE / CLOSED |
| D3-E1 evidence extraction | COMPLETE |
| D3-E2 scientific decision | PATH B SELECTED |
| D4 branch | CREATED BY D4-A |
| Active D4 method | VGRA CANDIDATE V1 SPEC FROZEN |
| Original DAR formulation | REJECTED AS NOVELTY CORE |
| Original GDS formulation | REJECTED AS NOVELTY CORE |
| VGRA implemented | FALSE |
| VGRA source frozen | FALSE |
| VGRA training authorized | FALSE |
| attention authorized | FALSE |
| final architecture frozen | FALSE |
| B-VAL role | DEVELOPMENT / SELECTION |
| B-TEST access | NONE |

## Phase tracker

| Phase | Status | Exit condition |
|---|---|---|
| D4-A Transition/context freeze | COMPLETE | remote-verified commit 3cb2745119d7920192654d86d259876d763aecd5 |
| D4-B1 Prior-art collision gate | COMPLETE | original DAR/GDS rejected as novelty cores |
| D4-B2A Multi-view feasibility audit | COMPLETE / STRONG | 85%+ paired study groups and ~94% patients with pair |
| D4-B2B Object/error complementarity gate | COMPLETE / MODERATE | annotation nonredundancy + asymmetric TP evidence, strongest for fracture |
| D4-B2C1 Novelty landscape/candidate shortlist | COMPLETE | crowded fusion/distillation paths rejected; VGRA working candidate promoted |
| D4-B2C2 VGRA architecture/math/promotion freeze | COMPLETE | candidate V1 spec frozen; implementation authorized, training locked |
| D4-C0 VGRA implementation plan/worktree | COMPLETE | dedicated implementation branch/worktree + execution plan frozen |
| D4-C1 VGRA core math implementation | COMPLETE | standalone mathematical primitives + CPU unit tests PASS |
| D4-C2 VGRA pair/visibility manifest | NEXT | deterministic TRAIN/VAL pairing contract; no test access |
| D4-C3/C4/C5/C6 VGRA integration | LOCKED | sequential governed implementation transactions |
| D4-D Structural verification | LOCKED | D4-C complete |
| D4-E Source freeze/experiment contract | LOCKED | D4-D PASS |
| D4-F Primary LPQ training | LOCKED | separate training authorization |
| D4-G Residual diagnostics | LOCKED | D4-F preserved |
| D4-H Attribution ablations | LOCKED | D4-G decision |
| D4-I Optional attention gate | LOCKED | evidence-based authorization only |
| D4-J Final architecture freeze | LOCKED | development evidence complete |
| D5 Robustness/multiseed/efficiency | LOCKED | D4-J complete |
| D6 Comparator freeze | LOCKED | final method stable |
| D7 Final-test contract | LOCKED | all models/checkpoints frozen |
| D8 Sealed B-test | LOCKED | D7 authorization |
| D9 Manuscript evidence | LOCKED | D8 preserved |

## Current next action

`D4-C1_VGRA_CORE_MATH_IMPLEMENTATION`

## Do not do yet

- do not edit model/loss code;
- do not run GPU training;
- do not add attention;
- do not access Split-B test;
- do not call LPQ the final architecture;
- do not claim novelty before D4-B.
