# D4 LPQ Master Tracker

## Overall state

| Item | State |
|---|---|
| D3 execution | COMPLETE / CLOSED |
| D3-E1 evidence extraction | COMPLETE |
| D3-E2 scientific decision | PATH B SELECTED |
| D4 branch | CREATED BY D4-A |
| LPQ status | CONCEPT REDESIGN REQUIRED |
| Original DAR formulation | REJECTED AS NOVELTY CORE |
| Original GDS formulation | REJECTED AS NOVELTY CORE |
| LPQ implemented | FALSE |
| LPQ source frozen | FALSE |
| LPQ training authorized | FALSE |
| attention authorized | FALSE |
| final architecture frozen | FALSE |
| B-VAL role | DEVELOPMENT / SELECTION |
| B-TEST access | NONE |

## Phase tracker

| Phase | Status | Exit condition |
|---|---|---|
| D4-A Transition/context freeze | COMPLETE | remote-verified commit 3cb2745119d7920192654d86d259876d763aecd5 |
| D4-B1 Prior-art collision gate | IN PROGRESS | record collisions and redesign constraints |
| D4-B2 Revised hypothesis/math freeze | LOCKED | D4-B1 complete |
| D4-C Implementation | LOCKED | D4-B2 complete |
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

`D4-B2_REVISED_METHOD_HYPOTHESIS_PRIOR_ART_AND_MATHEMATICAL_FREEZE`

## Do not do yet

- do not edit model/loss code;
- do not run GPU training;
- do not add attention;
- do not access Split-B test;
- do not call LPQ the final architecture;
- do not claim novelty before D4-B.
