# ARCH-CORR-01 Master Tracker

## Global status

`VERIFICATION / RECORD FREEZE IN PROGRESS — TRAINING NOT AUTHORIZED`

Parallel baseline work:
- canonical branch: `research/baseline-refresh`;
- E200 scratch calibration: running externally on Kaggle.

Architecture branch:
- `research/arch-corr-01b`.

## Tracker

| Work item | Implementation | CI/static | Runtime transfer audit | Local reproduction | Final record |
|---|---|---|---|---|---|
| Historical C3k2_SC forensic audit | complete | N/A | N/A | code review required | documented |
| TPSC Early | complete | pass previously | pass previously | pending | pending |
| TPSC G4 | complete | pass previously | pass previously | pending | pending |
| DySample | existing | focused test being added | pending closure | pending | pending |
| Historical ResEMA-V2 | historical | N/A | N/A | code review required | retired/documented |
| CanonicalEMA | complete | static tests present | current-head rerun pending | pending | pending |
| C3k2_TPEMA | complete | static tests present | current-head rerun pending | pending | pending |
| Architecture training | not started | — | — | — | BLOCKED |
| Split-B test | sealed | — | — | — | SEALED |

## Known CI history

- `907d371efe2e9f1ca350d7a9c2e7da0fb6a5f285`: TPSC focused/runtime audit passed.
- `9b2a8ff6cb67af0e7882c8a44122007af010cef2`: Research Integrity passed; dedicated runtime workflow stopped at a stale exact-string TPSC parser assertion after 11 tests passed.
- `9416a56795767ea13d215dfb705a87141bc76878`: stale parser assertion repaired; current-head CI must pass before local verification lock.

## Closure criteria

ARCH-CORR-01 is not closed until the exact final verification head has:
- GitHub Research Integrity PASS;
- dedicated architecture runtime audit PASS;
- local focused tests PASS;
- local official-checkpoint TPSC audit PASS;
- local official-checkpoint EMA audit PASS;
- local DySample focused audit PASS;
- clean worktree;
- verification report hashes recorded;
- final documentation/decision tracker committed.
