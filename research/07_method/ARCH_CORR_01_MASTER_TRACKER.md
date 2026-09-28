# ARCH-CORR-01 Master Tracker

## Global status

`LOCAL VERIFICATION COMPLETE — CLOSURE CANDIDATE CI PENDING — TRAINING NOT AUTHORIZED`

Parallel baseline work:
- canonical branch: `research/baseline-refresh`;
- E200 scratch calibration: running externally on Kaggle.

Architecture branch:
- `research/arch-corr-01b`.

Verified implementation head:
- `a30df2165cb24a0796086a32b07445ea04b5c7bc`.

## Tracker

| Work item | Implementation | CI/static | Runtime transfer audit | Local reproduction | Final record |
|---|---|---|---|---|---|
| Historical C3k2_SC forensic audit | complete | N/A | N/A | code reviewed | documented |
| TPSC Early | complete | PASS | PASS | PASS | closure candidate |
| TPSC G4 | complete | PASS | PASS | PASS | closure candidate |
| DySample | retained | PASS | PASS focused tests | PASS | closure candidate |
| Historical ResEMA-V2 | historical | N/A | N/A | code reviewed | retired/documented |
| CanonicalEMA | complete | PASS | PASS | PASS | closure candidate |
| C3k2_TPEMA | complete | PASS | PASS | PASS | closure candidate |
| Architecture training | not started | — | — | — | BLOCKED |
| Split-B test | sealed | — | — | — | SEALED |

## Verified implementation evidence

GitHub:
- Research Integrity run #33 / `36430937501`: PASS.
- ARCH-CORR runtime audit run #5 / `36430937722`: PASS.
- Runtime audit artifact id: `10973751936`.
- Runtime audit artifact digest:
  `sha256:50c23ceb9cf63f797bb926dfa10cbef74e705c959e9135699c878c82b4b3ca30`.

Local:
- focused tests: 16/16 PASS;
- locked yolo11s.pt SHA256:
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`;
- TPSC audit SHA256:
  `f8aed12d60a3e60937bf5c5917bf61ce657f50eb45a6a239273c645e602f918c`;
- EMA audit SHA256:
  `d2002175791c9035af0894de3c5e32823a786db8b7e763628c2b4d274858258d`;
- local verification master SHA256:
  `ac2d2076d7f936999349809b83141f63dd45a88ad4a8b3ae6ff45ef35bf54420`;
- local worktree clean after verification;
- training_started=false;
- dataset_access=NONE;
- test_access=NONE.

## Known CI history

- `907d371efe2e9f1ca350d7a9c2e7da0fb6a5f285`: TPSC focused/runtime audit passed.
- `9b2a8ff6cb67af0e7882c8a44122007af010cef2`: Research Integrity passed; dedicated runtime workflow stopped at a stale exact-string TPSC parser assertion after 11 tests passed.
- `9416a56795767ea13d215dfb705a87141bc76878`: stale parser assertion repaired.
- `a30df2165cb24a0796086a32b07445ea04b5c7bc`: Research Integrity PASS, dedicated runtime audit PASS, local exact-head reproduction PASS.

## Closure sequence

1. commit documentation/evidence closure candidate;
2. require Research Integrity PASS on the candidate;
3. require dedicated architecture runtime audit PASS on the candidate;
4. record those candidate run IDs in a final closure attestation;
5. verify Research Integrity on the final attestation;
6. declare ARCH-CORR-01 complete;
7. only then authorize corrected single-module training.

Combination experiments remain forbidden until single-module results are available.
