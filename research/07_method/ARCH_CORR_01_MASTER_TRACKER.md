# ARCH-CORR-01 Master Tracker

## Global status

`COMPLETE — TECHNICAL VERIFICATION AND RECORD FREEZE CLOSED`

Parallel baseline work:
- canonical branch: `research/baseline-refresh`;
- E200 scratch calibration: running externally on Kaggle.

Architecture branch:
- `research/arch-corr-01b`.

Verified implementation head:
- `a30df2165cb24a0796086a32b07445ea04b5c7bc`.

Closure candidate:
- `1801379e7cfdf4260e54d08b4e12391b795dc32f`.

## Tracker

| Work item | Implementation | Remote runtime | Local reproduction | Final disposition |
|---|---|---|---|---|
| Historical C3k2_SC forensic audit | complete | N/A | code reviewed | historical only |
| TPSC Early | complete | PASS | PASS | technically verified candidate |
| TPSC G4 | complete | PASS | PASS | technically verified candidate |
| DySample | retained | PASS | PASS | technically verified / retain |
| Historical ResEMA-V2 | historical | N/A | code reviewed | retired from new design |
| CanonicalEMA | complete | PASS | PASS | technically verified candidate |
| C3k2_TPEMA | complete | PASS | PASS | technically verified candidate |
| Architecture training | not started | — | — | requires separate authorization |
| Split-B test | sealed | — | — | SEALED |

## Evidence

Verified implementation head:
- Research Integrity run #33 / `36430937501`: PASS.
- ARCH-CORR runtime audit run #5 / `36430937722`: PASS.
- local focused tests: 16/16 PASS.
- local locked-checkpoint and identity audits: PASS.

Local hashes:
- TPSC: `f8aed12d60a3e60937bf5c5917bf61ce657f50eb45a6a239273c645e602f918c`;
- EMA: `d2002175791c9035af0894de3c5e32823a786db8b7e763628c2b4d274858258d`;
- master: `ac2d2076d7f936999349809b83141f63dd45a88ad4a8b3ae6ff45ef35bf54420`.

Closure candidate:
- Research Integrity run #34 / `36436122481`: PASS.
- ARCH-CORR runtime audit run #6 / `36436122550`: PASS.
- runtime audit artifact id: `10975698003`.
- runtime audit artifact digest:
  `sha256:37495f012103bc6898bb7d1ba23d2f9a3623c203aeba45ff96e879affc51ea93`.

## Next phase

Register controlled **single-module pretrained Split-B-original** experiments.

No combination experiment is authorized until single-module validation evidence is
available. The E200 scratch calibration remains a separate baseline-budget diagnostic.

Split-B test remains sealed.
