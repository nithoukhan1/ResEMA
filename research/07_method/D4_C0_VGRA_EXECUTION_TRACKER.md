# D4-C0 VGRA Execution Tracker

Design authority: `research/lpq-method-01 @ a5d260ca83236c774e1920ce729e08eaacfcf5d4`

Implementation branch: `research/vgra-impl-01`

| Phase | Status | Training? | Exit |
|---|---|---:|---|
| D4-C0 plan/worktree | COMPLETE | NO | branch/worktree + plan frozen |
| D4-C1 core math | COMPLETE | NO | standalone tests PASS |
| D4-C2 pair manifest | COMPLETE | NO | deterministic membership + visibility schema PASS |
| D4-C3 head integration | COMPLETE | NO | zero-gate + box/DFL firewall PASS |
| D4-C4 data/batching | COMPLETE | NO | pair-safe augmentation + pair-preserving batch audit PASS |
| D4-C5A visibility supervision | COMPLETE | NO | synthetic class-target/weighted-CE tests PASS |
| D4-C5B runtime + native loss | COMPLETE | NO | mixed paired/single + 4-component native loss CPU tests PASS |
| D4-C5C trainer/validator | COMPLETE | NO | synthetic paired trainer/validator stack CPU PASS |
| D4-C6 implementation closure | COMPLETE | NO | audited source/evidence inventory + CPU regressions PASS |
| D4-D1 unit/static | COMPLETE | NO | D1R repair + D1B independent 75-test CPU verification PASS |
| D4-D2 transfer/identity | COMPLETE | NO | verified pretrained Early transfer, zero-rho identity, synthetic EMA/load/gradients |
| D4-D3 integrated synthetic | COMPLETE_CONDITIONAL | NO | accepted CPU synthetic/pretrained replay; numerical boundary remains conditional |
| D4-E R3 scientific verification | HOLD | NO | R3-03C 17/17; R3-04 EMA timing gap; additional checks pending |
| D4-E source freeze | LOCKED | NO | source/recipe not frozen |
| D4-F MV-00/MV-01 | LOCKED | separate auth | both preserved |
| D4-G/H/I/J | LOCKED | governed | diagnostics/ablations/final freeze |

Current next action: `D4_E2_R3_04_PREUPDATE_EMA_SAFETY_REVIEW`

Permanent firewalls: B-test NONE; GPU training unauthorized; attention unauthorized; hyperparameter search unauthorized; additional modules unauthorized.

## D4-D1R confirmed blockers and guarded remediation

Two D4-D1 R1 blockers were confirmed and repaired in dedicated VGRA code.
D4-D1 exit remains OPEN until D4-D1B independent re-verification.
No D4-D2, GPU training or B-TEST authorization.

## R3-04 local vault and hold (2026-10-09)

Evidence ZIP SHA256 `f621cd797aea1684887e8667eac1fa3420ee45e596ea5e97f11c892611a15ee0`; 25 files, no missing logs. See `research/07_method/D4_E2_R3_04_EVIDENCE_AND_SCIENTIFIC_HOLD.md`. No GPU training, source freeze, or B-TEST access.
