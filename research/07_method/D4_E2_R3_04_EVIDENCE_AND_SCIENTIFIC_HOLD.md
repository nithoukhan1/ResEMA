# D4-E2 R3-04 — State/EMA synthetic diagnostic and scientific HOLD

**Observed on:** 2026-10-09, as provided by the user in a Windows PowerShell log.
**Source branch:** `research/vgra-impl-01`.
**Pinned HEAD:** `48cc059cfa133ae5fbfcec607f15daeda843d3ba`.
**OS / runtime:** Windows, Python 3.11.9 previously established; R3-04 logs show Torch 2.5.1+cu121, CPU.
**Test script:** `D4_E2_R3_04_SYNTHETIC_STATE_AND_EMA_PROBE_2026-10-09.py`.
**User-checked script SHA256:** `7ac076ee53eeb8479446e80409e62749f5527fe546f15103c490a62b36533744`.
**Command result:** `R3_04_ExitCode=0`, `Ran 13 tests in 0.039s`, `OK`.
**Modeling caveat:** tiny synthetic model with real SGD optimizer and synthetic scaler/EMA stubs, not native Ultralytics trainer/AMP integration.

## Explicit test observations

| Test | Observed result |
|---|---|
| 01 finite setup registers optimizer post hook | PASS |
| 02 pretraining nonfinite model parameter rejection | PASS |
| 03 pretraining nonfinite BN-like buffer rejection | PASS |
| 04 pretraining nonfinite optimizer state rejection | PASS |
| 05 pretraining nonfinite scaler state rejection | PASS |
| 06 pretraining nonfinite EMA rejection | PASS |
| 07 pre-update corrupt model blocks optimizer and EMA | PASS |
| 08 pre-update optimizer poison blocks step | PASS |
| 09 pre-update scaler poison blocks step | PASS |
| 10 finite optimizer and EMA update | PASS |
| 11 newly corrupt poststep model rejected before EMA | PASS |
| 12 newly corrupt EMA rejected after update | PASS |
| 13 existing EMA corruption not checked before optimizer step | **COUNTEREXAMPLE OBSERVED**; test harness marks observation as `ok` |

Verbatim key observation from user output:

> `OBSERVED_LIMITATION: preexisting poisoned EMA survived pre_update, optimizer.step happened before post_update_ema abort`

The entire unit suite passed because test 13 intentionally asserted the current unsafe timing. `13/13` is **not** scientific acceptance of the safety gate. `R3_04_13_PROBES_PASS; SCIENTIFIC_GATE=HOLD`.

## Causal inference limits

The demonstrated fact is a pre-update EMA validation gap in the **tested guard path** using a stub EMA. We cannot claim this exact failure necessarily manifests in every native Ultralytics training condition without testing real integration. The risk matters because the project's intended first-failure semantics require corrupted state to abort before a parameter update rather than after it.

## Decision and next scientific action

- **D4-E2 R3-04: HOLD.**
- Review-only design next: add a fail-closed pre-update EMA state check, preserving existing pretraining/post-update checks, optimizer and EMA semantics, and all frozen VGRA math.
- If justified, prepare a narrowly scoped source patch and negative/positive synthetic tests; verify with SHA-pinned local application transaction and rerun corrected regressions.
- No change, commit, push or real-data training authorized by this record.
- Later native AMP, optimizer, full trainer and compute checks remain open.

## Source / worktree state per user output

`git status --short` showed exactly one modified tracked trainer and five untracked paths (four candidate files and pre-existing `ultralytics/settings.json`):

```text
 M ultralytics/models/yolo/detect/vgra_trainer.py
?? research/tests/test_d4_e2_r2_guarded_unit.py
?? ultralytics/models/yolo/detect/early_paired_trainer.py
?? ultralytics/models/yolo/detect/early_paired_validator.py
?? ultralytics/models/yolo/detect/vgra_fail_closed.py
?? ultralytics/settings.json
```

No Git commit/push or dataset/training access reported. Git HEAD remains unchanged; `git diff --check` produced no errors.

## Source integrity reported after R3-03C correction

| Path | Canonical CRLF→LF SHA256 |
|---|---|
| `ultralytics/models/yolo/detect/vgra_trainer.py` | `9d8e81e00eabf65dd432470935c2384179c0c8a03565de2e9b495d111231b202` |
| `ultralytics/models/yolo/detect/early_paired_trainer.py` | `5b45a2b0aab0a78eb13497138c617163978582a042b1e45a983ad231c22e69b5` |
| `ultralytics/models/yolo/detect/early_paired_validator.py` | `599d1bc94f6122b7f8cce18728b8aaa23848749a97955589be36e5cf322d72b4` |
| `ultralytics/models/yolo/detect/vgra_fail_closed.py` | `fe3124a74d0769bc958df79ea678443ca8f3c4b620e519dd39bebc6b93276221` |
| `research/tests/test_d4_e2_r2_guarded_unit.py` | `36a7a5bb113c9a9c66b15b8b69d6829e8fe5df7bac57b56de32264bc2c4d9be8` |

The completed R2 local evidence-vault capture reverified the five source hashes against the Windows worktree. The exact local ZIP identity and SHA256 are recorded in the verified-vault section below. Scientific EMA status remains HOLD.

## External provenance / backups

User-reported successful R3-03C backup:
- Original script output: `C:\Users\ik_pa\AppData\Local\Temp\D4_E2_R3_03C_PRESERVE_vz0lga5h`
- User also supplied `C:\Users\ik_pa\Downloads\D4_E2_R3_03C_PRESERVE_vz0lga5h`, but its contents were **not verified**. Preserve both until checked and copied to durable storage.

The corrected R2 evidence collector has already copied and hashed the available original terminal logs into the local vault. The earlier R1 collector failure and its R2 recovery remain preserved; the R3-04 numerical test was not rerun for archival recovery.

## Verified local vault binding and collector recovery (2026-10-09)

Vault ZIP: `E:\PhD\Admitted\Research\Project 1\Detection_Evidence_Vault\D4_E2_R3_04_EVIDENCE_20261009_012330.zip`

Bytes: `77695`. SHA256: `f621cd797aea1684887e8667eac1fa3420ee45e596ea5e97f11c892611a15ee0`.

Archive capture: 25 files, zero missing optional; R1 failed due to stdout/stderr routing, R2 passed. R3-03C 17/17 PASS; R3-04 13/13 diagnostic PASS with EMA scientific HOLD. Evidence is user-local; no claimed native trainer or real-data validation.
