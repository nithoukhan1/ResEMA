# D4-D3 — Conditional CPU/synthetic verification acceptance record

> **CONDITIONAL ACCEPTANCE; EFFECTIVE ONLY AFTER VERIFIED REMOTE GIT TRANSACTION.**
> Preparation, local staging, or local commitment alone do not establish acceptance. The exact documentation-only commit must be independently verified at the remote branch. **No D4-E source freeze, GPU/real-data training, or B-TEST authorization.**

**Prepared:** 2026-10-08.
**Research branch and checked parent HEAD:** `research/vgra-impl-01 @ aecaaadbc60c075c8a0965e5d80fd27650093196`.
**Evidence:** `D4_D3D1_ACCEPTANCE_EVIDENCE_R1.zip` (external, not Git-tracked), SHA256 `23558c8bf26e76e861b27c7d90aea279f7d5b4d2e30f2ac4896e8e088cad497f`; contains replay manifest, individual stdout/stderr logs and internal SHA256SUMS.
**Decision:** A1 approved, B1 conditionally approved, as documented in `D4_D3_SCIENTIFIC_DISPOSITIONS_APPROVED.md`.
**Related existing frozen authority:** D4-B2C2 VGRA mathematical specification, D4-B2C2 ablation/promotion contract, D4-C6 open-gates checklist, D4-D2 accepted transfer record and gate JSON.

## 1. Accepted scope *if and only if* independent remote Git verification succeeds

Candidate status: `COMPLETE_CONDITIONAL_CPU_SYNTHETIC_VERIFICATION`.

This is a bounded independent **CPU synthetic/pretrained implementation verification** covering numerical origin diagnostics, exact checkpoint transfer, pair-preserving persistent sampling, two synthetic optimization epochs and EMA, isolated native NMS/metrics parity, companion isolation, signed-beta mathematics, MV-00/MV-01 common pipeline mechanical parity, and full actual VGRA validator `__call__` on a synthetic trainer stub and in-memory dataset. It is **not** a claim of clinical utility, dataset-level training, D4-F readiness, mAP improvement or final architecture.

## 2. Independent immutable evidence review

1. The replay evidence archive's SHA256 is `23558c8bf26e76e861b27c7d90aea279f7d5b4d2e30f2ac4896e8e088cad497f` and its internal ZIP CRC passes.
2. All **22 archive members** listed in `SHA256SUMS.txt` matched their recorded member hashes (10 stdout, 10 stderr, replay manifest and review summary).
3. The replay manifest reports **10/10** processes with exit code zero, matching required success markers, no missing markers, and clean repo worktree after every step.
4. D3A1 returned exit code zero **because the diagnostic deliberately reproduced a known failure**, not because fresh initialization was numerically safe; the nonfinite observation must be retained in B1 risk documentation.
5. D3B2 **R1** was an audit-harness failure (`DetectionValidator.end2end` uninitialized). The corrected **R2** uses the native `init_metrics` before `postprocess`; replay R2 is pinned. Neither is evidence of a model-source defect.
6. The source remained `aecaaadbc60c075c8a0965e5d80fd27650093196` throughout replay. The report is fully based on CPU synthetic inputs; no B-TRAIN/B-VAL/B-TEST image data were accessed in the replay.

### Exact replay scripts and outcomes

| Audit step | Recorded script SHA256 | Exit | Evidence status |
|---|---|---|---|
| D3A1 | `11e85038a6b2cafeaaf4ea88bd7a9e98089004feb4460854a08dc83ac9179dec` | 0 | PASS |
| D3A2 | `5ad1c2782faae9bc4279eec87d47f8124e79b678dda0e637344978d1d3cda952` | 0 | PASS |
| D3A3 | `3550fa2a08d2bcd12929ab26dba4cec8d0dda47a88bb3aba006e8560d96419ad` | 0 | PASS |
| D3A4 | `2d4deeef0c90b138deb8e922417911830b933ffe94715d74e2e495c665e5bc3a` | 0 | PASS |
| D3A5 | `38cd8c71b0180853f447caecfa31db551601a200b52d186e1b709506d829af68` | 0 | PASS |
| D3B1 | `ac55a3448a4b5422f1f8fd1b1d4ebe727f5825ddea785ee0ff6c75f5539d9f3c` | 0 | PASS |
| D3B2_R2 | `daad8ecd2fe592f24ec208120037282c9b4d5b13c97203bbcc4c26ad7d949a8f` | 0 | PASS |
| D3C1 | `3e5f2c02b9f001467dfabc3d0e26c7da26be5b1c04b9e09f3ad8e62722c41d70` | 0 | PASS |
| D3C2 | `27b3737f6c93e3bfdf7716920ed35ac703bac53db091e259e558f748a6237959` | 0 | PASS |
| D3C3 | `0e87f207437b3d6f74cb7f2340766286eb0a5430f39396cfc7d6a3877594217b` | 0 | PASS |

The authoritative replay manifest JSON values are preserved unchanged in `D4_D3_INDEPENDENT_REPLAY_MANIFEST.json`; Git normalizes the original CRLF line endings to LF under the frozen `.gitattributes` policy. The original external evidence ZIP remains SHA256-identical. The manifest retains historical `PENDING_FORMAL_AUTHORITY` decision fields at replay time; subsequent user approvals are documented separately.

## 3. Technical stage-by-stage findings

| Stage | Verified technical finding | Boundary |
|---|---|---|
| D3A1–A4 | Fresh-init zero-image nonfinite backward gradients identified and isolated, including an FP32 BN overflow boundary. | This is a **confirmed hazard**, not an all-pass stability assertion. |
| D3A5 | Verified pretrained Early source gave finite gradients in eight synthetic control cases. | Small synthetic CPU case matrix only. |
| D3B1 | Two synthetic epochs, six optimizer updates, EMA save/reload, persistent pair-preserving loader, finite post-update all-zero and zero-label cases. | No actual B-TRAIN; cannot extrapolate long-run numerical stability. |
| D3B2 R2 | Native zero-rho decoded/NMS identity, controlled synthetic TP/FP/FN metrics, nonzero-rho companion influence and independent-single isolation, box firewall. | Synthetic scores are **not** research performance metrics. |
| D3C1 | Signed-beta math, four-state coefficients, detach policy and nonzero rho derivative verified. | Negative beta reverses nominal polarity (A1 explicitly accepts this). |
| D3C2 | Equivalent pair-safe augmentation, paired sampler and shared pretrained native loss at rho zero for both pipeline arms. | Full MV-00/MV-01 real-run recipe parity deferred to D4-E. |
| D3C3 | Real validator `__call__` completed synthetic two-batch six-image execution, native metrics and all four loss components. | Synthetic trainer stub rather than full production trainer; no medical images. |

## 4. A1/B1 conditional dispositions (explicitly approved)

- **A1** `APPROVED_RETAIN_SIGNED_WITH_EXPLICIT_POLARITY_REVERSAL`: retain frozen `beta_l=2*tanh(rho_l)` and exact class-wise visibility coefficient semantics. No model changes. Research description must not falsely claim always-positive cross-view assistance.
- **B1** `CONDITIONALLY_APPROVED_PRETRAINED_PATH_ONLY_GUARDS_MANDATORY`: preserve fresh-init failure, pin pretrained `best.pt`/archive SHA256 and 537/537 tensor transfer, and mandate fail-closed finite-loss, gradient, parameter/buffer and optimizer/EMA safeguards in the D4-E protocol before any D4-F training authorization. Scratch-only initial MV-01 is not authorized. The later safeguard implementation requires independent verification.

## 5. Explicit outstanding obligations and firewalls

- **D4-E:** separate transaction to freeze exact MV-00/MV-01 dataset bindings, shared pair-safe preprocessing, seed, image size, batch, optimizer, LR schedule, epoch budget, checkpoint selection and evaluator; verify TRAIN-only weights and actual dataset hashes; implement, test and register finite guards; freeze source only after its own review.
- **D4-F:** no training until D4-E and explicit separate authorization.
- **Sealed B-TEST:** no access, including diagnostic, selection or ad hoc validation uses.
- **Novelty/results:** no empirical improvement, clinical applicability or final paper-architecture claim; global architecture remains unfrozen.
- **Production trainer:** not certified by D3C3 stub and two separate CPU component tests; implementation-run equivalence and real training controls belong to the D4-E/F contracts.

## 6. Transaction boundaries

Only these **four newly added documentation files** may appear in the proposed D4-D3 acceptance commit:

1. `research/07_method/D4_D3_SCIENTIFIC_DISPOSITIONS_APPROVED.md`
2. `research/07_method/D4_D3_FORMAL_ACCEPTANCE.md`
3. `research/07_method/D4_D3_GATE_RECORD.json`
4. `research/07_method/D4_D3_INDEPENDENT_REPLAY_MANIFEST.json`

**No edits** to `ultralytics/**`, `research/07_method/D4_D2_GATE_RECORD.json`, original B2C2 specifications, assignment manifests, model YAML, data files, or training scripts. The four-file staging transaction does not commit or push. Precommit review must verify source remains pinned and the file inventory exactly matches the allowlist. The acceptance becomes valid only after human inspection, a separately authorized narrowly controlled commit, and independent **remote Git** verification.

## 7. Acceptance condition and immutable no-go restrictions

`D4_D3_ACCEPTANCE_EFFECTIVE_IF_AND_ONLY_IF=EXACT_FOUR_DOC_COMMIT_VERIFIED_AT_REMOTE_HEAD`
`D4_D3_PREPARATION_AND_STAGING_DO_NOT_CONSTITUTE_ACCEPTANCE=TRUE`
`D4_E_SOURCE_FREEZE=NOT_AUTHORIZED`
`GPU_TRAINING_AUTHORIZED=FALSE`
`B_TEST_ACCESS=NONE`
`MIGRATION_TRIGGER=AFTER_REMOTE_VERIFIED_D4_D3_COMMIT_BEFORE_D4_E`
