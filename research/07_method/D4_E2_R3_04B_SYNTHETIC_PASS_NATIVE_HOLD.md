# D4-E2 R3-04B — Local synthetic EMA correction: PASS; native integration: HOLD

## Scope and scientific disposition

- Date: 2026-10-09.
- Scientific source: `research/vgra-impl-01 @ 48cc059cfa133ae5fbfcec607f15daeda843d3ba`; exact five-file local candidate remains uncommitted.
- Modified path only: `ultralytics/models/yolo/detect/vgra_fail_closed.py`, canonical LF SHA256 `e1947b645202f3d73f6db035ffa673350837bd162ddd2822cc5ee6cd5bde082f` after the guarded R3-04B patch.
- New behavior: reject already-corrupt EMA parameters and buffers **before** the synthetic optimizer step. Original post-update checks remain unchanged.
- Corrected EMA synthetic probes: **17/17 PASS**, exit 0 (PyTorch 2.5.1+cu121 on CPU, real SGD, stub EMA and scaler).
- Pre-existing guarded regression suite: **17/17 PASS**, exit 0, including post-accumulation overflow detection.
- Negative R3-04 observation remains historical; its old 13/13 diagnostic PASS did **not** imply safety acceptance.
- **Decision:** synthetic correction accepted only. `SCIENTIFIC_GATE=HOLD_NATIVE_MODEL_EMA_AMP_TRAINER_PENDING`.

## Durable local evidence identity

| Field | Exact value |
|---|---|
| ZIP | `D4_E2_R3_04B_EVIDENCE_20261009_120417.zip` |
| External location | `E:\PhD\Admitted\Research\Project 1\Detection_Evidence_Vault\D4_E2_R3_04B_EVIDENCE_20261009_120417.zip` |
| ZIP SHA256 | `0cdb58ffec2f55bbe58064b5fe1d6739bdc748f23eaea670ab1469a141da629a` |
| ZIP size | `56195` bytes |
| ZIP membership | 24 members; ZIP CRC, internal SHA256 and manifest bindings checked by R2 |
| R2 sealing log | `D4_E2_R3_04B_SEAL_R2_20261009_123533.txt` (vault root) |
| Log SHA256 | `0b5d0200a259a5fe23ca9d8f558599de4eb9451f36cc2d072dd3411343760d25` |
| R3-04B execution transcript SHA256 | `3f0675080c611ad0fd84a3e8b170c01cb88c17ca40db0ae09304c03dbebd956f` |

The original source raw backup, both test streams, four executed scripts, the R1 archival-failure record, R2 corrected sealer, source snapshots and evidence manifest are protected by ZIP member hashes. The ZIP is stored **outside Git**. Remote Git contains only its index and scientific interpretation, not the raw archive bytes.

## Collector failure and recovery (do not erase)

R1 demanded the literal `R3_04B_SOURCE_APPLICATION_PASS` marker in the PowerShell transcript and stopped. R2 identified transcript encoding `UTF-8-BOM`, detected only **4/6** direct markers, and retained the missing-marker information. Acceptance of archival capture required full six-stage PowerShell host chronology, the final transaction PASS, independently hashed redirected raw test logs, five exact source hashes and the prepatch raw backup. R2 succeeded without rerunning any test, training or dataset operation. This archival recovery does **not** establish native scientific safety.

## Earlier R3-04 documentation branch provenance

The R3-04 evidence-only documentation commit is `a89f58783dc5dd0a804c52e17c0b50490451c346`. Its R1 commit guard scope-check failure, R2 staged-whitespace failure, R3 recovery and R4 verified remote commit are documented in append-only `research/PROJECT_LOG.md`; their original local scripts and recovery logs remain in the user evidence vault. This follow-up record is prepared on that documentation branch rather than modifying the uncommitted scientific checkout.

## Open R3-05 audit gates

1. R3-05A: inspect true Ultralytics `ModelEMA` initialization, `.ema` state, update schedule, and failure timing; verify buffer coverage.
2. R3-05B: inspect native PyTorch AMP/GradScaler and optimizer semantics, including skipped steps and post-hook ordering. Use synthetic CPU only if later separately authorized.
3. R3-05C: verify the actual trainer dispatch path, multi-microbatch gradient accumulation and fail-closed first-update semantics. No real TRAIN/VAL images or B-TEST.

**No D4-E source freeze, D4-F GPU/real-data training, Git scientific source commit/push, or B-TEST access is authorized.**
