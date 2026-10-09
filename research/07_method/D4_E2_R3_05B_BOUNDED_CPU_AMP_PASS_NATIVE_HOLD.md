# D4-E2 R3-05B — Bounded native CPU AMP components: PASS; production trainer: HOLD

## Scientific decision

R3-05B is accepted **only for the bounded nine-case synthetic CPU component scope**. Independently audited uploaded evidence supports 9/9 recorded PASS, zero FAIL. It is neither full `BaseTrainer` integration nor training authorization.

## Evidence and provenance

- Source: `research/vgra-impl-01 @ 48cc059cfa133ae5fbfcec607f15daeda843d3ba`; the local scientific candidate is uncommitted.
- Original ZIP: `E:\PhD\Admitted\Research\Project 1\Detection_Evidence_Vault\D4_E2_R3_05B_CPU_NATIVE_AMP_R1_20261009T111650Z.zip`.
- SHA256: `d85587a4d263ea82ca5b2e3aa7b10bca0890c339a91aab35b5aa4f42dcfa82df`; **57,318 bytes; 13 members**, CRC PASS; internal manifest 12/12 PASS.
- Exact executed script SHA256: `43fcff66d8f6e0cfeac491ed25704a9e2862e39232642a0e038633415c25dfa3`; five candidate LF SHA256 5/5 and three native Git blob SHA1 3/3 PASS.
- Windows stdout SHA256: `b5c0b1683a63d523600692869cd565ee3cd2f33f71f30bac81fd9845f308dfa9`; stderr empty SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. These raw logs exist separately in the user vault, **not in the original 13-member ZIP**.
- Independent audit record: `D4_E2_R3_05B_INDEPENDENT_AUDIT_AND_R3_05C_HANDOFF_2026-10-09.zip` (Chat 15 output). Uploaded original ZIP was byte-verified independently; Windows tests were not rerun here.

## Cases and meaning

| ID | Outcome | Observed / limitation |
|---|---|---|
| AMP-00 | PASS | PyTorch 2.5.1+cu121, native CPU GradScaler enabled, SGD fixture |
| AMP-01 | PASS | Finite parameter 1.0 to about 0.8, one optimizer post-hook, one EMA update; authored `events` list is NOT a full measured trace |
| AMP-02 | PASS | Nonfinite gradient -> 0 optimizer steps, scaler 8 to 4 |
| AMP-03 | PASS | Guard rejected bad gradient before step in extracted fixture |
| AMP-04 | PASS | One optimizer step invocation may produce zero parameter change |
| AMP-05 | PASS | Already-poisoned EMA parameter and floating buffer blocked before optimizer step |
| AMP-06 | PASS | Native `ModelEMA` updated floating state, not integer buffer |
| AMP-07 | PASS | Injected post-model/post-EMA faults detected, without rollback |
| AMP-08 | PASS (inspection) | `optimizer=auto` can select AdamW or MuSGD based on estimated iterations; neither executed |

## Open native-integration gates

- Production `BaseTrainer._setup_train` and `_do_train` are **not** tested; MV-00/MV-01 dynamic runtime dispatch and gradient accumulation remain OPEN.
- CPU `GradScaler` does not prove CUDA `GradScaler`, final optimizer handling, or AMP-aware MuSGD/fused step behavior.
- EMA absent/disabled policy unresolved; optimizer step hook means invocation, not numerical change; post-step rejection is not rollback.
- R3-05C must capture **actual instrumented runtime call traces**, paired recipe parity, bounded synthetic microbatch accumulation, negative controls, source/optimizer provenance, and SHA256-sealed outputs. R3-05C is separately governed; not executed by this record.

**Scientific gate:** `HOLD_NATIVE_MODEL_EMA_AMP_TRAINER_PENDING`.

**Forbidden here:** scientific source edits, model source freeze, Git scientific commit/push, Kaggle/GPU/real data training, B-VAL metric selection, or sealed B-TEST access.
