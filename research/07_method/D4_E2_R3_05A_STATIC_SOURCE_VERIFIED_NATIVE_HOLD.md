# D4-E2 R3-05A — Windows static native-EMA source review and local evidence sealing

**This record documents local R3-05A static-source verification only; native ModelEMA/AMP/trainer acceptance remains on HOLD.**

## Scope and exact provenance

- Local date: 2026-10-09 (Singapore). Research/scientific worktree `research/vgra-impl-01 @ 48cc059cfa133ae5fbfcec607f15daeda843d3ba`; five local candidate source/test files remain uncommitted; `ultralytics/settings.json` remains untracked.
- Separate documentation branch head observed before proposing this record: `research/vgra-r3-evidence-20261009 @ d0bc750362c98ed0118f89ccf6c838e6ad0fe350`. No docs commit/push authorized by this packet.
- The original read-only R3-05A source auditor (SHA256 `8626f6b067ea7adbd0f4d37909506f629aa6760791cddfb39a8858e0d2dbcbe6`) ran on Windows with exit 0, 10/10 static checks PASS, no Python/project/PyTorch imports, no model or dataset execution, no Git mutations. Reported Python 3.11.9; PyTorch distribution 2.5.1+cu121; Ultralytics distribution 8.4.7.
- Auditor stdout exact reported raw SHA256 `dc5a4173f28794bd1f450922cb1b307f718add84ca4ddb4347c26f8311dc097a`. Empty stderr SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Source verified via normalized LF SHA256: `vgra_trainer.py` `9d8e81e00eabf65dd432470935c2384179c0c8a03565de2e9b495d111231b202`; `early_paired_trainer.py` `5b45a2b0aab0a78eb13497138c617163978582a042b1e45a983ad231c22e69b5`; `early_paired_validator.py` `599d1bc94f6122b7f8cce18728b8aaa23848749a97955589be36e5cf322d72b4`; `vgra_fail_closed.py` `e1947b645202f3d73f6db035ffa673350837bd162ddd2822cc5ee6cd5bde082f`; `test_d4_e2_r2_guarded_unit.py` `36a7a5bb113c9a9c66b15b8b69d6829e8fe5df7bac57b56de32264bc2c4d9be8`.
- Native pinned Git blob checks PASS: `ultralytics/engine/trainer.py` `a3ad643da2141a425b92cee6c51bcd3417ec3506`; `ultralytics/models/yolo/detect/train.py` `074826b7b319d6999d3b2031f9eed23ce7154348`; `ultralytics/utils/torch_utils.py` `4a07f04a2b52992b15bd5f450d5ef838c8daebcb`.
- Exact six-path worktree status matched expectations; no staged files. No source fixes were made in R3-05A.

## Archive collector failure and corrected sealing (preserve both)

- R1 sealer SHA256 `f38d733ab30d090e16e2691b0b9e65a878ee812081b91bd406cd08449f5e9281` FAILED after raw log-hash verification while decoding Windows PowerShell redirected audit stdout as UTF-8; error: `utf-8 codec can't decode byte 0xff in position 0: invalid start byte`.
- R2 sealer SHA256 `4cef2f484166099be56e63d421178d56afb74e57b4c81a3a77de2c5b6a8a660e` recognized UTF-16 LE BOM on the *same original raw log*. Its user-shared Windows execution reported `R3_05A_SEAL_R2_EXIT_CODE=0` and empty stderr; it did not rerun the original audit.
- R2 reported both prior R1 failure logs (`R3_05A_SEAL_STDOUT.txt`, `R3_05A_SEAL_STDERR.txt`) preserved and zero missing failure logs.
- External evidence ZIP (Windows report): `D4_E2_R3_05A_EVIDENCE_20261009T091906Z.zip`, 34000 bytes, 16 members, SHA256 `da8fc99357c0aaa3317ec641d51f545d288d73b019026894f837ae3b6e0ef4e1`; `E:\PhD\Admitted\Research\Project 1\Detection_Evidence_Vault\D4_E2_R3_05A_EVIDENCE_20261009T091906Z.zip`; accompanying `.zip.sha256` file. The 16-member ZIP bytes were **not uploaded into this chat for independent ZIP rehash**; treat the result as a verified-local execution *reported* by the user and do not claim independent remote-byte verification.

## Exact static findings and limitations

1. Native `BaseTrainer._setup_train` constructs `ModelEMA` before calling `resume_training`, and the mixin installs the native optimizer post-step hook after its super setup. Both MV-00 and MV-01 source inheritance chain reaches the guarded mixin; static dispatch PASS is **not** native dynamic dispatch proof.
2. Native `ModelEMA` holds an eval-mode model copy and updates floating-point state-dictionary values, including floating-point buffers. It skips non-floating buffers. Pre-update model and available EMA finite scans are present in the corrected candidate.
3. Pre-update already-poisoned EMA rejection was verified at **source level** (the R3-04B synthetic correction). EMA absent or disabled is not a mandatory abort; no native ModelEMA-backed transaction was executed in R3-05A.
4. Native `BaseTrainer.optimizer_step` orders unscale, clip, GradScaler step/update, gradient clear and EMA update. The guarded mixin's post-step hook detects optimizer method completion but is not mathematical proof of parameter mutation. GradScaler skip/overflow and EMA update-count parity remain separate native checks.
5. Post-update model/EMA detection is fail-closed *after* potential mutation, not rollback. Checkpoint reload, scaler state coverage and actual multi-microbatch gradients remain pending.

## Decision and next scope

`R3_05A_STATIC_PROVENANCE_AND_SOURCE_AUDIT=COMPLETE_LOCAL_VERIFIED`

`R3_05A_LOCAL_VAULT_SEAL=REPORTED_SUCCESS_EXIT_0_SHA256_REGISTERED`

`SCIENTIFIC_GATE=HOLD_NATIVE_MODEL_EMA_AMP_TRAINER_PENDING`

Next: a separately authorized **R3-05B** bounded native `ModelEMA`/`GradScaler` synthetic-only probe design and execution after its own review. Later R3-05C addresses actual production trainer dispatch and accumulation. No source freeze, scientific Git commit, Kaggle/GPU/real image training, B-VAL metric selection or B-TEST access authorized by this document. Registry/docs-branch updates require a separately governed transaction.
