# Detection Project — Chat 26 C1 Native GPU Acceptance Handoff (2026-10-22)

**Status:** ACCEPTANCE SUCCESS through **nonofficial fresh epoch 1** and **nonofficial authentic epoch-2 resume** on both physical Tesla T4 GPUs. Epoch-1 off-session verification succeeded on Windows. Epoch-2 local G4 exports succeeded; **independent Windows byte-level verification remains PENDING until the user executes the pinned verifier and reports its output**. This handoff is an additive Chat26 record, **not yet** the full Chat1–26 migration ZIP.

## Governance and scientific scope

- Active line for the most recent C1 acceptance: **MV00 = SCConv-Early YOLO11s**, **MV01 = SCConv-Early YOLO11s + VGRA-V1 residual classifier-logit correction**. Historic C3k2 + DySample + ResEMA work remains part of the longer project's prior-experiment record; no scientific claims should be inferred from C1 acceptance.
- Published immutable training source: branch `research/vgra-r9-training`, release commit `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
- Frozen science provenance: `b783eb3a470ab010785a8a24a90a666e9a3f2ed6`.
- C1 official integration **not authorized**: `official_training_authorized=false`, `B-TEST access=NONE`, `run_class=NONOFFICIAL_FINALPATH_ACCEPTANCE`. These are GPU-path acceptance tests, not the first two epochs of an authorized official 100-epoch experiment.
- Both bounded native workers retain original `epochs=100` model-training configuration and stop at an authenticated G4 boundary; fresh acceptance targets epoch1 and resumed acceptance targets epoch2, with the same original run IDs.
- Runtime recipe evidenced in live log: `imgsz=1024`, `batch=8`, seed42, optimizer SGD, base Ultralytics 8.4.7, CUDA Tesla T4 and two independent workers (`CUDA_VISIBLE_DEVICES` separate). Do not use these logs as proof of published comparative performance.
- Split-B runtime train 14,227 cases/images and runtime validation scanned 3,049, train/val only. B-TEST sealed, never accessed in these acceptance tests.

## Environment and frozen input identity

- Kaggle real checkout: `/kaggle/working/c1_official_fixed_checkout` (not a symlink).
- Input six-file exact pin mapping:
  - Runtime data YAML SHA `f2cc6e31f5f1be152f4dd7603a397e142963923f1d35de613eb7c61b168bf0c0`;
  - VGRA assignment SHA `87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4`;
  - visibility SHA `9de034fa7df668d17effa843674c097341af9dd6461963f9e2c454c5f358b289`;
  - Early pretrained best SHA `620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1`;
  - Early pretrained archive SHA `4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396`;
  - R7D original recovery config by original path under `nithoukhan2/detect-1/R7D_RESTORE_CONTEXT_SUPPORT_R1/inputs/RECOVERY_CONFIG.json`.
- Corrected Chat25 restager `CHAT25_C1_R0_STALE_PATH_SAFE_RESTAGE_CORRECTED.py` SHA `88cd4a88c373f451e54ba64736e961e735757d94b3779feea55a526868c45f4f`; 20 source pins and 16 stale paths restaged. Both MV00 and MV01 passed R0 checks.

## Timeline, failures, fixes and successful evidence

1. **Initial manual/duplicated notebook attempt** raised `AssertionError` on guarded fresh-output directories after a previously performed preparation pass. Do not remove that guard, reuse dirty output roots, or repeatedly run all cells in an occupied session.
2. **First actual nonofficial GPU attempt** launched both workers and reached train label scanning. Interrupted near 2,172/14,227 labels; MV01 ended with `KeyboardInterrupt`. No validated G4 or epoch completion. Incident ZIP: `CHAT26_C1_EPOCH1_INTERRUPTED_INCIDENT.zip` SHA `c3fab0e7806960bfb34711bdd1b8ad9478f57f5bb9ff3a8b0a730c0c805d0fc4` (created in Kaggle; external preservation of this incident not asserted).
3. **Streamlined Version1 fresh epoch1 notebook** at GitHub notebook branch commit `ef7c3cd56abaebd3711a14c35d9ca344fae4e824`, path `research/r9_training/notebooks/CHAT26_C1_ONE_CLICK_NONOFFICIAL_FRESH_EPOCH1_T4X2.ipynb`. Live on Kaggle Oct21: R0, six pins, one epoch per model + validation, both local G4 exports PASS, `DUAL_SESSION_EXIT_CODE:0`:
   - MV00 original epoch1 G4 ZIP blob SHA `4fc1c22adda6265a48917de15efea4b1f0766db9a8ed6502aaedfe208e7d44e4`.
   - MV01 original epoch1 G4 ZIP blob SHA `5b064fdb8a41a3d525fc6f45650e79bdeb53395fd7fc1a09fcc3d5150a83f4e7`.
4. **Windows epoch1 archive preservation and independent verification**: downloaded one outer `CHAT26_C1_EPOCH1_OFFSITE_TRANSFER.zip` of 465,613,445 bytes; Windows SHA `eb432867d3640ce267458f599ec7620e60f7cd40a2db9f9af2dc1c87d683f75d`. Original archives+sidecars+index extracted to Windows and inspected with exact R9 pinned source. The Windows verification script exact source hash `6473887c1d75d2ef2e98865bccc23c0f99b07612603235a7d87d78ef92e45538`; both original G4 transport records PASS.
   - MV00 epoch1 manifest pin `b6de5a6f6463e91fcfb0ef076755849e1f5b503716be04f0efe7a89fdad8c6e7`; selected_best checkpoint SHA `0462acc851b91a9db49669b868c6dd2a4cb959cde1c71beed5cf5d282c2ba193`.
   - MV01 epoch1 manifest pin `a1a5bc59a572a4a263588dc9fcb6184e957581bf41a174bb40fb5bcbccb1d810`; selected_best checkpoint SHA `34579d7ffdc279d5c38419c09f29b4ae30f0d9f5038adc667587c7768e639aba`.
   - Windows-verified private Kaggle resume staging outer archive `CHAT26_C1_EPOCH1_VERIFIED_FOR_KAGGLE.zip` size 444.13MB, SHA `993ffc04e8a637b9b12db070dec04567ed08a2fcd29e774cb95dcd37f4ecb556`; Kaggle initially rejected backslash-containing directory names inside Windows-created ZIP. Fixed by producing a new **flat seven-file ZIP**, preserving individual original blob bytes and attestation JSON. Original Kaggle private dataset mount `/kaggle/input/datasets/nithoukhan/chat26-c1-epoch1-kaggle-upload-fixed-r1`.
5. **Genuine native epoch2 resume** using both independently verified epoch1 original archives + sidecars + Windows attestations; notebook at GitHub commit `4e367ffdac2bc7da0f796ce4550ad3263857a62f`, `research/r9_training/notebooks/CHAT26_C1_ONE_CLICK_NONOFFICIAL_RESUME_EPOCH2_T4X2.ipynb`. On 2026-10-22 both mounted G4 SHA prechecks, R0, six pins, and `RESUME_EVIDENCE_MOUNT_GATE BOTH_ARMS_PASS` succeeded; actual training reached native `2/100`, validation, `DUAL_SESSION_EXIT_CODE:0`. Both G4 epoch2 original ZIP locally verified:
   - MV00 epoch2 original `MV00_epoch002_G4_LOCAL_EXPORT.zip.zipblob` SHA `9c436620f9b10af927b8f6a964ea4ce666ba81b30174afa59bd5cfd038aa2b22`.
   - MV01 epoch2 original `MV01_epoch002_G4_LOCAL_EXPORT.zip.zipblob` SHA `66bc10de6abb9728752550999c70491018c0070ee56b886c156657d839ccc55e`.
   - Final `BOTH_ARMS_RESUMED_EPOCH2_LOCAL_G4_PASS_OFFSITE_UNVERIFIED_SAVE_KAGGLE_VERSION`. **Do not mark epoch2 externally verified until the Windows follow-up below succeeds**.
6. **Epoch2 transfer** completed by user: `C:\Users\ik_pa\Downloads\CHAT26_C1_EPOCH2_OFFSITE_TRANSFER.zip`. It was created from 12 original files + a transfer manifest, for 13 expected ZIP members (index, original .zipblob and matching .sha256 per arm, plans/approval and logs, and transfer manifest). The epoch2 outer SHA was not provided in conversation and MUST NOT be invented.

## Current required next step: Windows epoch2 independent proof

- Pinned Windows verification script (not yet run): `research/r9_training/notebooks/CHAT26_C1_EPOCH2_WINDOWS_VERIFY_R1.ps1` GitHub notebook branch commit `b7838be70586550578025280763b9a55cb4defef`. This checks archive member inventory, every member's manifest hash/size, output-index governance and run identity, both original G4 archive exact hashes/sidecars, and independently runs immutable `R9_OFFSESSION_TRANSPORT_R1.py verify` on Windows to create two external nonofficial G4 attestation JSON files.
- The stage is PASS only when the user provides the produced Windows terminal output `BOTH_WINDOWS_EPOCH2_OFFSESSION_ARCHIVE_INSPECTIONS_PASS`, two arm PASS statuses, the outer SHA, and evidence report.
- Keep the user's original ZIP unchanged; Windows script creates a fresh evidence directory, refuses overwrites and never launches GPUs.

## Official training launch gate — UNRESOLVED

The C1 published frozen release is explicitly integration-candidate/nonofficial. It has not authorized official 100-epoch training, despite successful technical GPU acceptance. A separate release/approval decision with exact intended production source commit, input pins, epoch quotas, external preservation mechanics, run provenance and B-TEST sealing is necessary. Do **not** mistake the acceptance epoch 1+2 for the official 100-epoch run; do **not** start an official run from these nonofficial checkpoints without a separately reviewed policy.

## Migration checkpoint

This additive handoff captures **Chat26** but is not a substitute for the user-provided cumulative Chat1–25 migration archive `DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT25_FOR_CHAT26_2026-10-21_FINAL.zip`. Once epoch2 Windows verification passes, combine previous master migration + this Chat26 supplement + final Windows verification records, hashes, GitHub provenance, experiment registry, outstanding release risks, and Chat27 first prompt in one comprehensive migration. **No master Chat1–26 ZIP is claimed to have been generated yet.**
