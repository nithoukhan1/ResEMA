# DETECTION PROJECT — CHAT26 C1 FINAL VERIFIED ACCEPTANCE RECORD
Date: 2026-10-22
Status: Two-arm nonofficial native GPU acceptance completed, both epoch-1 and epoch-2 original G4 checkpoint archives independently inspected on a separate Windows host. **No official 100-epoch training has yet been authorized or started. B-TEST is sealed.**

## Foundation
- Cumulative pre-Chat26 master input: `DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT25_FOR_CHAT26_2026-10-21_FINAL.zip` (or PORTABLE variant, only with explicit provenance), supplied among Detection Project sources.
- Detailed Chat26 chronology, incidents, remedial actions, architecture, frozen source, exact input pin list and Kaggle dataset mapping: `research/r9_training/notebooks/CHAT26_C1_NATIVE_GPU_ACCEPTANCE_MIGRATION_ADDENDUM_2026-10-22.md`, GitHub notebook development branch commit `71d984cd6f57356337420a3aed9eed6eb8e886b7`.
- Frozen release `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`, science freeze `b783eb3a470ab010785a8a24a90a666e9a3f2ed6`.
- MV00 = SCConv-Early YOLO11s; MV01 = SCConv-Early YOLO11s + VGRA-V1. Same exact run identities for fresh and resume: `C1ACCEPT_MV00_20261021`, `C1ACCEPT_MV01_20261021`.
- Two physical Tesla T4s, `imgsz=1024`, batch8 each, 100-epoch native config **bounded** to one nonofficial fresh epoch and one nonofficial authentic resume epoch; 14,227 train and 3,049 validation scans. No B-TEST.

## Definitive acceptance matrix

| Evidence | MV00 | MV01 |
|---|---|---|
| Published frozen source + R0 20-pins, 16 path restoration | PASS | PASS |
| Six original SHA-bound runtime inputs | PASS | PASS |
| Epoch 1 training + validation | PASS | PASS |
| Epoch1 G4 original archive SHA | `4fc1c22adda6265a48917de15efea4b1f0766db9a8ed6502aaedfe208e7d44e4` | `5b064fdb8a41a3d525fc6f45650e79bdeb53395fd7fc1a09fcc3d5150a83f4e7` |
| Epoch1 independent Windows original-byte G4 inspection | PASS | PASS |
| Epoch2 explicit RESUME_FROM_OUR_OWN_RUN, native logs `2/100`, validation | PASS | PASS |
| Epoch2 original G4 archive SHA | `9c436620f9b10af927b8f6a964ea4ce666ba81b30174afa59bd5cfd038aa2b22` | `66bc10de6abb9728752550999c70491018c0070ee56b886c156657d839ccc55e` |
| Epoch2 independent Windows original-byte archive and G4 inspection | PASS | PASS |
| Epoch2 zero-based completed epoch | 1 | 1 |
| Checkpoint terminal flag | false | false |

Kaggle epoch2 supervisor ending `DUAL_SESSION_EXIT_CODE: 0`, `LOCAL_G4_EPOCH2_RESUME_ARCHIVE_PASS` for both arms, `BOTH_ARMS_RESUMED_EPOCH2_LOCAL_G4_PASS_OFFSITE_UNVERIFIED_SAVE_KAGGLE_VERSION`. These were **local** assertions; independent Windows preservation was performed subsequently.

## Epoch2 Windows external verification — confirmed 2026-10-22

The user ran `CHAT26_C1_EPOCH2_WINDOWS_VERIFY_R1.ps1` downloaded from development notebook-branch commit `b7838be70586550578025280763b9a55cb4defef`, and first independently confirmed the existing file SHA matched the same pinned GitHub bytes:
`d82c22dcfa680d00860f7854cf35fab62802226abf75966d04e852d6af112cce`.

- Original downloaded **outer epoch2 transfer ZIP** SHA256 (Windows): `59b41703df7a94768751a34647fe3d2133035ca34b9024aa7d8dd21630b0a38d`.
- The outer ZIP was **not compared with a Kaggle-side outer SHA or original matching outer sidecar** (`NOTE_NO_ORIGINAL_OUTER_PIN`), so preserve this caveat. All 12 original inside files passed extracted SHA+size against `TRANSFER_MANIFEST.json`, including both inner archive bytes whose SHA were independently pinned from the earlier Kaggle logs.
- ZIP had exactly 13 safe members (`THIRTEEN_SAFE_MEMBERS_PASS`); 12 original files passed `ALL_TWELVE_ORIGINAL_MEMBERS_SHA256_AND_SIZES_PASS`.
- Pinned, immutable source `R9_OFFSESSION_TRANSPORT_R1.py` fetched from `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3` had Windows SHA256 `765459792d2b4ccaa6e0dab9235c8ca6b8b72b0a039c524c360c2d5e07b88128`.
- **MV00 epoch2 G4 manifest SHA:** `e692921b6e37975f30706d69e4a16591a83788eea6bc40358f0af09e57b640d2`.
- **MV01 epoch2 G4 manifest SHA:** `b5e2bc7c0dae3c99f1c9c2f0dc65a815751a2eb6328e0894cb49ce28eb624da7`.
- Windows original archive and R9 inspector printed `MV00_WINDOWS_EPOCH2_ARCHIVE_AND_INTERNAL_G4_PASS`, `MV01_WINDOWS_EPOCH2_ARCHIVE_AND_INTERNAL_G4_PASS`, and `BOTH_WINDOWS_EPOCH2_OFFSESSION_ARCHIVE_INSPECTIONS_PASS`.
- Evidence folder `C:\Users\ik_pa\Downloads\CHAT26_C1_EPOCH2_WINDOWS_VERIFIED_R1`; report `CHAT26_C1_EPOCH2_WINDOWS_VERIFICATION_R1.json` under that folder; preserve separately.
- The inspector's `checkpoint_sha256` is **`selected_best.pt`**, **not** necessarily `last.pt` (verified directly from pinned `R9_OFFSESSION_TRANSPORT_R1.py` in release). Values reported:
  - MV00 selected_best SHA `0462acc851b91a9db49669b868c6dd2a4cb959cde1c71beed5cf5d282c2ba193` (unchanged from epoch1).
  - MV01 selected_best SHA `34579d7ffdc279d5c38419c09f29b4ae30f0d9f5038adc667587c7768e639aba` (unchanged from epoch1).
  - This identity alone **does not mean epoch2 failed to train**. Selected best can remain unchanged if epoch2 did not improve validation fitness. For additional release QA, inspect the SHA of `last.pt` across epoch1 and epoch2 as a separate check; do not confuse selected best with last. Native logs independently show epoch2 training executed and all full native G4 custody checks passed.
- `offsession_persistence_confirmed:false` is an **intentional verifier limitation**, not a report that no Windows copy exists. The verifier does byte-level original archive checks, not independent GPU reproduction or automated external-storage certification.

## Official release requirements (still pending)

- Candidate frozen source release is a **nonofficial acceptance release**. Its original approval gates have explicit `official_training_authorized=false`.
- A **separate, reviewed official production release/authorization path** must be established without mutating the science-frozen source in place. Must preserve independent MV00/MV01 run identities, training config, resume and external preservation, epoch budget, reproducibility, and sealed B-TEST.
- Nonofficial epoch1+epoch2 checkpoints are NOT official epoch1+epoch2 of the future 100-epoch experiment. Do not resume official training from them without separate release authorization and study-protocol rationale.
- No comparative scientific finding or VGRA benefit is established by these path acceptance runs; comparative reporting requires properly authorized full training/validation and follow-up research design.
- Future number of 100-epoch Kaggle sessions cannot be promised in advance; bounded GPU sessions and externally verified checkpoints may be necessary due to quota.
- Do not start long running GPU work until master Chat1–26 migration and C1 release gate checklist are both reviewed.

## Chronology of Chat26 notebooks and scripts

1. One-click fresh: `research/r9_training/notebooks/CHAT26_C1_ONE_CLICK_NONOFFICIAL_FRESH_EPOCH1_T4X2.ipynb`, commit `ef7c3cd56abaebd3711a14c35d9ca344fae4e824`.
2. Pinned epoch1 Windows verifier: `research/r9_training/notebooks/CHAT26_C1_EPOCH1_WINDOWS_OFFSITE_VERIFY_AND_STAGE.ps1`, commit `96613d99841d994a5d35d6433dce65821200b446`, script SHA `6473887c1d75d2ef2e98865bccc23c0f99b07612603235a7d87d78ef92e45538`.
3. One-click native epoch2 authentic resume: `research/r9_training/notebooks/CHAT26_C1_ONE_CLICK_NONOFFICIAL_RESUME_EPOCH2_T4X2.ipynb`, commit `4e367ffdac2bc7da0f796ce4550ad3263857a62f`.
4. Epoch2 Windows inspector: `research/r9_training/notebooks/CHAT26_C1_EPOCH2_WINDOWS_VERIFY_R1.ps1`, commit `b7838be70586550578025280763b9a55cb4defef`, script SHA `d82c22dcfa680d00860f7854cf35fab62802226abf75966d04e852d6af112cce`.
5. Chat26 initial archive/history addendum: `research/r9_training/notebooks/CHAT26_C1_NATIVE_GPU_ACCEPTANCE_MIGRATION_ADDENDUM_2026-10-22.md`, commit `71d984cd6f57356337420a3aed9eed6eb8e886b7`.
6. This **final verification** record supplements item 5; both are required in next cumulative migration.

## Migration handoff and next task

Assemble the previous full Chat1–25 migration archive **unchanged**, include both Chat26 addenda, first prompt for Chat27, a provenance manifest with SHA-256 of every added file, and the independently produced Windows verification JSON and two attestation JSON files when available. The original multi-hundred-MB `.zipblob` checkpoint files are preserved separately; do not conflate a small migration ZIP with external original checkpoint storage. Mark the new migration as complete only after actual ZIP generation and independent hash verification.

**Next Chat27 research task**: review complete cumulative migration and newly proven dual-epoch acceptance; audit exact frozen-source official release gate and prepare a separately authorized fresh two-arm 100-epoch plan with per-session resume/verification checkpoints. Neither the migration nor this note constitutes authority to execute official training.
