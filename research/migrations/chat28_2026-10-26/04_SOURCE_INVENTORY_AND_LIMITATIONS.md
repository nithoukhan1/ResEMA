# Chat28 Migration — Source, Provenance, and Inclusion Limits

## Two independent evidence repositories
1. **ChatGPT Detection Project historical sources:** existing user-supplied `DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT26_FOR_CHAT27_2026-10-22_PORTABLE(1).zip`, **30,515,326 bytes**, stored in this Project's attachments. The historical portable ZIP includes prior nested chat migration history. Its raw binary is NOT available inside the GitHub checkout, and this GitHub-produced migration archive DOES NOT include it.
2. **Public GitHub nithoukhan1/ResEMA:** source code, frozen scientific C1 implementation, Git-tracked experiment/provenance ledgers, Chat27 DDP branch and full migration handoff docs. The GitHub Actions archive includes the complete current tracked checkout snapshot (except Git metadata), plus original historical Chat26 handoff and checkpoint registry copied verbatim from project attachments.

**Do not describe these as one byte-for-byte historical migration ZIP.** To recreate the full nested prior archive in a new context, retain/upload the prior 30.5MB portable source ZIP in addition to this Chat28 research/code ZIP, or open both from the same Detection Project's Sources. The new ZIP is comprehensive for GitHub-tracked code and the summarized handoff, not for original raw chats/assets missing from GitHub.

## Explicit exclusions
- Original raw Kaggle/Windows G4 checkpoint binaries `.zipblob`, epoch checkpoint `.pt`, source-side private manifests, raw patient X-ray images, Kaggle private source dataset. Do NOT fabricate or claim embedded.
- Immutable source assets and trained weights whose hashes are recorded but bytes are not in the Git checkout.
- Complete private conversation transcripts from 27 chats, apart from historical handoff narrative. The older 30MB portable archive is the authoritative raw historical carrier.

## Full project evidence path
- Previous attached `HISTORICAL_MASTER_HANDOFF_CHAT1_TO_CHAT26_2026-10-22.md` transcribed byte-for-byte (text; original line-end normalization possible).
- Previous attached `CHAT26_CHECKPOINT_REGISTRY_SOURCE_2026-10-22.json` copied from source.
- GitHub repository source snapshot includes all frozen `research/r9_training/frozen_source/` tracked files, DDP code/test files, research results/tracker docs and complete workflow configuration. Original `.git` excluded, but Git provenance exported to plain-text logs, commit hash and branch list in package.
- DDP final verified code SHA **1b1696d3d77cf327452bcd40cb236c05c91098a2**; subsequent migration doc commits are documentation-only.
- Scientific frozen C1 SHA **fa6a7675daae385a04c7b77e69a9de29f9e1b2f3**.
- All stage/status and SHA attestation notes must be read in the source docs and verified against GitHub, rather than trusting filenames alone.

## Integrity requirements for packaged archive
The GitHub Actions bundler must record HEAD, `git log --all` text, deterministic per-member SHA256 JSON, ZIP CRC check, and outer archive SHA256. A valid completed workflow and generated artifact count as actual packaging evidence. A locally constructed filename without created bytes is NOT an archive.

## Kaggle resource and historical result limits
The C1 acceptance Epoch1/2 was independent single-T4-per-arm, not DDP. The DDP design's global16 two-T4-per-arm is an untested new condition. Historic C1 nonofficial engineering metrics cannot be moved into the DDP official comparison. Kaggle two notebooks simultaneously require actual permitted resource availability (total four T4).
