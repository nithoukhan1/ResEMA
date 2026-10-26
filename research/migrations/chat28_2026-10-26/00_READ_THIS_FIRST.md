# DETECTION PROJECT — CHAT 28 MASTER MIGRATION (26 OCTOBER 2026)

**Scope:** cumulative *research and implementation* migration, Chats 1–27; intended next chat Chat 28.

## Mandatory reading order
1. `01_MASTER_HANDOFF_CHAT1_TO_CHAT27.md` — present research baseline and history, scientific decisions, GitHub references.
2. `02_DDP_B16_IMPLEMENTATION_AND_EVIDENCE.md` — new Chat27 DDP source-level implementation, tests, failures, and outstanding blockers.
3. `03_NEXT_STEPS_AND_GATES.md` — the sole active engineering sequence; no extra model variants.
4. `04_SOURCE_INVENTORY_AND_LIMITATIONS.md` — precise included/excluded evidence and how to validate.
5. `HISTORICAL_MASTER_HANDOFF_CHAT1_TO_CHAT26_2026-10-22.md` and `CHAT26_CHECKPOINT_REGISTRY_SOURCE_2026-10-22.json`.
6. Complete `REPOSITORY_SNAPSHOT/` research code, scientific frozen release, historical experiment ledgers, branch-specific DDP files and GitHub workflows.
7. `CHAT28_FIRST_PROMPT_2026-10-26.txt`.
8. Verify the archive's SHA256 and the per-member manifest before use.

## Critical priority
- Current active science: **SCConv-Early MV00 versus SCConv-Early + VGRA-V1 MV01** on the frozen patient-disjoint GRAZPEDWRI-DX Split-B.
- This does NOT discard historical C3k2 + DySample + ResEMA V1/V2/V3/Split A/Split B work, nor reauthorize the paused anatomy-conditioned direction.
- Frozen C1 base commit: `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
- DDP development branch: `research/vgra-ddp-b16`. Last **scientifically tested development checkpoint before packaging docs**: `1b1696d3d77cf327452bcd40cb236c05c91098a2` (both GitHub CI workflows PASS).
- DDP candidate is **CPU/source verified ONLY**. It is NOT executable/authorized Kaggle GPU training. Original C1 remains an independent preserved fallback.
- Official MV00 and MV01: **0/100 epochs each**. Prior 1–2 epoch C1 training was **nonofficial engineering acceptance**.
- Frozen B-TEST: **SEALED**, never read for training/debugging/model selection; use B-TRAIN and operational B-VAL only.
- No official run, new governance release, new dataset change, attention module, extra ablation, or B-TEST use without the existing governance rules and explicit authorization.

## This is a research migration, not a checkpoint archive
Original Kaggle/Windows binary `.zipblob`, 100-epoch checkpoint, original patient radiographs and inaccessible private datasets are NOT silently included or recreated. Cumulative historic *conversation* portable ZIP source (30,515,326 bytes) remains independently available in this ChatGPT Project and is not imported into this GitHub-based repo snapshot. It must be retained if a complete byte-for-byte chain of earlier chats is required. All known source omissions must be identified, not guessed.

## Continue here
Next approved direction: review the private DDP native startup/resume boundary against full model/loss behavior, then implement authenticated torchrun and CUDA/NCCL nonofficial acceptance, rank0 real B-VAL and genuine cross-session G4 replay. **Do not launch 100-epoch jobs now**. Do not create extra branches. Notify the user when the next significant migration is necessary.
