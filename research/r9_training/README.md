# R9 VGRA research — version-controlled source integration (20 Oct 2026)

## Provenance and authority

This tree is derived byte-for-byte from the user-accepted `DETECTION_PROJECT_R9_R4_DIRECT_NATIVE_ACCEPTANCE_CANDIDATE_2026-10-20.zip`. It is intended to be added to a **new independent branch** based at `48cc059cfa133ae5fbfcec607f15daeda843d3ba`.

- `frozen_source/` — **336 exact R9A scientific source files**. This is an ISOLATED Ultralytics package root, NOT an overlay to copy indiscriminately onto the repository top-level `ultralytics/` folder.
- `integration_modules/` — **7 exact original R9A integration module files**.
- `code/` — unchanged R9 R4 bounded-training and verification scripts (original candidate), source manifest and tests.
- `R9_RELEASE_RECIPE_CANDIDATE_R1.json` — byte-identity-pinned recipe.
- `provenance/` — original R9B R3 embedded ancestor archive for reproducibility (binary evidence, NOT a model checkpoint).

This bundle preserves the EXACT source hashes, without rewriting VGRA, patching source or authorizing official training. It has **no** Kaggle checkpoint (`*.pt`), GRAZPEDWRI-DX image, or official 100-epoch runner.

## Safety / limits

This R9 R4 executable is the *nonofficial one-epoch GPU acceptance* code. The earlier MV00 and MV01 epoch-1 runs and the Windows→Kaggle transport were successful. This version-controlled source by itself does NOT demonstrate genuine epoch-2 GPU resume nor implement a production-ready 100-epoch trainer. We must implement, review, and test those before official training. B-TEST remains sealed.

The GitHub/Kaggle workflow will clone a fixed commit of this branch, verify all frozen source paths, use Kaggle Save & Run All for training, and persist complete checkpoints across subsequent sessions. Source code + text reports go in Git; multi-hundred-MiB checkpoint ZIPs belong in external versioned artifacts / local storage, NOT regular Git commits.

## Frozen gates

`code/R9B_R3_NATIVE_WORKER_INTEGRATION.py` exposes `verify_science_source(science, modules, manifest)`. Running `VERIFY_R9_SOURCE.py` from this directory checks all 343 pinned scientific-source and integration module files and recipe hash.

**Do not run `R9_R4_DUAL_LAUNCH.py` as an official 100-epoch training launcher**: it is explicitly bounded to one nonofficial epoch from fresh.
