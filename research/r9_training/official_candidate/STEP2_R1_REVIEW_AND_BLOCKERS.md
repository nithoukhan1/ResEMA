# Detection Project — Chat25 Step 2 R1 — OFFICIAL launcher engineering (PRE-AUTHORIZATION)

## Evidence boundary
This is an additive, **CPU-only** review contract and original-byte preflight, NOT an executable production trainer, Kaggle notebook or approved 100-epoch release. No GitHub commit, remote push, GPU training or Kaggle Save Version has been executed by this package.

The user has preserved both **nonofficial** epoch2 original ZIPs and the supporting-evidence ZIP on Windows with verified matching SHA256. The optional private Kaggle .zipblob mirror is not required for the next official-fresh experiment. Any missing raw logs are optional historical support.

## Current fixed science (do not change)
- MV00 = SCConv-Early YOLO11s paired control. MV01 = matched control + VGRA V1 on classification logits only.
- Frozen patient-disjoint GRAZPEDWRI-DX Split B, B-TRAIN/B-VAL, B-TEST sealed.
- User has NOT authorized official training. Official experiments must initialize independently from the original approved SCConv-Early pretrained source, not R9 rehearsal capsules.
- Baseline Git commit `b783eb3a470ab010785a8a24a90a666e9a3f2ed6` (reference to previously verified Chat24 commit, NOT independently queried live by this package).
- Pinned frozen manifest 336 source files + 7 modules and candidate recipe SHA are embedded in `R9_OFFICIAL_RELEASE_PREFLIGHT_R1.py`. It can reverify exact bytes when run in the R9 worktree; it does not claim to verify the current branch HEAD.

## Why the R9 R4 bounded worker cannot be used unmodified for official 100 epochs
1. `R9_R4_BOUND_GPU_WORKER.py` requires `--accept-nonofficial-bounded-gpu`, rejects genuine terminal output, limits FRESH to target 1, RESUME to one subsequent epoch, and explicitly refuses the 100th epoch.
2. The `R9B_R3_NATIVE_WORKER_INTEGRATION.verify_plan()` forbids `target_completed_epoch == 100` and recognizes only the historical nonofficial resume receipt schema.
3. The `R9BPostPublicationSessionStopMixin` accepts only nonterminal completed epochs 1–99 (1-based target) and throws its exception after authenticated G4 publish. It must not be employed to certify terminal completion.
4. The original `G4IntegratedEpochMixin` has a distinct terminal path through `final_eval()` and `d4e_g3b_guard_terminal()`, but this pathway lacks the Chat24 real-GPU terminal acceptance. It publishes an intentionally terminal, **nonresumable** best/last artifact and must be tested before release.
5. `G4` original metadata includes `official_training_authorized:false` as historical publication schema; R9 transport expects exactly that false value. An official release needs a separate immutable user authorization + release receipt, not a silent rewrite of historical G4/schema/recipe values.
6. In `ultralytics/engine/trainer.py`, stop/early-stop/final-epoch handling and native `self.epochs` determine terminal behavior. A quota-aware session stop must occur AFTER full guarded checkpoint and G4 publication and BEFORE ordinary final evaluation, at safe nonterminal boundary. An arbitrary timer interrupt risks unpreserved in-progress state.
7. Session scheduling requires conservative epoch-duration monitoring and a safety margin before the Kaggle runtime limit. The Kaggle Save Version output-retention and private dataset re-stage must be tested in actual saved versions, not inferred from an interactive session.

## Step 2 completion criteria (NOT yet achieved)
A. New production native worker that takes a **new** reviewable R9 official plan, reconstructs pinned source/data/R0 pretraining identity, and supports fresh+verified recovery with immutable authorized run_id (never R9 rehearsal run IDs).
B. Dual independent T4 process launcher, fail-closed arm/gpu mapping and process exit validation.
C. Post-authenticated-publish quota-aware nonterminal safe stop, original full-state G4 capsule, separate offsite persistence receipt and next-session exact byte re-stage.
D. Terminal epoch100 evaluation: exact 100th epoch completed, B-VAL final-eval, authenticated best/last G4 terminal, offsite and re-download integrity verified, terminal cannot resume.
E. Tests for each guard, resume cursor, failure propagation, time budget, external persistence and terminal publish; relevant bounded GPU acceptance and Kaggle Save & Run All validation.
F. Versioned GitHub new release commit + notebook pinned to **that** reviewed commit; explicit user authorization AFTER review.

## Safe use of this R1 package (Windows PowerShell)
1. Verify that the original worktree is clean and at the previously accepted SHA, without attempting a new clone or modifying frozen code:
   `git -C 'E:\PhD\Admitted\Research\Project 1\ResEMA-Github Repo\ResEMA-R9-TRAINING' status --short`
   `git -C 'E:\PhD\Admitted\Research\Project 1\ResEMA-Github Repo\ResEMA-R9-TRAINING' rev-parse HEAD`
2. Extract this ZIP to a *separate safe inspection directory*. Copy only the `research/r9_training/official_candidate` directory into the existing R9 worktree **after confirming it does not already exist**.
3. `python research/r9_training/official_candidate/TEST_R9_OFFICIAL_RELEASE_CONTRACT_R1.py`
4. `python research/r9_training/official_candidate/R9_OFFICIAL_RELEASE_PREFLIGHT_R1.py --contract research/r9_training/official_candidate/R9_OFFICIAL_RELEASE_CONTRACT_R1.json --repo-root .`
5. Review new-file-only Git diff/status. Do **not** launch a GPU experiment, reinterpret this candidate as authorization, or push the new files yet without a reviewed production implementation.

## Project position
Step 1: external epoch2 preservation COMPLETE. Step 2: static release-contract subphase R1 developed; native official worker/launcher and safe terminal behavior PENDING. Step 3 Kaggle production notebook PENDING. Step 4 production acceptance PENDING. Step 5 release authorization PENDING. Official B-TEST SEALED.

## Additional Step2 R1 CPU policy candidate
`R9_SESSION_BOUNDARY_POLICY_R1.py` calculates a conservative post-G4 decision after a verified nonterminal epoch boundary. It explicitly rejects unverified G4, premature native early-stop, nonterminal epoch100, early terminal publication and invalid time inputs. It does **not** make GPU/clock calls or integrate into the native worker yet. The next implementation must wrap it into an authenticated training hook, verify original G4 bytes, and export+persist checkpoints before claiming cross-session safety. The policy's conservative next-epoch estimate is a configurable engineering parameter and cannot guarantee Kaggle quota; server Save Version retention requires separate validation.

Local static/CPU evidence at generation: 10 release-contract tests PASS, 12 session-boundary policy tests PASS, exact 336+7 frozen-source pin audit PASS against uploaded R9 source integration ZIP expanded into an offline tree. This is *not* verification of the user's currently checked-out Git HEAD, and not acceptance of terminal native epoch100.
