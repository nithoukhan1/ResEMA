# Chat27 DDP-B16 — Complete Engineering Ledger

## 1. Motivation, design and immutability
Original C1 used two independent workers, GPU0 MV00 and GPU1 MV01, each global batch8. User requested **separate Kaggle notebook per arm, T4 x2 per notebook, global batch16/8 per rank**, while retaining patient-effective Split-B, complete AP/LAT pairs, shared Early pretrained transfer, exact VGRA architecture, AMP full-state recovery and scientific comparison. The change is execution infrastructure only, not a new attention module or scientific feature. DDP branch `research/vgra-ddp-b16` off frozen `fa6a7675...`. Preserve C1 source as read-only (verified GitHub compare).

Do **not** equate training one arm faster to total project throughput without verifying availability of four GPUs for concurrent notebooks. No quota/terms circumvention.

## 2. File-by-file implementation with verification level

| File | Purpose | Confirmed scope |
|---|---|---|
| `distributed_pair_sampler.py` | Deterministic two-rank balanced complete-pair sharding | Unit CPU PASS; not real image-loader benchmark |
| `ddp_batch_loader.py` | Rank-local InfiniteDataLoader, workers0, true batch_sampler epoch/reset | CPU fake-runtime PASS |
| `native_source_adapter.py` | Exact C1 blob-checked **five** private candidate edits; native engine, two trainers, two validators | Source, syntax, tamper tests PASS |
| `ddp_epoch_boundary.py` | Rank0-only full paired B-VAL and 3,049-image cardinality check; synced fitness/metrics/errors | Two-rank CPU/Gloo dummy validator PASS |
| `ddp_launch_guard.py` | Refuse Ultralytics unsafe auto-DDP launcher; requires authenticated torchrun child | Source CPU tests PASS |
| `ddp_worker_stage.py` | Authenticate/copy 336 frozen scientific source and 7 unchanged integration modules, then apply five private patches | CI source identity + private-stage pass |
| `ddp_worker.py` | Strict MV00/MV01 input plan, original asset R0 preflight and FRESH-only private staging | CPU guards PASS; `--execute` explicitly LOCKED |
| `ddp_g4_recovery.py` | Complete two-rank epoch-boundary rank-local model, optimizer, EMA, scheduler, scaler, grads, RNG, sampler, SHA256 manifest | Two-rank CPU Gloo save/restore and tamper refusal PASS; offsite pin simulated |
| `ddp_native_g4_hook.py` | Selected-best/native-last SHA, rank step agreement, common nonterminal G4 capture | Two-process Gloo normal, missing-best, step-drift, terminal tests PASS |
| `tests/*.py` | lightweight stdlib source and arithmetic tests | Research Integrity CI PASS |
| `torch_acceptance/*.py` | Actual CPU two-rank Gloo autograd, capsule, rank0 validator and epoch callback tests | CPU Numerics CI PASS |
| `.github/workflows/ddp-b16-cpu-numerics.yml` | Isolated PyTorch/NumPy CPU CI; prevents torch import in plain stdlib CI | PASS |
| `.github/workflows/research-integrity.yml` | Original integrity checks + DDP CPU test discovery | PASS |

Frozen scientific body remains unchanged, including `vgra.py`, `vgra_head.py`, `vgra_runtime.py`, `vgra_targets.py`, model YAMLs, source trainers and original C1 launchers.

## 3. Pairing and exact cardinality
Train units 6,496 pairs + 1,235 singles. Steps 812 complete-pair global batches (up to 8 pairs), 78 singles global batches, **890 synchronized steps/epoch**; per-rank 7–8 images (no 2/1 tail). Every image once, no pair split across ranks, deterministic epoch shuffling `seed+epoch`. The sampler is a rank-aware TRAIN-only loader; rank0 validation uses a distinct complete B-VAL loader.

## 4. Gradient semantics and real CPU verification
The existing YOLO detection criterion scales weighted loss by **LOCAL image count**; native BaseTrainer sums loss vector then multiplies by **world size=2**; PyTorch DDP averages gradients. Therefore optimizer effectively sees sum of local objectives. VGRA visibility term is `0.25 * (2*local_pairs) * weighted_cross_entropy_mean`; zero-pair rank emits zero valid visibility contribution with no fabricated targets; `find_unused_parameters=True` necessary in zero-pair condition.

**Do not assert mathematical identity to single unsharded batch16**: native YOLO classification/box target-score normalization occurs separately per rank. Both arms under identical DDP setup are matched, but historical C1 batch8 scores not directly the same run condition. Toy two-rank CPU Gloo backward with unequal local samples and unused VGRA-only parameters PASS. Native YOLO model/backward on actual data **not tested**.

AMP source preserves initial scale1/growth_interval2000 and original finite/optimizer-skip guard. CUDA AMP under DDP **not verified**, including GradScaler rank parity and no silent optimizer-step skips.

## 5. Native launch: what failed in review and how safeguarded
Stock Ultralytics auto-DDP generates child trainer from `cfg` and `overrides`, discarding required VGRA constructor inputs: assignment manifest, train-derived visibility counts/weights, authenticated Early checkpoint and recovery identity. `ddp_launch_guard.py` now refuses that path. `ddp_worker.py` performs read-only authenticated PRE-FLIGHT and private staging only, **does not execute training**. A verified explicit child with all inputs bound on each rank is not yet implemented; setting internal readiness booleans alone does not prove actual release authorization.

Old C1 G4 and G3B/R7C trainer startup checks are source and batch8 specific and **cannot** be reused by merely switching `world_size`; current private adapter does not fully replace these startup lifecycle hooks, so code is NOT runnable.

## 6. B-VAL governance and synchronization
Current B-VAL operational 3,049. Old pair-aware validators both explicitly forbid world_size2; exact-source private DDP copies replace that with *rank0 only and temporary reviewed guard*. Native private loop substitutes `rank0_validate_and_sync` rather than stock `self.validate()`, avoiding rank0-only invocation of a function that contains DDP EMA buffer broadcast and could hang.

CPU Gloo test verifies: only rank0 fake-validator call, rank1 no validation, metric/fitness agreement, strict 3049 seen count, both-rank failure on wrong cardinality. Real image-level B-VAL with native VGRA paired forward and 3049 detections pending.

## 7. G4 rank states and native epoch hook
Two-rank capsule carries each rank's model buffers/weights, grads, optimizer, EMA, scheduler, AMP scaler, loader generator, Python/NumPy/CPU and local CUDA RNG, Albumentations internal RNG, sampler cursor/population, stopper/selected-best hash and optimizer counter. A rank0-published combined SHA256 manifest must match both immutable rank files. Resume API requires **external/off-session** manifest pin and verifies both rank files before weights_only loading. No claim of successful real offsite transport yet.

After rank0 native `save_model()` and `on_fit_epoch_end` callback, private source now calls `ddp_b16_g4_epoch_end` instead of C1 single-worker G4. Rank0 hashes selected-best and last; both ranks must agree optimizer-update/last-step/accumulate; two complete rank files and shared manifest published. Terminal early-stop/final epoch refuses next-epoch capsule; terminal final-eval/export remains an unproven separate gate.

CPU Gloo scenarios accepted: normal nonterminal publication, missing rank0 `best.pt` rejection on both ranks, step divergence rejection on both ranks, terminal no-capsule. **Synthetic** local files/augmentation snapshots; no actual CUDA NCCL replay.

## 8. CI and failure chronology (preserve!)
- During DDP native work, initial lightweight Research Integrity failed when it attempted to import the PyTorch-only Gloo test without torch installed. Corrected by moving real Gloo test under `torch_acceptance/` with separately pinned CPU PyTorch workflow. Both workflows PASS.
- After widening private patches from three files to five (to cover both validators), an old test assumed three files; CI failed at assertion `5 != 3`. Corrected test to check five; latest CI PASS. No frozen original mutated.
- Original source audit exposed stock YOLO DDP auto-launch dropping constructor governance inputs; launch guard added rather than bypassing preflight.
- Audit discovered stock BaseTrainer.validate collective conflict; rank0-only guarded direct validation and cross-rank broadcast added rather than calling native validator from only rank0.
- Audit discovered original batch8 G4 identity incompatible with new DDP global16; separate two-rank capsule and native hook added. Genuine offsite resume still pending.

## 9. Immutable evidence identifiers
Frozen C1: `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
DDP accepted **CPU code state pre-migration**: `1b1696d3d77cf327452bcd40cb236c05c91098a2` (GitHub diff +54 commits, zero behind; original C1 frozen files untouched).

GitHub Research Integrity: https://github.com/nithoukhan1/ResEMA/actions/runs/40287227288 PASS.
GitHub two-rank PyTorch CPU acceptance: https://github.com/nithoukhan1/ResEMA/actions/runs/40287227412 PASS.
Use source commit SHAs and source test reports, not a generated handoff statement alone, as direct evidence.

## 10. Exact next gate (do not add unnecessary steps)
Integrate a strictly authenticated per-rank `torchrun` startup/resume interface in the private DDP source; replace/bridge incompatible C1 single-GPU guard hooks with equivalent verified DDP lifecycle; ensure train/val/callback/save/terminal order cannot desynchronize. Then, **and only then**, conduct shortest required nonofficial GPU acceptance: T4x2 actual paired forward/backward/AMP on MV00 and MV01, real 3049 B-VAL, one true cross-session epoch-2 recovery with independently verified receipts. Freeze one immutable DDP launch release; seek explicit authorization before official 100-epoch runs.

No new attention, no new sampling/data changes, no new branch, no B-TEST access.
