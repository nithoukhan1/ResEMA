# VGRA DDP-B16 — single-branch implementation (candidate)

**Status:** Pair sampler, rank-local loader and exact-C1 native-trainer source adapter are implemented and CPU-reviewed in GitHub Actions. **The candidate is not GPU-runnable, validated or authorized yet.**
**Parent / preserved C1 release:** `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
**Development branch:** `research/vgra-ddp-b16`.
**No edits to frozen C1 source; no official training, B-VAL inference or B-TEST access.**

## Fixed experiment
- Two separate Kaggle notebooks/accounts only where permitted by Kaggle terms.
- Each notebook trains **one model** on **T4 x 2**, DDP world size 2.
- **Global batch 16 = 8 images/rank**; epochs 100; seed 42; image 1024.
- MV00 = paired-pipeline SCConv-Early control; MV01 = identical Early + VGRA V1.
- Same authenticated Early initial checkpoint, Split-B TRAIN/VAL, SGD/schedule and VGRA design.
- This DDP-B16 is a **new condition**, not a resume of C1's batch-8 training.

## Implemented now: distributed_pair_sampler.py
- Pure-Python, import-free batch sampler for existing `VGRAYOLODataset.vgra_units`.
- Shuffles complete pair units and singles deterministically by `seed + epoch`.
- Shards units across 2 ranks *before* flattening image indices.
- Does not duplicate, drop or fabricate dataset images.
- Distributes uneven single-view batches across steps; avoids a 2/1-image last step.
- 6,496 pairs -> 812 global paired steps; 1,235 singles -> 78 global single steps.
- **890 synchronized training steps/epoch**, with local batches 7 or 8 images.
- All 14,227 training images are processed once per epoch, and each pair remains on a single rank/step.
- Exposes `set_epoch(epoch)`, `__iter__`, `__len__`, and aggregate `audit(epoch)`.

**Important integration detail:** the old infinite loader stores the actual batch sampler under
`loader.vgra_pair_batch_sampler`. The frozen native training loop instead calls
`loader.sampler.set_epoch(epoch)`. A new DDP native adapter must bind the *real*
distributed pair sampler, verify loader iterator reset and keep source hashes
separately governed. Do not pretend the sampler is already connected to training.

## Added: ddp_batch_loader.py
- Builds a rank-local **TRAIN-only** `InfiniteDataLoader` using the actual pair batch sampler.
- Preserves the frozen workers=0 and deterministic generator policy.
- Validates `loader.batch_sampler.sampler` identity and exposes explicit `begin_distributed_pair_epoch(loader, epoch)`.
- The native trainer must call this hook on both ranks and replace the stock `loader.sampler.set_epoch(epoch)`; the hook resets the cached iterator.
- Pure CPU fake-runtime tests verify rank wiring, epoch reset and recipe refusal. This does not substitute for actual torch/GPU acceptance.

## Added: native_source_adapter.py
- Produces **in-memory** candidate revisions of the pinned C1 `vgra_trainer.py`, `early_paired_trainer.py` and `engine/trainer.py`; does not overwrite or create executable C1 files.
- Verifies each original file by its exact Git blob SHA and rejects missing, tampered or extra source entries.
- Keeps existing pretrained authentication and TRAIN-only visibility binding logic, changes only 2-rank/batch-16 startup and data loader selection, resets the actual pair sampler at epoch start, and revises single-GPU AMP guards for later separate acceptance.
- CPU CI verifies exact-source hashing, Python syntax, presence of the intended integration hooks and rejection of modified source.
- **Not yet executable:** requires isolated private staging, correct DDP gradient/loss policy, rank-0 full validation, synchronization and rank-complete G4 recovery.

## DDP loss and AMP verification (CPU acceptance)

The frozen `v8DetectionLoss` returns three weighted loss components multiplied
by the **local** image batch size. The frozen VGRA criterion appends
`0.25 * (2 * local_pair_count) * weighted_visibility_CE_mean`.
The native trainer multiplies the summed scalar loss by DDP world size 2.
DDP's mean-gradient reduction therefore returns the **sum of the two local
objective gradients**. No artificial visibility target is used for a
zero-pair local batch. The uneven TRAIN tail has local batch sizes 8/7.

The mathematical/reference regression tests and a real two-process CPU/Gloo
PyTorch autograd acceptance passed. Tests include a rank where VGRA-only
parameters are unused (`find_unused_parameters=True`) and unequal local
image counts. Source-only AMP checks confirm the new candidate retains
`init_scale=1`, `growth_interval=2000`, gradient-finiteness checks and
the original fatal-on-optimizer-skip policy.

**Critical distinction:** Native YOLO's classification/box loss divides by
each rank's local target-score sum. Thus rank-summed DDP objective is not
bitwise or mathematically the same as a single-process unsharded global-16
YOLO loss. This is an explicit DDP-B16 training condition applied equally
to MV00/MV01, not a guarantee of original C1 numerical equivalence.

**Still blocked:** actual frozen YOLO/VGRA DDP backward with real datasets,
CUDA AMP/scaler synchronization and finite-guard behavior, full B-VAL rank-0
validation with safe barriers, and authenticated multi-rank G4 resume. CPU
Gloo smoke tests are not GPU training authority.

## Native DDP startup blocker resolved at the refusal boundary

The C1 Ultralytics `BaseTrainer.train()` calls
`generate_ddp_command()` when invoked from a non-torchrun process. Its
generated child reconstructs the trainer with only `cfg` and `overrides`.
This **loses mandatory VGRA constructor inputs** (frozen assignments,
visibility evidence, authenticated Early checkpoint, recovery setup).

The candidate now injects `require_authenticated_ddp_child(self)` into
`VGRADetectionTrainer.train()`; `EarlyPairedTrainer` inherits the guard.
It refuses stock Ultralytics auto-DDP invocation, wrong ranks/global batch,
missing independently authenticated worker readiness or missing governed
recovery setup. GitHub CI verifies refusal paths. **No DDP worker currently
sets the verified-worker readiness marker, so actual training stays blocked.**
The future worker must explicitly run under torchrun with two ranks and bind
all C1-class scientific inputs and DDP-compatible recovery before use.
A manually set internal boolean is NOT proof of release authorization.

## Added: explicit DDP-B16 worker PRE-FLIGHT (no training)

- `ddp_worker_stage.py` verifies all 336 original C1 source files,
  seven existing integration modules, and the SHA-pinned historical
  source manifest/recipe. It creates a **new disposable private source stage**,
  changes only the three exact-source-gated native DDP candidates there, and
  writes a new full-file SHA256 inventory. Original C1 files are never edited.
- `ddp_worker.py` has strict **MV00/MV01, FRESH-only, two-rank batch-16**
  review-plan parsing, a pinned Git commit requirement, read-only historic R0
  authentic asset verification, and private staging. It has **no CUDA or
  training implementation** and explicitly rejects `--execute`.
- CPU tests check full private staging, original-source preservation,
  fixed MV00/MV01 identities, sealed-test paths and unauthorized launch refusal.
- Historical R0 file authentication proves *input identity only*. Its original
  batch-8 source/recipe and G4 recovery identity **cannot** authorize a
  batch-16 DDP run.

The explicit per-rank `torchrun` training child is **still to be
implemented**, including correct validated DDP recovery, before any GPU
execution request is accepted.

## Minimum remaining implementation
1. Connect the verified worker preflight/private stage to a **new explicit two-rank torchrun TRAINING child**, after implementing DDP-specific G4 recovery. Stock Ultralytics auto-DDP is forbidden; bind all mandatory authenticated inputs separately per rank. Keep `frozen_source/` and C1 unchanged.
2. Prove the complete **native YOLO/VGRA** DDP forward/loss and optimizer
   accumulation using the private source stage; then verify CUDA AMP, finite
   guards, and exact 537-tensor pretrained transfer on both ranks. Synthetic
   DDP loss/gradient math alone has passed.
3. Run whole B-VAL validation **once on rank 0** after synchronized training
   steps, while rank 1 waits safely; all 3,049 operational images counted once.
4. Adapt G4 recovery for per-rank RNG/sampler state plus model, optimizer,
   EMA, scaler, gradients and scheduler; off-session verified transport required.
5. Add two slim notebooks calling the same verified DDP engine. Run short
   *nonofficial* GPU acceptance and real independent-session epoch-2 resume
   before freezing/authorizing the 100-epoch research experiments.

**Known unsolved engineering boundary:** batches containing no AP/LAT pair
may leave VGRA-only parameters unused. The future DDP trainer must handle
this correctly (e.g. verified dynamic-unused-parameter behavior) without
fabricating visibility labels. The visibility loss's rank-local count
normalization and the native batch-scaled detection loss require a
numerical gradient-equivalence test; no scaling policy is declared accepted yet.

## Tests and running them

```bash
python -m unittest discover -s research/r9_training/ddp_b16/tests -v
```

GitHub Actions executes the sampler, loader-factory, epoch-reset and exact-C1 native-source patch tests. PASS here establishes CPU/source checks only, **not** two-GPU numerical or recovery acceptance.

## Repository discipline

Only this one DDP development branch. No new attention, parameter sweeps,
alternative datasets, B-TEST access or unrequested experiments. C1 is the
immutable fallback. Large checkpoints remain outside Git with SHA256 receipts.

Next action: build DDP-specific G4 recovery and the guarded torchrun training child, then verify actual rank-aware YOLO/VGRA, CUDA AMP and synchronized B-VAL. Do **not** launch official training yet.
