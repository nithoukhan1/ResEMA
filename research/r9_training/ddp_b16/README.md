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
- Produces **in-memory** candidate revisions of five pinned C1 files: the two trainers, the engine trainer, and both paired validators. Does not overwrite or create executable C1 files.
- Verifies each original file by its exact Git blob SHA and rejects missing, tampered or extra source entries.
- Keeps existing pretrained authentication and TRAIN-only visibility binding logic, changes only 2-rank/batch-16 startup and data loader selection, resets the actual pair sampler at epoch start, and revises single-GPU AMP guards for later separate acceptance.
- CPU CI verifies exact-source hashing, Python syntax, presence of the intended integration hooks and rejection of modified source.
- **Not yet executable:** native G4 callbacks, full-source CUDA tests and authenticated distributed recovery integration are still incomplete.

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
CUDA AMP/scaler synchronization and finite-guard behavior, actual real-image
B-VAL evaluation, and authenticated multi-rank G4 resume. CPU/Gloo tests are
not GPU training authority.

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
  changes only the five exact-source-gated native DDP candidates there, and
  writes a new full-file SHA256 inventory. Original C1 files are never edited.
- `ddp_worker.py` has strict **MV00/MV01, FRESH-only, two-rank batch-16**
  review-plan parsing, a pinned Git commit requirement, read-only historic R0
  authentic asset verification, and private staging. It has **no CUDA or
  training implementation** and explicitly rejects `--execute`.
- CPU tests check full private staging of the five candidate patches,
  original-source preservation, fixed MV00/MV01 identities, sealed-test paths
  and unauthorized launch refusal.
- Historical R0 file authentication proves *input identity only*. Its original
  batch-8 source/recipe and G4 recovery identity **cannot** authorize a
  batch-16 DDP run.

The explicit per-rank `torchrun` training child is **still to be
implemented**, including correct validated DDP recovery, before any GPU
execution request is accepted.

## Added: DDP-B16 two-rank G4 capsule candidate (CPU proven)

`ddp_g4_recovery.py` adds a **separate** epoch-boundary recovery format,
without modifying the historical batch-8 D4E/G4 classes or C1 receipts.

- **Each rank** independently captures model/BN buffers, parameter gradients,
  optimizer, EMA, scheduler, AMP GradScaler, CPU + local CUDA RNG,
  Python/NumPy RNG, loader generator, Albumentations RNG snapshot,
  frozen pair-sampler seed/epoch/population, stopper and optimizer cursor.
- An explicit distributed collective requires matching run/release/recipe/data
  identity and epoch on both ranks, then barriers around both rank saves and
  the common SHA256 receipt manifest.
- Saving refuses overwrites. Reading a rank checks the **externally supplied**
  manifest SHA256, both rank receipts/checkpoints, and the requested rank.
  `torch.load(..., weights_only=True)` is used only after byte verification.
- The rank-specific sampler contract is exactly seed 42, next epoch cursor,
  6,496 pairs, 1,235 singles, 14,227 images, 890 steps/rank.
- PyTorch CPU/Gloo two-process CI confirms snapshot/restore, model/EMA,
  optimizer, scheduler, scaler, loader state, rank checks, missing external
  manifest pins and post-publication tamper rejection.

**Evidence boundary:** the test digest is only a *simulated* external pin.
Actual off-session transport, independent SHA receipt, CUDA AMP,
interrupted-vs-uninterrupted training equivalence, native callback integration,
actual full B-VAL inference, native G4 callback publication and terminal-epoch logic are **not proven**.
The normal C1 epoch saver must NOT call or consume this DDP capsule until those
integration gates pass. No official execution has been enabled.

## DDP rank-0 validation and native epoch source hook (CPU verified)

`ddp_epoch_boundary.py` performs one rank-0-only governed pair-aware B-VAL
evaluation, requires the validator to count all 3,049 operational images,
and broadcasts finite, identical metrics and fitness to rank 1. A rank-0
Python error is broadcast to both ranks instead of leaving rank 1 waiting.
Only the **privately patched** validators accept two-rank training with the
temporary rank-0 validation guard; rank 1 remains forbidden to invoke them.

The exact-source private engine candidate replaces the native
`self.validate()` call with `rank0_validate_and_sync(self, epoch=epoch)`.
This deliberately avoids the stock trainer's extra EMA-broadcast collective
inside a rank-0-only validation call. Both rank-local callbacks are not yet
rewired to the new G4 capsule; the historical C1 single-worker hooks remain
a **hard execution blocker**.

GitHub CPU/Gloo two-process tests prove that only rank 0 invokes the fake
validator, both ranks agree on metrics and cardinality, and an incorrect
image count fails both ranks. This **does not** prove real YOLO/VGRA
validation, CUDA/NCCL behavior or Kaggle checkpoint continuity.

## Added: synchronized native G4 epoch-end callback (CPU/Gloo verified)

\`ddp_native_g4_hook.py\` now provides the **nonterminal epoch** callback
for the private two-rank candidate. The exact-source engine adapter inserts
\`ddp_b16_g4_epoch_end(self, epoch=epoch, nb=nb, last_opt_step=last_opt_step)\`
*after* rank-0 native best/last save and on-fit-epoch callbacks, in place
of invoking original C1 single-worker G4 callbacks for \`world_size=2\`.

- Rank 0 checks existence and hashes of native selected \`best.pt\` and
  \`last.pt\` and broadcasts the authoritative selected-best SHA and stopper
  state. Rank 1 must not independently select a checkpoint.
- Both ranks collectively verify optimizer-update count, last optimizer-step
  index, accumulation, 890-step pair sampler and identity before saving.
- Both ranks publish their own complete G4 capsule and a synchronized
  combined manifest through the already CPU-verified G4 library.
- Missing selected best or divergent rank-step counters fail **both**
  processes without publishing a resumable capsule.
- Early-stop/final epochs produce **no** next-epoch recovery capsule; final
  evaluation and terminal offsite publication remain pending.
- The candidate callback requires explicitly bound DDP G4 identity/root
  and refuses legacy \`_d4e_recovery_configured\` batch-8 mode.

Real two-rank CPU/Gloo test cases pass for nonterminal publication, missing
best, step divergence and terminal no-capsule. These are synthetic
checkpoints and a fixture augmentation RNG provider, not scientific GPU
training acceptance.

**Hard blockers remain:** existing C1 startup checks still depend on
legacy D4E/G3B4 and intentionally prevent executing this candidate; the
authenticated torchrun child, full native loss/AMP/BN behavior, CUDA/NCCL
G4/validation, independently pinned off-session transfer, interrupted-vs-
uninterrupted replay and terminal export have **not** been authorized
or proven.

## Minimum remaining implementation
1. Integrate the CPU-verified G4 epoch-end candidate with a properly authenticated **torchrun training child and separate two-rank startup/resume boundary**. Existing C1 D4E/G3B4 startup guards must remain fail-closed until an independently tested replacement exists. Stock auto-DDP forbidden; C1 unchanged.
2. Prove the complete **native YOLO/VGRA** DDP forward/loss and optimizer
   accumulation using the private source stage; then verify CUDA AMP, finite
   guards, and exact 537-tensor pretrained transfer on both ranks. Synthetic
   DDP loss/gradient math alone has passed.
3. Run real whole B-VAL validation **once on rank 0** after synchronized training
   steps, while rank 1 waits safely; verify all 3,049 operational images.
4. Finish two-rank startup and resume integration, terminal evaluation/export and external SHA-pinned off-session transport; prove genuine interrupted-vs-resumed GPU replay. Rank-0 VAL and nonterminal G4 callback synchronization now pass CPU tests only.
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

Next action: freeze and archive the present CPU-verified DDP candidate before long Kaggle logs, then implement the authenticated native torchrun startup/resume path, real CUDA/NCCL validation and off-session recovery acceptance. Do **not** launch official training yet.
