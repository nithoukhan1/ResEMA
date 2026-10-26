# VGRA DDP-B16 — single-branch implementation (candidate)

**Status:** pair-sharding source and CPU tests added; *not executable DDP training*.
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

## Minimum remaining implementation
1. Add one **private staged DDP source adapter** and loader wiring for MV00/MV01.
   Do not edit the frozen `frozen_source/` or C1 launcher. Ensure every rank
   uses the same deterministic global step plan and the native epoch cursor.
2. Correctly test DDP forward, 4-component VGRA loss, zero-pair batches,
   last-step rank batch differences, gradient averaging, optimizer accumulation,
   AMP / finite guards and original 537 pretrained tensor transfers.
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

The sampler tests cover frozen-population cardinality, no cross-rank pair splits,
unique full data coverage, balanced ranks, exact step count, epoch determinism,
non-shuffled validation behavior and invalid configuration rejection.

These Python tests must be **executed and verified** before trainer integration
is promoted. Adding tests to Git alone is not a passing test result.

## Repository discipline

Only this one DDP development branch. No new attention, parameter sweeps,
alternative datasets, B-TEST access or unrequested experiments. C1 is the
immutable fallback. Large checkpoints remain outside Git with SHA256 receipts.

Next action: verify the CPU sampler tests, then add the private staged DDP
dataloader/native integration. Do **not** launch official training yet.
