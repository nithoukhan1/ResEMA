# D4-C4 VGRA Pair-Aware Dataset / Batching

## Status
`COMPLETE / CPU-VERIFIED / FROZEN-MANIFEST-AUDITED`

Branch: `research/vgra-impl-01`
Parent: `110a231621bc66046b338a6c948fccab25f2ed85`

## R1 safe-stop and R2 correction
R1 (`D4_C4_IMPLEMENT_VGRA_BATCHING_R1.py`) passed source compilation and all 38
focused CPU tests. The separate full-manifest audit then failed to import
`ultralytics.data.vgra` because its subprocess executed a file from
`research/tools/`, making that directory the leading Python module search path.

The R1 transaction stopped before any commit. Its rollback reported:
`PRECOMMIT_ROLLBACK_CLEAN=TRUE`.

R2 keeps the VGRA dataset/sampler/collate code, focused tests, and audit
implementation identical to R1. It checks that `ultralytics.data.vgra` resolves
to this worktree and launches the audit using
`python -m research.tools.d4_vgra_batching_audit` from the worktree root.
This records the resolved environment issue without changing VGRA semantics.

## Implementation
Added:
`ultralytics/data/vgra.py`

Stock `ultralytics/data/dataset.py` and `ultralytics/data/build.py` remain unchanged.

## Pair-safe augmentation
For VGRA pair-aware training, the following cross-study composition transforms are
forced to zero in a copied hyperparameter object:
- Mosaic;
- MixUp;
- CutMix;
- Copy-Paste.

Ordinary per-image transforms remain available and are applied independently to each
radiograph.

The same pair-safe augmentation policy is required for D4-MV-00 and D4-MV-01.

## Runtime assignment binding
Every runtime image is matched by filestem to the frozen D4-C2 image-assignment manifest.

Valid TRAIN pairs:
`structural_role=PAIRED` + `operational_role=PAIR_CANDIDATE`

Valid VAL pairs:
`operational_role=PAIRED`

The readable companion of the known unreadable VAL image becomes an ordinary
single-view fallback.

## Batch contract
`VGRAPairBatchSampler` treats each valid AP/LAT pair as an indivisible two-image unit.

The collated batch emits:
- `vgra_pair_index`: companion batch position or -1;
- `vgra_pair_valid`: valid paired-member mask;
- `vgra_view_code`: AP/LAT/other code;
- frozen pair IDs/filestems/operational roles for provenance.

If a valid pair is accidentally split across batches, collation fails closed.

## Rectangular validation decision
VGRA V1 uses `rect=False` in pair-aware train and validation data construction.

Reason:
stock rectangular validation assigns shapes according to conventional aspect-ratio
batches. Pair-preserving batches are a different grouping and could otherwise contain
members formatted for incompatible rectangle shapes.

This preprocessing difference is controlled by D4-MV-00:
the pair-pipeline Early control and full VGRA use the exact same square pair-safe pipeline.

## Distributed training
VGRA V1 pair dataloader is intentionally single-process only.
DDP fails closed.

If distributed training becomes necessary, pair-preserving distributed sharding requires
a separate governed design/verification transaction.

## Full metadata-only audit at batch size 16
TRAIN:
- operational images: 14227
- exact pair units: 6496
- single units: 1235
- batches: 890
- pair split events: 0

VAL:
- operational images: 3049
- usable pair units: 1401
- single/fallback units: 247
- batches: 191
- pair split events: 0

No image or YOLO label file was opened.

## Deferred
- model runtime dispatch from `vgra_pair_index`;
- visibility target generation from B-TRAIN labels;
- visibility loss;
- trainer;
- validator;
- real-data execution;
- GPU training.

## Next
`D4-C5_VGRA_CRITERION_TRAINER_VALIDATOR_INTEGRATION`
