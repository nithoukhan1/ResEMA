# Detection Project — Chat28 Step 2F4: Native Import Boundary

Date: 2026-10-28. Status: **review/import candidate, not a training release**.

## Starting point

- Branch `research/vgra-ddp-b16` from CPU-verified Step2F3
  `c4aaf092f6b9aa3a08bd7d36abd6d14146347e6a`.
- Frozen science C1: `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
- Both official MV00/MV01 remain 0/100; B-TEST is sealed.

## A. Close review-to-import identity drift

`ddp_native_binding.py` now includes the independently authenticated
`checkout_root` in the exact, fully SHA256-digested native binding. This
lets the bootstrap derive expected source bytes from **frozen C1**, rather
than relying only on a user-modifiable stage receipt.

`ddp_native_bootstrap_candidate.py` now runs
`reauthenticate_private_stage_for_import()` immediately before any
staged Ultralytics import. The function:

1. Checks the exact fixed constructor binding and pinned DDP recipe.
2. Requires the private `source` and `modules` paths to be siblings
   outside the original checkout.
3. Revalidates original C1 source and deterministic five-file adapter
   output, then recomputes SHA256 of all 336 staged source files and
   seven integration modules.
4. Compares the stage manifest and receipt hashes to the original,
   independently authenticated binding.
5. Refuses any edited source, module or receipt prior to native import.

The native bootstrap test creates a real private stage from frozen C1,
verifies it, then changes BOTH `trainer.py` and its corresponding receipt
digest. It must reject before any `importlib.import_module` call. These
tests do not deserialize pretrained weights.

This just-in-time check reduces the review/import gap but is NOT an
operating-system-level immutable mount or a cryptographic defense against
an attacker able to mutate files during the import operation itself.
Actual asset and source mounts should be read-only at acceptance.

## B. Two-rank native class import-only probe

The existing `ddp_child_review.py` adds:

`--native-import-probe --project /kaggle/working/<NEW_PRIVATE_PROJECT>`

It requires an explicit two-rank `torchrun` mapping and a previously
created C1-derived private stage. Both ranks independently execute
frozen R0 asset checks, source-stage verification, the fixed native
constructor mapping and the existing Gloo constructor consensus.

Only after both ranks agree does the probe:

- confirm two visible Tesla T4 devices, with fixed `CUDA_VISIBLE_DEVICES=0,1`;
- reauthenticate full staged code at the import boundary;
- import the exact MV00 or MV01 native *class*, checking its module origin;
- use a Gloo collective to propagate a per-rank import failure, or
  confirm that both ranks imported the same class from the same binding.

**The probe does not instantiate the native trainer, read B-TEST, call
`trainer.train()`, call `_do_train()`, load model weights, initialize
NCCL training, perform optimizer steps or save a training checkpoint.**
It is a topology/class-import candidate, not actual CUDA training
acceptance. CI uses synthetic two-process CPU/Gloo source/topology
failure injection, not physical Tesla T4 cards.

The preexisting `--execute` options in both `ddp_worker.py` and
`ddp_child_review.py` remain locked before plan reading/training.

## C. Required evidence for next milestone

- Exact file paths and runtime artifact inventory from an actual
  fresh Kaggle T4×2 session.
- Both-rank full R0/stage provenance and native constructor agreement.
- Live two-T4 native import-only probe pass, without model construction.
- Explicitly bounded, **nonofficial** true NCCL/AMP/model-construction
  and training-step acceptance separately approved and tested.
- Real DDP-B16 rank-specific G4 checkpoint export and independent-session
  pinned resume with interrupted-vs-uninterrupted replay equivalence.
- Rank-0 selected-best B-VAL evaluated exactly 3,049 operational images;
  B-TEST still untouched.

Until those gates pass, report `CPU_REVIEW_ONLY`, not
`GPU_ACCEPTED`, `OFFSESSION_RESUME_VERIFIED` or `OFFICIAL_TRAINING`.
