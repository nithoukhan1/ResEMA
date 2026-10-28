# Chat28 Step 2A — DDP-B16 native startup/recovery contract

**Status: CODE CANDIDATE, NOT EXECUTABLE/GPU AUTHORIZED.**
Date: 2026-10-28. Branch: `research/vgra-ddp-b16`.

## Source-verified reason for this change

At the prior HEAD `0fca6025d1d2345826b2b123725eaca5653df2b1`,
the DDP launch guard required the legacy batch-eight
`_d4e_recovery_configured` flag while the DDP G4 hook rejected it.
The frozen native engine also required D4E/G3B4 startup binding, even
though DDP-B16 must use separate two-rank G4 and a different recipe.

## Candidate correction

- `ddp_launch_guard.py`: reject legacy D4E/G3B4 and require a distinct
  `_ddp_b16_g4_native_bound` marker plus structurally valid arm-matched identity.
- `ddp_startup_boundary.py`: add a second pre-loop verification after native
  model/optimizer/loader construction. Reject absent authenticated worker,
  missing two-rank G4 binding, wrong recipe/paired dataset, legacy recovery,
  and missing DDP-specific fresh/resume state. Off-session resume remains locked.
- `native_source_adapter.py`: exact-original-blob-gated, private-source-only
  engine patch routes two-rank startup to the new boundary while retaining
  original D4E/G3B4 checks for non-DDP paths.
- Negative tests cover missing authority, legacy recovery, wrong environment,
  corrupt identities, wrong recipes, native pre-loop refusal, and source pins.

## Not implemented or authorized

The existing `ddp_worker.py --execute` still rejects before touching a plan.
There is no authenticated executable torchrun child, actual DDP fresh/resume
binding, real CUDA AMP acceptance, GPU replay, terminal rank-0 best validation,
or official experiment authorization. Structural unit-test acceptance is NOT
a substitute for independently authenticated worker evidence.

## Next tightly scoped engineering step

Create the authenticated fresh-and-resume worker that derives proof from
verified R0 file identities and staged source hashes, initializes actual
trainer state after native setup, restores rank-local optimizer/scheduler/EMA/
scaler/gradients/RNG from externally pinned G4 for resume, and sets the
native pre-loop state only after proving the complete restoration. Add CPU/Gloo
integration tests and rank-0 terminal selected-best validation. Run nonofficial
Kaggle T4x2 only after explicit authorization. Preserve sealed B-TEST.
