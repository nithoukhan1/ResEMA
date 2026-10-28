# Chat28 Step 2C — Exact staged-byte DDP identity candidate

**Status: CPU/REVIEW CANDIDATE; not a GPU run release.**

The pre-existing worker validated read-only historical R0 assets but did not
bind the actual five-file private DDP adapter and other 338 source/module
files to a single proposed DDP G4 identity.

The `--attest-staged` mode is now available after `--review-and-stage`:
it repeats full historical R0 verification and rehashes the existing private
stage's 336 science-source files and seven integration modules, compares every
file to the frozen-derived stage receipt, checks the five adapted files,
and refuses symlinks, extras, missing files, falsified authorization flags and
duplicate JSON keys.

It produces a seven-field candidate G4 identity from the actual checkout HEAD,
frozen assignment and pretrained hashes, fixed DDP-B16 recipe digest and
private patched-source manifest. No editable caller hash overrides or dataset
split changes are permitted.

**Important:** the historical R0 and private source review do not authorize
100 epochs, GPU deserialization, a torchrun child, or CUDA/G4 resume.
The existing `ddp_worker.py --execute` still refuses before loading the plan.
User authorization and actual nonofficial GPU acceptance remain separate.
