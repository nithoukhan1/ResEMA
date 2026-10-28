# Chat 28 Step 2F5 — Real-Kaggle IMPORT-ONLY launcher candidate

Status: **READY FOR REAL-KAGGLE TEST; NO REAL GPU EVIDENCE COLLECTED BY CI.**

The new `ddp_kaggle_import_only.py` is an executable **import-only
preflight** and intentionally has no train path. It validates the strict
original fixed plan, independently rehashes pinned R0 assets and frozen C1
source, stages 336 original-plus-patched source files and seven integration
modules in a NEW private directory, and binds the immutable MV00/MV01 native
constructor to the exact source and asset digests.

Its `--run-import-probe` mode invokes exactly one two-rank
`python -m torch.distributed.run` command, preserving the full child log,
requiring the child to verify both T4 devices, rejecting rank disagreement,
and accepting exactly one nontraining rank-0 evidence line. The evidence
receipt contains SHA256 of the plan, staged-source manifest, G4 identity,
constructor binding and stdout/stderr log, along with the observed GPU names,
PyTorch runtime and CUDA version. This cannot claim training or off-session
recovery acceptance.

`--prepare-only` creates only the private verified stage and metadata;
it does not start torchrun or inspect any GPU.

A real session must use the **exact pinned repository HEAD**, and pre-existing,
externally reviewed R0 JSON mapping all pinned Kaggle input files. No code
auto-discovers, downloads or opens the sealed B-TEST dataset. No model
checkpoint is deserialized, no trainer is instantiated and no optimizer
updates occur. The launcher refuses overwriting an existing stage, project
or evidence directory.

A passed GitHub Actions CPU mock is NOT a Kaggle GPU pass. The actual
Kaggle session requires user execution or an authorized connected runtime.
