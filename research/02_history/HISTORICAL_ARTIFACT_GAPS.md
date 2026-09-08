# Historical Artifact Gaps

Date: 2026-09-08

This file records historical provenance fields that could not be
reconstructed with sufficient confidence.

Missing information is intentionally recorded rather than guessed.

## B0

Known:
- historical recipe/pipeline anchor
- independent validation results are preserved in EXPERIMENT_REGISTRY.csv
- experiment was completed

Not currently reconstructed:
- immutable Kaggle Model path/version
- best.pt SHA256
- last.pt SHA256
- exact artifact byte sizes

Policy:
Do not invent these values. Populate them only if an original Kaggle
artifact, historical handoff, notebook output, or archived manifest is found.


## B1

Known:
- clean legacy recipe experiment
- independent validation results are preserved
- experiment was completed

Not currently reconstructed:
- immutable Kaggle Model path/version
- best.pt SHA256
- last.pt SHA256
- exact artifact byte sizes

Policy:
Do not invent these values.


## B2

Known and verified:
- best.pt SHA256:
  40e9b1afdbbe237b3e8dae3e513dad2d82ec45a0cabbbc8b4ac8cab9db8f2bfe

Not currently reconstructed:
- exact immutable Kaggle Model path/version
- byte size

The verified hash is preserved in ARTIFACT_REGISTRY.csv.


## A0/B3 seed42

Known and verified:
- Kaggle model directory:
  /kaggle/input/models/vickyhit/b3-2/pytorch/default/1/B3_splitB_clean_conservative_pretrained_seed42

- best.pt SHA256:
  08977fa37c4c3d764e284ad1fcc130b292cbd6942b06065a39fcf749ccbcc8d5

Not currently reconstructed:
- exact checkpoint byte size


## Resolution rule

Historical gaps do not block future experiments.

If an original artifact is later found:
1. verify it independently;
2. compute SHA256;
3. update ARTIFACT_REGISTRY.csv;
4. append the change to DECISION_LOG.md / PROGRESS_LOG.md;
5. never silently overwrite the historical record.
