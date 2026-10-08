# D4-C0 VGRA Implementation Governance

## Branch isolation

Frozen scientific design authority:
`research/lpq-method-01 @ a5d260ca83236c774e1920ce729e08eaacfcf5d4`

VGRA implementation authority:
`research/vgra-impl-01`

Do not implement VGRA on the design branch. Do not manually copy modified source files between worktrees.

## Transaction discipline

One governed transaction per commit. No force-push, squash, or rebase of remote-verified D4 commits. A failed transaction stops; corrections are issued as full R2/R3 scripts.

## Firewalls

Before D4-E: no GPU training, no validation model-selection inference, no B-test access, no attention, no hyperparameter search, no extra modules. Synthetic fixtures are allowed for structural tests.

## Artifact policy

Git may contain source, config, small manifests, tests, and small JSON/Markdown evidence. Checkpoints, datasets, training runs, large prediction CSVs, plots and ZIPs belong outside Git under `artifacts_external/` with provenance registration.

## Upstream-edit policy

Prefer additive VGRA-specific files/subclasses. Generic Ultralytics files may be touched only for minimal registration/dispatch. Stock non-VGRA behavior must remain unchanged.

## Augmentation firewall

Cross-study Mosaic/MixUp/CutMix-style composition is forbidden in paired VGRA mode unless a future governed pair-preserving formulation is explicitly frozen. The paired control and VGRA run must use the identical pair-safe recipe.

## Test firewall

`B_TEST_ACCESS=NONE` until D7/D8.
