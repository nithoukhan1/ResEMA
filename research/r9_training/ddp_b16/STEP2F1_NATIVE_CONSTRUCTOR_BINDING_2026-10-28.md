# Chat 28 — Step 2F1 Exact Native Constructor Contract

Date: 2026-10-28. Candidate status: **review-only, no training authority**.

The module `ddp_native_binding.py` rechecks the entire C1-derived
336-file/private-stage inventory and its parent manifest, verifies the four
frozen R0 input assets and the pinned explicit TRAIN/VAL-only YAML by SHA256,
and binds the exact constructor arguments for MV00/MV01 to a fixed 16-global-
batch, two-T4, 100-epoch configuration.

It rejects any user-supplied training hyperparameter override, arbitrary
checkpoint, untrusted stage receipt, B-TEST path, duplicate class identity,
noncanonical output location and any attempt to treat historical batch-eight
R0 verification as permission to train with DDP-B16.

The returned `binding_sha256` derives from the exact on-disk paths, fixed
recipe, 7-field G4 identity and model-specific native constructor mapping.
Both ranks must independently derive the same binding and compare it
before any model import or GPU activity. The binding cannot authorize
`--execute` and does not deserialize pretrained weights.

Limits: CI tests use synthetic path checks and mocks because the private
radiographs and actual pretrained checkpoint are not present in CI. The
production binding function MUST rehash actual local Kaggle input files.
No native trainer has been instantiated here; no CUDA/NCCL acceptance;
no official MV00/MV01 training; B-TEST remains sealed.
