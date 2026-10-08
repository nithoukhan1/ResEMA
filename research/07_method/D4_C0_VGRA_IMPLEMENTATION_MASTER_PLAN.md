# D4-C0 VGRA Implementation Master Plan

## D4-C1 — standalone VGRA core
Implement `ultralytics/nn/modules/vgra.py`: descriptors, four-state visibility predictor, AP/LAT target coefficients, low-rank compatibility, bounded zero-initialized residual. Unit-test shapes, state mapping, gate signs, stop-gradient, rho=0 identity, bounds, finite backward. No data/trainer edits.

## D4-C2 — deterministic pair/visibility contract
Build exact TRAIN/VAL pair manifests from frozen metadata. Account for every valid image exactly once, preserve singles, verify patient isolation, freeze the four-state target schema, and prove test access NONE. Actual state targets/counts/weights are deferred to D4-C5 and must come from B-TRAIN YOLO labels only.

## D4-C3 — detector/head integration
Connect VGRA to P3/P4/P5 classification logits only. Native box/DFL stays unchanged. Prefer a VGRA-specific Detect subclass/adapter with minimal parser registration. Prove VGRA-off/rho=0 identity and box invariance.

## D4-C4 — pair-aware dataset and batching
Create pair-preserving batches with explicit companion map/view code/pair-valid mask. Paired members must co-batch; singles remain allowed; no artificial duplicate companion; no silent image dropping; seeded deterministic shuffle. Disable cross-study composition augmentation for both MV-00 and MV-01.

## D4-C5 — criterion/trainer/validator
Subclass stock detection components where possible. Keep native detection loss and add `0.25 * L_vis` only for valid pairs. Visibility state weights come from B-TRAIN only. Pair-aware validator returns ordinary per-image predictions; singles/unreadable companions use exact single-view fallback.

## D4-C6 — implementation closure/config
Freeze VGRA config, exact changed files, constants, parameter count, checkpoint transfer contract, pair-safe augmentation and runtime dispatch. Exit with `VGRA_IMPLEMENTED=TRUE` and training still locked.

## D4-D1/D2/D3 — verification
D1: static/unit suite. D2: transfer, zero-gate identity, missing-view fallback, box/DFL invariance, save/load, parameter cap <= 9,870,093. D3: synthetic pair/single/mixed pipeline forward/backward/validator tests with no NaN/Inf.

## D4-E — source freeze + training contract
Freeze source commit, hashes, dataset bindings, pair manifests, pair-safe augmentation, seed, pretrained initialization, optimizer/schedule, image size, epoch budget, preservation paths and evaluator. Training requires a separate authorization.

## D4-F — first two GPU runs only
`D4-MV-00`: Early + identical pair-aware pipeline, VGRA disabled.
`D4-MV-01`: same recipe + full VGRA V1.
No hyperparameter sweep before both are complete and preserved.

## D4-G/H/I/J onward
G: D3-compatible diagnostics + VGRA-specific visibility/rescue analysis. H: attribution ablations only if PROMOTE/HOLD. I: optional extra support module only for an evidenced residual weakness. J: final architecture freeze. Then D5 multiseed/efficiency/robustness, D6 fair comparator reproduction, D7 final-test contract, D8 one sealed B-test transaction, D9 manuscript evidence.

## D4-C5A/B/C implementation subdivision

D4-C5 is subdivided to avoid combining three independent high-risk subsystems.

- C5A: synthetic-tested visibility class-target and objective semantics; never uses real images or labels.
- C5B: paired feature/head dispatch and correct native loss normalization for mixed pair/single batches.
- C5C: pair-aware trainer and validator with controlled synthetic end-to-end tests.

C5 is not complete until all three parts are remote-verified.
A signed-beta semantic gate must be resolved via a governed decision before training.
