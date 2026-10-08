# D4-C0 VGRA File-Change Map

## Preferred additive files
- `ultralytics/nn/modules/vgra.py` — core VGRA math.
- dedicated VGRA data/pairing module (exact path frozen in D4-C4).
- dedicated VGRA trainer/model/validator module under `ultralytics/models/yolo/detect/` (exact split frozen in D4-C3/C5).
- `ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml`.
- phase-prefixed tests under `research/tests/`.

## Generic files allowed only if required
Potential minimal registration/dispatch edits: `ultralytics/nn/modules/__init__.py`, `ultralytics/nn/tasks.py`, `ultralytics/models/yolo/detect/__init__.py`, `ultralytics/models/yolo/model.py`.

## Preferred unchanged
- stock `ultralytics/utils/loss.py` (use subclass/composition instead);
- stock `Detect` behavior for non-VGRA models;
- historical DySample/EMA code;
- frozen Split-B manifests;
- all B-test content.

If a preferred-unchanged file becomes necessary, stop and document the blocker before editing it.

## Documentation ownership
`research/07_method/`: method/implementation/verification.
`research/04_data/`: deterministic pair manifests.
`research/05_experiments/`: training contracts and small results.
`research/06_diagnostics/`: residual/error analyses.
`research/08_robustness/`: multiseed/efficiency.
`research/09_final_test/`: locked until D7/D8.
`research/10_manuscript/`: manuscript evidence after method stabilization.
