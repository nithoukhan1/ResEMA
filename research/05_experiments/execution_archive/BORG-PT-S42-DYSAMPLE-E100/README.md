# BORG-PT-S42-DYSAMPLE-E100

## Purpose
Single-module screen Candidate 3: YOLO11s + DySample.

## Archive status
- Code status: `EXACT_CAPTURE`
- Scientific execution status: `RUNNING`
- Kaggle preservation mode: `KAGGLE_SAVE_VERSION`
- The archived Kaggle wrapper was executed without scientific modification and training started successfully on Kaggle T4x2.

## Frozen scientific binding
- Branch: `research/single-module-screen-01`
- Training source: `660475b3aa3614f58f21c98f760561c72e3f1436`
- Execution/authorization commit: `9d65b7adae3d488f1cb70856476fef0488e9749a`
- Canonical runner: `research/runtime/single_module_runner.py`
- Canonical resume runner: `research/runtime/single_module_resume_runner.py`
- Data binding: `DATA01:B-ORG:v1`
- Initialization: pretrained
- Seed: 42
- Epochs: 100
- Test access: NONE

## Reproducibility note
- PyTorch emitted a warning that CUDA `grid_sampler_2d_backward` has no deterministic implementation under the active runtime.
- This warning is expected from DySample because its implementation uses `torch.nn.functional.grid_sample`.
- The frozen recipe remains unchanged; the run is retained and the warning is documented as a module-specific reproducibility caveat.
