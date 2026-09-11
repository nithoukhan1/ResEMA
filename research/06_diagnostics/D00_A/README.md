# D00-A ? Pretrained Transfer Coverage

Status: implementation contract in progress.

D00-A measures how much of the official YOLO11s pretrained checkpoint
is actually transferable into each historical architecture before any
new publication-model mechanism is implemented.

## Scientific question

Is pretrained-transfer disruption real, and which historical module or
module combination causes it?

## Architecture matrix

The audit covers exactly eight YOLO11s-scale architectures:

| ID | Architecture |
|---|---|
| `baseline_yolo11s` | baseline YOLO11s |
| `sc_e50` | SC-e50 |
| `dysample` | DySample-only |
| `resema` | ResEMA-only |
| `sc_dysample_e50` | SC-e50 + DySample |
| `sc_resema_e50` | SC-e50 + ResEMA |
| `dysample_resema` | DySample + ResEMA |
| `full_e50` | SC-e50 + DySample + ResEMA |

Seeds are not duplicated because transfer coverage is an architecture
property, not a stochastic training outcome.

## Transfer definition

The primary view reproduces the repository's actual Ultralytics loading
contract: a source state-dict tensor transfers only when the target has
the same key and the exact same tensor shape.

The audit additionally reports a stricter same-top-level-module-type
subset. This prevents layer-index shifts caused by inserted modules from
being interpreted as architecturally equivalent transfer merely because
a key and shape happen to match.

## Detect-head control

The source checkpoint is the official 80-class YOLO11s checkpoint while
the target task has 9 classes.

Therefore D00-A reports both:

1. overall parameter transfer coverage; and
2. non-Detect transfer coverage.

The non-Detect view prevents the expected 80-to-9 class-head difference
from being mislabeled as backbone/neck transfer disruption.

The unchanged baseline must achieve exactly 100% non-Detect parameter
coverage. The runner fails if it does not.

## Source checkpoint policy

`yolo11s.pt` is external.

The runner does not automatically download a checkpoint. The user must
provide an explicit `--weights` path. The SHA256 of that exact file is
recorded in the run provenance.

Before any architecture is audited, the checkpoint topology is checked
against an 80-class YOLO11s reference built from this repository.

## Runtime provenance

The D00-A0 Kaggle runtime binding is stored in:

`D00_A0_RUNTIME_BINDING.json`

Binding SHA256:

`eddffe25ded04c7df598d76769a3dd224a794efe542da040a25dadc3c16f90ef`

The transfer audit itself does not open dataset images or labels.

## Outputs

A successful runtime audit produces:

- `transfer_coverage.csv`
- `transfer_unmatched_tensors.csv`
- `transfer_unmatched_layers.csv`
- `transfer_type_mismatch_matches.csv`
- `D00_A_RUN_PROVENANCE.json`
- `D00_A_TRANSFER_COVERAGE_REPORT.md`

`transfer_unmatched_tensors.csv` contains one row per unmatched target
state-dict tensor.

`transfer_unmatched_layers.csv` aggregates those unmatched tensors by target
top-level model module; it is therefore genuinely layer-level evidence rather
than a renamed tensor-key table.

These outputs are generated into a user-specified empty output directory and
are not produced by the implementation transaction.

## Governance

Training: NONE.

Dataset inference/evaluation: NONE.

Split-B test predictions, metrics, errors, and diagnostic outcomes remain
sealed until P10.

No D00-A result may be used to access or justify pre-P10 test analysis.
