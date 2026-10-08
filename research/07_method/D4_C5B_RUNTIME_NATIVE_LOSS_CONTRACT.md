# D4-C5B VGRA Mixed-View Runtime and Native Loss Contract

## State

`IMPLEMENTED / CPU SYNTHETIC VERIFIED`

Branch: `research/vgra-impl-01`
Parent: `b7edfcc35697b2b29bd5d98fb9bd087abb4de2b4`

## Scope

Two additive source/test files:
- `ultralytics/models/yolo/detect/vgra_runtime.py`
- `research/tests/test_d4_c5b_vgra_runtime_loss.py`

No stock Ultralytics file is changed. No dataset, trainer or validator file is
changed. No real images/YOLO labels, GPU training, or B-TEST access.

## Model dispatch

`VGRADetectionModel` subclasses the existing `DetectionModel`.

For a VGRA training/validation batch, `forward_vgra_batch(batch)`:

1. validates reciprocal pair metadata before any feature computation;
2. processes the complete image batch through the shared YOLO backbone/neck
   exactly once, preserving common BatchNorm statistics;
3. runs native Detect box/class heads on the entire batch exactly once;
4. selects valid paired AP/LAT feature rows and evaluates the frozen VGRA core;
5. adds per-view VGRA residuals to only the corresponding class-score rows
   using out-of-place tensor index accumulation;
6. returns all image predictions in original batch order;
7. returns an empty visibility-logit tensor if the batch contains no pairs.

Singles receive EXACT native classification logits. The native box/DFL tensor
is returned unchanged for all images. The ordinary `.predict(Tensor)` method
is still inherited native Detect behavior.

The previously established C3 pair-head route remains available but is NOT
re-executed per pair during mixed-batch training, because doing so would
recompute batch-normalized classification heads on smaller subsets and break
the exact rho=0 full-batch native identity contract.

## Native detection objective: fork-specific requirement

THIS FORK's `v8DetectionLoss` returns:
- weighted native loss VECTOR of shape (3,), each component multiplied by
  the number of images in the batch;
- detached display vector of shape (3,).

The stock `BaseTrainer` explicitly calls `loss.sum()` before backward.

Therefore the VGRA criterion returns:
- **four-component weighted loss vector**:
  `[box_total, cls_total, dfl_total, vis_total]`;
- four-component detached display vector:
  `[box_display, cls_display, dfl_display, vis_display]`.

It does not pre-sum native components or alter their scaling.

Frozen VGRA visibility objective:
`vis_mean = mean(pair,class)[class_state_weight * CE(logits,state)]`.

Let N=number of input images and P=number of valid paired studies.

`vis_total = 0.25 * (2*P) * vis_mean`

`vis_display = vis_total/N`.

Thus an all-paired batch uses weight 0.25 per image, matching the frozen
paired objective, while singles contribute only their native detection loss.

When P=0:
- native predictions remain exact;
- `vis_total=0`;
- TRAIN state weights are not required.

When P>0, actual TRAIN-only state weights must have been explicitly bound.
The weights and readiness flag are persistent non-trainable model buffers,
not extra learned parameters. VAL/TEST weight binding is forbidden.

## Invariance tests

Synthetic CPU tests require:
- inherited native single-image inference path;
- rho=0 exact native mixed-batch class/box identity;
- active residual changes paired scores only;
- unpaired image class scores exactly native;
- box/DFL unchanged even with active VGRA;
- native detection-loss components exactly preserved;
- visibility scaling `0.25 * 2P * weighted_CE`;
- TRAIN-weight binding gate and B-VAL rejection;
- empty paired set native identity and zero visibility loss;
- finite gradient through native bbox/class/visibility/gate modules;
- malformed companion map rejection;
- weights persistent in state_dict;
- unchanged trainable model parameter count: 9,672,660.

## OPEN gates

- Signed-beta semantic polarity remains OPEN; no V1 math was changed.
- No real B-TRAIN state-weight values have yet been computed.
- Full trainer/validator routing, EMA behavior, checkpoint restoration,
  end-to-end synthetic pipeline, pair-aware validation decoding and
  experiment execution are D4-C5C/D4-D work.
- `compile=True`/DDP unsupported in VGRA V1 pending governed verification.

## Next

`D4-C5C_VGRA_TRAINER_VALIDATOR_INTEGRATION`
