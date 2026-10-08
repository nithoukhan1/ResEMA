# D4-C3 VGRA Detect-Head / Model-Parser Integration

## Status
`COMPLETE / CPU-VERIFIED`

Branch: `research/vgra-impl-01`
Parent: `7734ec1ab5aac39ae527ea552c572ed6ca1678da`

## Scope
Added a dedicated `VGRADetect(Detect)` subclass and VGRA-specific model YAML.

Stock `Detect` implementation in `head.py` remains unchanged.
Stock detection loss remains unchanged.
No data loader, criterion, trainer or validator integration occurred.

## Files
New:
- `ultralytics/nn/modules/vgra_head.py`
- `ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml`
- `research/tests/test_d4_c3_vgra_head.py`

Minimal generic registration:
- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`

## Pair path
`VGRADetect.forward_pair_heads(AP_features, LAT_features)`

The pair path:
1. computes VGRA visibility/context from the paired Detect-input features;
2. computes native box logits with the untouched native box heads;
3. computes native class logits;
4. adds only the VGRA class residual;
5. returns AP/LAT raw prediction dictionaries plus VGRA context.

The inherited ordinary `forward()` remains stock single-view Detect behavior.

## Box/DFL firewall
Cross-view information is never passed to:
- cv2 box heads;
- DFL;
- decode geometry;
- anchors/strides.

CPU tests prove boxes are exactly identical between native and paired paths both at
zero residual and with an activated VGRA residual.

## Parameter count
Early:
`9570093`

Early + VGRA:
`9672660`

Added:
`102567`

Frozen total cap:
`9870093`

Status:
`PASS`

## Deferred
- pair-aware full model execution;
- batch plumbing;
- visibility loss;
- trainer;
- validator;
- real data;
- GPU training.

## Next
`D4-C4_VGRA_PAIR_AWARE_DATA_AND_BATCHING`
