# ARCH-CORR-01D — DySample Retention Verification

## Status

`RETAINED OPERATOR — FOCUSED VERIFICATION REQUIRED BEFORE NEW TRAINING`

The current DySample implementation is not redesigned.

Verification scope:
- LP and PL output shapes;
- offset initialization contract;
- differentiability / finite gradients;
- invalid-configuration fail-closed behavior;
- YOLO11s DySample model parameter delta;
- no test data and no training.

For YOLO11s scale-s, replacing the two nearest-neighbor upsampling operations at
feature channels 512 and 256 with LP DySample (scale=2, groups=4) adds:
- 512 -> 32 offset conv: 512*32 + 32 = 16,416 parameters;
- 256 -> 32 offset conv: 256*32 + 32 = 8,224 parameters;
- total: 24,640.

Expected 9-class target:
- baseline: 9,431,275;
- DySample: 9,455,915.

This contract matches the historical registry and is used as an implementation-identity
check, not as evidence of detection benefit.
