# ARCH-CORR-01A — Historical Implementation Forensic Audit

## Status

`COMPLETE — HISTORICAL IMPLEMENTATIONS CLASSIFIED`

This record explains why the historical C3k2_SC / DySample / ResEMA experiments remain
useful exploratory evidence but cannot all be treated as clean causal module ablations
for the new study.

## Historical C3k2_SC

Native YOLO11 `C3k2` inherits C2f semantics:
- one `cv1` projection to two hidden partitions;
- progressive execution through `m`;
- retention/concatenation of intermediate outputs;
- final native `cv2` projection;
- positional semantics `(c3k, e, attn, g, shortcut)`.

Historical `C3k2_SC` instead uses:
- independent `cv1` and `cv2` projections;
- SC blocks on one branch;
- a two-terminal-branch concatenation;
- final `cv3`;
- constructor semantics `(shortcut, e)` after `n`.

Therefore historical YAML positional values that were native `c3k` values can be
reinterpreted as `shortcut` values, and native C2f/C3k2 aggregation disappears.

Consequences:
- SC effect is confounded with topology change;
- native C3k2 pretrained parameter paths/shapes are not preserved in replaced blocks;
- capacity increases materially;
- historical SC results are not clean evidence for Self-Calibrated Convolution alone.

Historical parameter registry:
- YOLO11s 9-class baseline: 9,431,275 parameters;
- historical SC-only: 10,649,355;
- increase: 1,218,080 (~12.92%).

Disposition: **do not reuse C3k2_SC for new experiments**.

## Self-Calibrated Convolution core

The project `SCConv` core retains the expected context-calibration structure:
average-pooled context branch, local branch, sigmoid calibration, multiplicative
feature calibration and final convolution.

Disposition: **retain the SCConv operator, replace the historical wrapper**.

## DySample

The current DySample operator retains the LP/PL point-sampling design, learned offset
initialization, optional scope modulation, pixel shuffle/unshuffle path and
`grid_sample(..., padding_mode="border", align_corners=False)` behavior.

Historical model-count difference is consistent with replacing the two YOLO11s nearest
upsamplers by LP DySample at channels 512 and 256:
- expected added parameters: 16,416 + 8,224 = 24,640;
- historical registry difference: 24,640.

Disposition: **retain; verify locally and in focused tests; no redesign in ARCH-CORR-01**.

## Historical ResEMA-V2

Historical ResEMA-V2:
- adds a preliminary channel bottleneck;
- creates directional H/W gating;
- builds normalized and 3x3 branches;
- reduces branches to channel-softmax descriptors;
- combines branches by channel weighting;
- adds an external residual connection.

It does not implement the canonical EMA cross-spatial descriptor-to-opposite-branch
matrix products that create an HxW attention field.

Disposition: **retire ResEMA-V2 from the new architecture; preserve results as history**.

## Replacement contracts

- `C3k2_TPSC` / `C3k2_TPSCG4`: native C3k2 topology preserved; new SC adapters are zero-gated.
- `CanonicalEMA`: canonical cross-spatial EMA operator.
- `C3k2_TPEMA`: native C3k2 preserved; canonical EMA added through a zero-gated adapter.
- DySample: retained unchanged.

No corrected-module training is authorized by this audit.
Split-B test remains sealed.
