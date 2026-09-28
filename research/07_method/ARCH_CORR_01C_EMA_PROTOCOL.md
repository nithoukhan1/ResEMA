# ARCH-CORR-01C — Canonical EMA Transfer-Preserving Prototype

## Status

`IMPLEMENTED FOR STATIC / REFERENCE / TRANSFER AUDIT — TRAINING NOT AUTHORIZED`

This work continues on `research/arch-corr-01b` while the E200 scratch calibration
runs independently on `research/baseline-refresh`.

## Why historical ResEMA-V2 is not reused

Historical ResEMA-V2 is a project-specific EMA-inspired module. It adds a preliminary
channel bottleneck, replaces the paper's cross-spatial matrix products with
channel-softmax branch weighting, and adds an external residual connection.

Those changes make it unsuitable as evidence for canonical Efficient Multi-Scale
Attention (EMA).

## Canonical EMA operator

`CanonicalEMA` follows the authors' released EMA structure:

1. reshape channel groups into the batch dimension;
2. pool along H and W and fuse them with a 1x1 convolution;
3. gate grouped features and normalize them to form branch 1;
4. use a 3x3 convolution to form branch 2;
5. obtain two global channel descriptors with softmax;
6. cross-multiply each descriptor with the other branch's spatial features;
7. sum the two cross-spatial terms;
8. sigmoid the resulting HxW attention field;
9. reweight the original grouped features.

Default `factor=32` matches the released reference implementation.

## Transfer-preserving integration

A standalone inserted attention layer would shift downstream YOLO module indices and
break ordinary checkpoint key matching. ARCH-CORR-01C therefore uses
`C3k2_TPEMA`, a native `C3k2` subclass at the same layer index.

Native members remain:

- `cv1`
- `cv2`
- `m`

The post-C3k2 adapter computes:

`y_out = y + tanh(alpha) * EMA(y)`

with `alpha=0` initially.

This gives exact native behavior at initialization after native weight transfer, while
all new EMA parameters are explicit additions.

## Placement candidate

`yolo11s-tpema-head-v1.yaml` replaces the four head C3k2 stages at native indices
13, 16, 19 and 22. The backbone and Detect index remain unchanged.

At YOLO11s scale their output channels are 256, 128, 256 and 512. With factor=32 the
expected added trainable parameter count is 4,148 (including four scalar gates), for
an expected target total of 9,435,423 versus the 9,431,275 baseline.

## Required gates before training

- reference-equation equivalence for `CanonicalEMA`;
- native constructor/positional semantics preserved;
- native 499-state-item structural contract preserved;
- whole-model zero-gate output identity;
- official locked yolo11s.pt still transfers exactly 493 source state items;
- six native class-output items remain the only baseline-native nontransfer items;
- every other nontransferred target item belongs only to the new EMA adapters;
- no training and no Split-B test access during audit.

DySample remains unchanged and is not combined with EMA or TPSC until the single-module
contracts are independently closed.

## Verification closure candidate

Verified implementation head:
`a30df2165cb24a0796086a32b07445ea04b5c7bc`.

Canonical EMA/TPEMA passed both remote and local exact-head transfer/identity
verification. Local evidence is registered in
`research/01_provenance/ARCH_CORR_01_VERIFICATION_LOCK.json`.

This note does not authorize training; final ARCH-CORR-01 closure attestation remains
required.

