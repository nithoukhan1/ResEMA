# ARCH-CORR-01B — Transfer-Preserving Self-Calibrated C3k2 Prototype

## Status

`IMPLEMENTED_FOR_STATIC_AND_TRANSFER AUDIT — TRAINING NOT AUTHORIZED`

Branch: `research/arch-corr-01b`

Base commit: `e59edd55ae7be1fb3b185e9224e192125fbb63fe`

This branch is intentionally separate from `research/baseline-refresh` while
`BASE-B-ORG-SCR-S42-E200-CAL` is running.

## Motivation

The historical `C3k2_SC` changes the native YOLO11 C3k2 topology and positional
semantics. It uses two independent projections, keeps only two terminal branches,
renames the native final projection, removes native C2f progressive aggregation, and
does not preserve the native `c3k` meaning. Historical SC results therefore combine
Self-Calibrated Convolution with topology, capacity and transfer changes.

The SCConv core itself is retained because its k2/k3/k4 context-calibration structure
is consistent with Self-Calibrated Convolution (SCNet, CVPR 2020).

## New module contract

`C3k2_TPSC` subclasses native `C3k2`.

Native parameterized members remain exactly:

- `cv1`
- `cv2`
- `m`

After each native `m[i]` transform, a new SC residual adapter computes:

`z_out = z + tanh(alpha) * SCConv(z)`

with `alpha=0` at initialization.

Therefore, after copying all native C3k2 weights, the TPSC block is functionally
identical to native C3k2 at initialization while preserving a learnable path into
self-calibration.

The constructor preserves native positional semantics through:

`(c1, c2, n, c3k, e, attn, g, shortcut)`

SC-specific controls are keyword-only.

`C3k2_TPSCG4` is a fixed four-group adapter variant with the same native positional
signature, avoiding ambiguous YAML argument reinterpretation.

## Candidate A — early-stage full SC

YAML:
`ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml`

Replace only backbone layers 2 and 4.

Expected added trainable parameters for YOLO11s-9C:

- layer 2 hidden c=32: 27,841
- layer 4 hidden c=64: 110,977
- total: 138,818
- expected target params: 9,570,093
- increase over 9,431,275 baseline: about 1.47%

The deeper native C3k2 blocks and the entire head remain unchanged.

## Candidate B — four-stage grouped SC

YAML:
`ultralytics/cfg/models/11/yolo11s-tpsc-g4-v1.yaml`

Replace backbone layers 2, 4, 6 and 8 with four-group TPSC.

Expected added trainable parameters:

- c=32: 7,105
- c=64: 28,033
- c=128: 111,361
- c=256: 443,905
- total: 590,404
- expected target params: 10,021,679
- increase over baseline: about 6.26%

## Transfer contract

Before any training is authorized:

1. every native nine-class YOLO11s state key must exist in each TPSC target with the
   same shape;
2. native state tensors must load without renaming or positional remapping;
3. the only additional target state belongs to `sc_adapters`;
4. native block output and TPSC output must match at zero-gate initialization after
   native state transfer;
5. with the locked official `yolo11s.pt`, the same 493 source tensors transferable
   to the ordinary nine-class baseline must also transfer to both TPSC targets;
6. the six class-output tensors remain the only baseline-native target tensors that
   cannot transfer from the 80-class checkpoint;
7. all additional nontransferred tensors must belong to newly introduced SC adapters.

This is a shared-weight transfer contract, not a target-total-percentage requirement.

## DySample / ResEMA disposition

DySample is not modified in ARCH-CORR-01B.

Historical ResEMA-V2 is not authorized for reuse in a new architecture. A canonical
EMA implementation, if retained, will be implemented and audited separately.

## Training firewall

No architecture training is authorized by this commit.

No Split-B test data may be accessed.

The E200 calibration continues independently on `research/baseline-refresh`.

## Verification closure candidate

Verified implementation head:
`a30df2165cb24a0796086a32b07445ea04b5c7bc`.

The TPSC transfer/identity contract passed both remote and local exact-head
verification. Local evidence is registered in
`research/01_provenance/ARCH_CORR_01_VERIFICATION_LOCK.json`.

This note does not authorize training; final ARCH-CORR-01 closure attestation remains
required.

