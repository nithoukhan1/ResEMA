# COMBINATION-SCREEN-01 — SCConv-Early + Canonical EMA

## Status

REGISTERED — TRAINING NOT AUTHORIZED

Branch:

research/combination-screen-01

Scientific parent:

research/single-module-screen-01 @ 9d65b7adae3d488f1cb70856476fef0488e9749a

## Scientific question

Does combining the independently beneficial SCConv-Early backbone refinement and Canonical EMA head refinement provide complementary validation improvement beyond the strongest single-module candidate under the frozen Split-B Original pretrained development condition?

## Architecture

No new fused Python module is introduced.

- SCConv-Early: backbone layers 2 and 4
- Canonical EMA: head layers 13, 16, 19 and 22
- Detect inputs remain 16, 19 and 22
- DySample excluded
- SCConv-4Stage excluded

Model YAML:

ultralytics/cfg/models/11/yolo11s-scconv-early-canonical-ema-v1.yaml

Model YAML SHA256:

6f9547db340b77b3947698fcd911a70d3cde63328e1190751bc135c11443df14

## Frozen development condition

- Split-B Original
- DATA01:B-ORG:v1
- official pretrained yolo11s.pt
- seed 42
- epochs 100
- imgsz 1024
- global batch 16
- SGD
- frozen baseline recipe
- validation-only architecture selection
- Split-B test access: NONE

## Promotion rule

Strongest SMS-01 single-module reference:

SCConv-Early validation mAP50-95 = 0.43130.

The combination must achieve validation mAP50-95 strictly greater than 0.43130 to demonstrate additional primary-metric value.

Beating baseline 0.41704 alone is not sufficient.

## Pre-audit hypotheses

- parameters: 9574241
- added parameters: 142966
- state-dict items: 565
- new state-dict items: 66
- native reference state items preserved: 499
- expected official-checkpoint transferable items: 493
- expected zero-gate native equivalence: true

All values above remain hypotheses until dedicated COMB-01 audits pass.

## Authorization firewall

TRAINING_AUTHORIZED=FALSE

Split-B test remains sealed.
