# D4 LPQ Redesign Constraints

## Goal

Redesign the solution while preserving the evidence-backed D3 problem statement.

## Must solve

The next candidate must target at least two of the following without sacrificing the
others:

- true-positive preservation;
- confidence robustness;
- duplicate reduction;
- false-positive suppression already achieved by Early;
- localization quality;
- efficiency.

## Prior-art exclusions

The novelty core must NOT be:

1. DFL/distribution statistics -> IoU-quality predictor alone;
2. class score multiplied/fused with predicted IoU alone;
3. TOOD/TAL-style task-aligned scoring alone;
4. generic one-to-one matching alone;
5. YOLOv10-style dual assignment alone;
6. simple pairwise margin/ranking loss between best and duplicate candidate alone;
7. a second-stage ROI adjudicator + score fusion;
8. broad attention/module stacking without an evidence-backed residual.

## Allowed as controls

The following may later be implemented as baselines/controls:

- stock YOLO11s;
- SCConv-Early;
- GFLV2-like distribution-guided quality estimation if reproducible;
- standard Ultralytics end-to-end one-to-one/one-to-many head;
- a published duplicate-suppression or quality-aware control.

Controls must never be relabeled as proposed novelty.

## Candidate search direction for D4-B2

The next search should focus on **lesion-level set supervision** rather than independent
candidate rescoring.

One hypothesis to investigate, NOT YET APPROVED:

`lesion-wise probability preservation + candidate-score concentration`

The distinction to test is whether the objective can:
- supervise a GT-associated candidate set as a set;
- guarantee/protect at least one confident representative per lesion;
- concentrate score mass so redundant candidates do not all remain high;
- preserve one-to-many box/feature supervision;
- add no or negligible inference overhead.

This hypothesis is not yet claimed novel. D4-B2 must perform a dedicated prior-art
review before any mathematical freeze.

## Implementation firewall

Until D4-B2 closes:

`LPQ_IMPLEMENTATION_AUTHORIZED = FALSE`

`GPU_TRAINING_AUTHORIZED = FALSE`

`ATTENTION_AUTHORIZED = FALSE`

`B_TEST_ACCESS = NONE`
