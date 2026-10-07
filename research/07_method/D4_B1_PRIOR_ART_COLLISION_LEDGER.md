# D4-B1 Prior-Art Collision Ledger

## Purpose

This document records the first formal novelty gate for the LPQ concept before any
implementation or training.

The conclusion is intentionally conservative:

`THE ORIGINAL DAR/GDS SKETCH MUST NOT BE IMPLEMENTED AS THE NOVELTY CORE.`

This is not a rejection of the D3 residual problem. It is a rejection of the first
solution sketch because important parts overlap directly with established literature.

## Repository-side finding

The current YOLO11 detection loss already uses Task-Aligned Assignment.

For each candidate, the assignment metric combines predicted class score and overlap:

`t = s^alpha * IoU^beta`

The normalized alignment signal then becomes a soft classification target.

Therefore, a simple proposal of "make classification confidence reflect localization
quality" is not a new mechanism in this codebase.

## Collision 1 — original DAR sketch

Original DAR concept:
- obtain localization reliability from the DFL bounding-box distributions;
- use that quality to reconcile/rerank the classification score.

Direct prior-art collision:

**Generalized Focal Loss V2 (GFLV2), CVPR 2021**
"Learning Reliable Localization Quality Estimation for Dense Object Detection."

GFLV2 explicitly:
- extracts statistics from learned bounding-box distributions;
- feeds them to a small Distribution-Guided Quality Predictor;
- predicts localization/IoU quality;
- combines classification representation and predicted quality for ranking.

Disposition:

`DAR_ORIGINAL_DISPOSITION = REJECT_AS_NOVELTY_CORE`

Using DFL distribution statistics for a localization-quality branch may still be used
later as a published control, but it cannot be presented as the central LPQ novelty.

## Collision 2 — generic quality-aware scoring

Relevant prior art includes:

- Generalized Focal Loss (NeurIPS 2020);
- VarifocalNet (CVPR 2021);
- TOOD / Task Alignment Learning (ICCV 2021);
- the TaskAlignedAssigner already present in this repository.

These methods already address classification/localization quality alignment through
soft quality targets, IoU-aware scores or task-aligned assignment.

Disposition:

`GENERIC_QUALITY_SCORE_FUSION = NOT_NOVEL_ENOUGH`

## Collision 3 — original GDS score-gap sketch

Original GDS concept:
- retain one-to-many supervision;
- identify a best candidate for each GT;
- force a confidence gap between the best candidate and secondary duplicates.

Direct/near-direct prior art includes:

- YOLOv10 consistent dual assignments (NeurIPS 2024);
- Ultralytics one-to-many + one-to-one end-to-end heads;
- DHLA (IEEE TCSVT 2025), which combines hybrid label assignment with a ranking loss
  that widens the score gap between the highest-scoring position and surrounding
  candidates to remove duplicate boxes;
- Group DETR / H-DETR and related hybrid matching schemes.

Disposition:

`GDS_ORIGINAL_DISPOSITION = REJECT_AS_NOVELTY_CORE`

A standard one-to-one or dual-assignment head remains useful as a CONTROL, not as the
claimed novelty.

## What remains scientifically valid

The D3 residual problem remains valid and important:

1. Early reduces several false-positive modes;
2. Early does not provide a robust net lesion-recovery gain;
3. Early becomes sensitivity-fragile at higher confidence;
4. duplicate false positives increase.

The failure is therefore in the first LPQ solution sketch, not in the D3 diagnosis.

## D4-B1 decision

- keep `LPQ` only as a temporary project container name;
- retire the original DAR formulation as a novelty claim;
- retire the original GDS ranking-gap formulation as a novelty claim;
- do not implement either formulation yet;
- redesign from the D3 residual with an explicit prior-art exclusion list;
- keep stock YOLO11, GFLV2-like quality estimation, and standard one-to-one/dual
  assignment as possible controls.

## Literature anchors

1. Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding
   Boxes for Dense Object Detection," NeurIPS 2020.
2. Li et al., "Generalized Focal Loss V2: Learning Reliable Localization Quality
   Estimation for Dense Object Detection," CVPR 2021.
3. Zhang et al., "VarifocalNet: An IoU-Aware Dense Object Detector," CVPR 2021.
4. Feng et al., "TOOD: Task-Aligned One-Stage Object Detection," ICCV 2021.
5. Wang et al., "YOLOv10: Real-Time End-to-End Object Detection," NeurIPS 2024.
6. "DHLA: Dynamic Hybrid Label Assignment for End-to-End Object Detection,"
   IEEE Transactions on Circuits and Systems for Video Technology, 2025,
   DOI: 10.1109/TCSVT.2024.3470230.

## Next research question

D4-B2 must search for a formulation that is:

- directly tied to the D3 lesion-preservation residual;
- not merely another localization-quality predictor;
- not merely score-times-IoU;
- not merely one-to-one / dual assignment;
- not merely a pairwise ranking margin between duplicate candidates;
- preferably training-time only or very low-overhead at inference;
- explicitly testable by ablation;
- compatible with the existing YOLO11 one-to-many detector.

Candidate ideas discussed in D4-B2 remain hypotheses until their own prior-art search
is complete.
