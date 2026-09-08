# GRAZPEDWRI-DX FINAL RESEARCH PROTOCOL V8
Date: 2026-09-08
Status: Framework locked; exact M1/M2/M3 internals remain conditional on D00 diagnostics.

## Final research objective
Develop a lightweight, clinically motivated and reproducible pediatric wrist abnormality detector that improves a strong COCO-pretrained YOLO11s baseline under a frozen patient-disjoint protocol, while preserving scientific comparability and publication-grade documentation.

## Final working method family
**Anatomy-conditioned, transfer-preserving, patient-effective YOLO11**

### M1 — TP-CDA
Transfer-Preserving Cortical Detail Adapter:
- retain pretrained YOLO11 C3k2 path;
- add small residual fracture-detail adapter;
- zero-initialize the adapter gate/output so the initial network equals the pretrained baseline;
- initial candidate placement: P2/4 and P3/8 high-resolution backbone stages;
- final operators chosen only after D00 localization/detail audit.

### M2 — APCF
Axis-and-Projection Conditioned Fusion:
- exploit GRAZPEDWRI-DX `axis` line annotation as auxiliary supervision;
- predict forearm orientation from P3;
- combine predicted anatomy orientation with verified projection metadata;
- modulate the two main top-down neck fusion nodes;
- initialize conditioning close to identity;
- support unknown/missing projection fallback.

### M3 — PELT
Patient-Effective Long-Tail Training:
- compute rarity using unique positive-patient counts, not only boxes/images;
- capped patient-aware sampling;
- fixed epoch length;
- mild patient-effective class weighting only if sampling alone is insufficient;
- optional geometry strata only if D00 supports them;
- zero inference cost.

## What is NOT part of the default final method
- DySample
- ResEMA
- generic EMA/CBAM/SE attention
- HWD/WTConv/SPDConv copied as headline novelty
- DCNv4/DYHead/Focaler loss unless D00 proves a localization-head bottleneck
- age priors as headline novelty
- cast suppression preprocessing
- paired-view fusion unless D00 confirms feasibility

## Why this direction survived the final literature review
Recent 2026 work already occupies:
- SPDConv + SC-type blocks + DySample + EMA (SDG-YOLO);
- edge enhancement + Slim-Neck + DYHead + class weighting (WFYOLO);
- HWD + multi-scale attention + Slim-Neck + DCNv4 + Focaler-CIoU (FracDet-v11);
- generic class reweighting + Class-Aware Mosaic;
- age-dependent epiphyseal priors + structure-aware diffusion;
- ROI hard-negative adjudication.

GRAZPEDWRI-DX still exposes underused structured information:
- an `axis` line on all 20,327 images;
- projection information in filenames/metadata;
- patient identity and repeated studies;
- severe patient-level long-tail concentration.

The proposed mechanisms exploit these gaps while preserving the strongest finding from our own experiments: transfer learning is highly valuable.

## Project phases
P0 repository/provenance consolidation
P1 D00 no-training diagnostics
P2 M01 TP-CDA
P3 M02 PELT
P4 M03 APCF
P5 combine only passing mechanisms
P6 second-seed confirmation
P7 optional paired-view branch
P8 method lock
P9 3-fold patient-grouped robustness
P10 sealed Split-B test
P11 external/comparability + efficiency + explainability
P12 manuscript + journal submission

## Hard scientific rules
- Split-B test remains sealed until P10.
- Screening uses one seed.
- Second seed only for shortlisted mechanisms/candidates.
- No failed mechanism is retained for narrative convenience.
- Every run originates from a committed Git SHA.
- Every run has a predeclared decision rule.
- Every negative result remains in the registry.
- No raw SOTA claim across incomparable splits.
