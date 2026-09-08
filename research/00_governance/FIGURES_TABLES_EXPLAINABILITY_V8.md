# Figures, Tables and Explainability Plan V8

## Main manuscript figures
### Figure 1 — Study design and data flow
- original dataset
- patient grouping
- Split-B train/validation/test
- sealed test
- development/CV/external evaluation flow

### Figure 2 — Dataset challenge
- object counts vs unique positive-patient counts
- rare-class concentration
- box geometry distribution
- projection distribution

### Figure 3 — Proposed model
- pretrained YOLO11 backbone
- TP-CDA placement
- axis head
- APCF neck conditioning
- PELT training path
- explicitly distinguish training-only and inference components

### Figure 4 — D00 evidence
Potential multi-panel:
- pretrained transfer coverage
- AP50 vs AP75 gap
- projection-specific error
- patient concentration / leave-one-out effect

### Figure 5 — Mechanism ablation
- baseline, M01, M02, M03, combination
- primary mAP50-95
- targeted secondary metric

### Figure 6 — Robustness
- paired 3-fold baseline/proposed results
- mean/SD
- optionally seed42/43 development confirmation

### Figure 7 — Clinical operating behavior
- fracture PR curve
- fracture FROC
- confidence calibration if retained

### Figure 8 — Explainability and failure cases
- baseline vs proposed detections
- axis prediction overlay
- APCF gate behavior
- TP-CDA residual/feature response
- representative TP/FP/FN

Keep main paper within about 6–8 figures; move extra plots to supplement.

## Main manuscript tables
### Table 1 — Dataset and split
patients, images, objects, represented classes, projection/metadata support.

### Table 2 — Protocol-aware literature comparison
with comparability tier.

### Table 3 — Historical baseline/negative ablation
B0–B3 and A0–C2 compressed to the scientifically relevant rows.

### Table 4 — New mechanism ablation
M01/M02/M03 and combinations.

### Table 5 — Locked model final performance
overall + Clinical-7 + Core-6 + per-class summary + CI.

### Table 6 — Robustness/generalization
3-fold CV, external FracAtlas, optional literature-comparable split.

### Table 7 — Efficiency
params, GFLOPs, model size, T4 batch-1 latency, FPS, peak GPU memory.

Supplement:
- all class APs
- all seed results
- full hyperparameters
- patient sensitivity
- detailed metadata subgroups
- complete uncertainty tables

## Explainability policy
Explainability is supportive evidence, not proof of clinical reasoning.

### Quantitative/intrinsic explanations preferred
1. axis prediction accuracy
2. APCF gate distributions by projection and feature level
3. TP-CDA residual magnitude inside lesion ROIs vs matched background
4. patient exposure diagnostics for PELT

### Qualitative
- Grad-CAM or detector-compatible activation maps baseline vs proposed
- feature/residual maps
- error panels

If Grad-CAM is used, report layer and parameters and do not claim it validates clinical causality.

### Failure taxonomy
At minimum:
- missed subtle fracture
- growth-plate/confound false positive
- localization failure
- rare class failure
- cast/metal/text interference
- projection-dependent failure
