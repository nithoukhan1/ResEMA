# D4-B2C1 Multi-View Novelty Landscape

## Purpose

This document narrows the D4 multi-view design space before equations or implementation.

The evidence entering this gate is:

- D4-B2A: strong paired-view availability;
- D4-B2B: moderate annotation/model-error complementarity;
- 11.48% view-exclusive pair/class annotation support;
- model-error complementarity modest overall and stronger for fracture.

The design must therefore be selective and must respect asymmetric visibility.

## Prior-art families that are NOT acceptable as the novelty core

### 1. Generic AP/LAT feature fusion

Existing fracture/radiograph work already fuses orthogonal projections.

Examples:
- Yang et al., Diagnostics 2024, DOI 10.3390/diagnostics14212425:
  paired AP/lateral scaphoid feature fusion.
- Hendrix et al., European Radiology 2023, DOI 10.1007/s00330-022-09205-4:
  multi-view scaphoid fracture detection using multiple radiographs.
- Hu et al., IEEE TBME 2026, DOI 10.1109/TBME.2026.3728034:
  cross-guided dual-view pre-training for paired orthogonal radiographs.
- Cura et al., 2026:
  dual-view AP/lateral input-level fusion for tibial plateau fracture detection.

Disposition:
`GENERIC_DUAL_STREAM_OR_ALWAYS_ON_FUSION = REJECT_AS_NOVELTY_CORE`

### 2. Generic confidence/uncertainty/conflict gating

Existing multi-view work already adapts fusion using confidence, uncertainty or inter-view disagreement.

Examples:
- CAMVF, MICCAI 2024:
  correlation-adaptive multi-view fusion using confidence and consistency.
- CEI-Net, MICCAI 2026:
  class-conditional disagreement modeling with evidential uncertainty and conflict gating.
- Wang et al., IEEE JBHI 2026, DOI 10.1109/JBHI.2025.3649056:
  uncertainty-driven dynamic weighting for incomplete multi-view data.
- Agreement-aware cross-view refinement and fusion, Knowledge-Based Systems 2026,
  DOI 10.1016/j.knosys.2026.117092:
  fusion explicitly conditioned on inter-view agreement/disagreement in dual-view X-ray imagery.

Disposition:
`UNCERTAINTY_OR_CONFLICT_GATING_ALONE = REJECT_AS_NOVELTY_CORE`

### 3. Generic cross-view / mutual knowledge distillation

Multi-view teacher/student and mutual-distillation methods are established.

Examples:
- Black et al., WACV 2024:
  multi-view classification using hybrid fusion and mutual distillation.
- Multi-view Teacher-Student Network, Neural Networks 2022.
- MVKD-Trans, 2025:
  multi-view knowledge distillation for medical ultrasound classification.

Disposition:
`GENERIC_MULTIVIEW_DISTILLATION = REJECT_AS_NOVELTY_CORE`

### 4. Cross-view geometric correspondence as the primary mechanism

The current GRAZPEDWRI-DX annotations do not provide explicit same-lesion AP/LAT correspondence.
Orthogonal projections are geometrically different.

Self-supervised multi-view X-ray correspondence learning exists (MICCAI 2025), but
introducing synthetic-view correspondence pretraining would substantially expand scope.

Disposition:
`EXPLICIT_CROSS_VIEW_BOX_CORRESPONDENCE = OUT_OF_SCOPE_FOR_FIRST_D4_CANDIDATE`

## Search result specific to GRAZPEDWRI-DX

Targeted searches found many 2025-2026 GRAZPEDWRI-DX detectors based on single-image
YOLO-style feature/backbone/neck/loss modifications.

The dataset itself explicitly contains paired PA/AP and lateral studies.

No directly matching peer-reviewed GRAZPEDWRI-DX object detector was identified in this
search that explicitly supervises per-class AP-only/LAT-only/both visibility and uses
that state to gate residual cross-view assistance.

This is NOT proof of absolute novelty. It is the current search result and must be
described conservatively.

## Design implication from B2B

A safe next mechanism should NOT force agreement between views.

It should distinguish:
- class absent in both;
- class visible/annotated in AP only;
- class visible/annotated in LAT only;
- class visible/annotated in both.

This directly addresses the measured view-exclusive annotation population.
