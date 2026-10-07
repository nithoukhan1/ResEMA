# D4-B2C1 Candidate Shortlist

## Evidence-backed design requirements

The preferred candidate must:

1. keep per-view localization independent;
2. avoid assuming AP/LAT pixel or box correspondence;
3. preserve strong single-view behavior;
4. use the paired view only as residual assistance;
5. explicitly protect legitimate view-exclusive findings;
6. support missing-view fallback;
7. add modest parameter/inference overhead;
8. target classification/TP preservation rather than forcing box transfer.

## Candidate A — Always-on dual-view fusion

Architecture:
two shared-weight branches -> concatenate/cross-attend -> detection.

Pros:
simple; likely to exploit paired information.

Cons:
poor fit to MODERATE complementarity; can contaminate already-correct cases; highly
crowded prior art.

Disposition:
`REJECT`

## Candidate B — Uncertainty/conflict-gated fusion

Architecture:
per-view predictions -> uncertainty/disagreement -> dynamic fusion.

Pros:
selective; handles weak views.

Cons:
direct collision with recent multi-view medical and X-ray literature, especially CEI-Net
and agreement-aware dual-view fusion.

Disposition:
`REJECT_AS_NOVELTY_CORE`

## Candidate C — Cross-view mutual distillation

Architecture:
paired-view teacher/fusion branch -> view-specific students.

Pros:
training-time transfer and potentially low inference overhead.

Cons:
generic multi-view distillation is well-established; novelty weak without a highly
specific new objective.

Disposition:
`CONTROL_OR_FUTURE_VARIANT_ONLY`

## Candidate D — Visibility-State-Gated Cross-View Residual Assistance (working name: VGRA)

### Core idea

For each class in a paired wrist study, explicitly supervise one of four projection-visibility states:

- `00`: absent from both AP and LAT annotations;
- `10`: AP-only;
- `01`: LAT-only;
- `11`: present in both.

A lightweight pair-context branch predicts this four-state distribution per class.

Cross-view feature assistance is then allowed only as a residual classification-path
update when the learned state supports shared visibility.

The core safety rule is:

`DO NOT FORCE THE ABSENT/VIEW-EXCLUSIVE PROJECTION TO DETECT A CLASS SIMPLY BECAUSE THE OTHER VIEW CONTAINS IT.`

### Proposed high-level structure

AP image -> shared YOLO11s/Early encoder -> AP neck -> AP detection path
                                                   |
LAT image -> shared YOLO11s/Early encoder -> LAT neck -> LAT detection path
                                                   |
               pooled pair-level semantic context
                              |
                    class-wise 4-state visibility
                              |
                    visibility-controlled residual
                         /                 \
                to AP class path       to LAT class path

Bounding-box regression remains view-specific and receives no cross-view box geometry.

### Why it fits D4 evidence

- B2A: paired studies are common enough for deployment/training.
- B2B: 11.48% view-exclusive annotation support makes naive agreement unsafe.
- B2B: one-view-only TP errors exist, especially for fracture, so selective assistance
  has a plausible rescue target.
- D3: the main unresolved problem is TP preservation without losing useful FP suppression.

### Distinction from rejected approaches

VGRA is NOT:
- generic concatenation;
- score averaging;
- confidence-times-IoU;
- uncertainty-only fusion;
- JS-divergence conflict gating;
- mutual distillation;
- box correspondence/alignment.

The candidate novelty hypothesis is:

`explicitly supervised class-wise projection visibility controls whether cross-view
semantic information is allowed to modify each view's detection classification features.`

This remains a HYPOTHESIS, not a novelty claim.

### Main risks

1. visibility-state prediction may be wrong;
2. class-level context may be too weak to rescue subtle lesions;
3. the pair branch may overfit frequent classes;
4. auxiliary visibility supervision may improve classification but not object localization;
5. inference requires both views for assisted mode;
6. missing-view behavior must reduce exactly to the original per-view detector.

## Candidate disposition

`VGRA = PROMOTE_TO_D4-B2C2_MATHEMATICAL_DESIGN_CANDIDATE`

This does not authorize implementation or training.

## Required D4-B2C2 work

Before implementation:
- define exact input feature levels;
- define pair-context representation;
- define 4-state per-class target construction;
- define gating equation;
- define residual injection location;
- define stop-gradient policy if any;
- define loss;
- define missing-view behavior;
- define parameter budget;
- define ablations;
- define promotion criteria;
- perform one final targeted novelty search around supervised projection visibility.
