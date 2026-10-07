# D4-B2B Multi-View Complementarity Closure

## Status

`COMPLETE / PRESERVED / MODERATE COMPLEMENTARITY`

Source branch:
`research/lpq-method-01`

Source HEAD:
`4963f8e81264cb42d2ef81c06aefe65e4ffe7fe2`

Package SHA256:
`21d738e9b269c791fcc953eeab6fb2c2e1252ca42255d1bd9c4b4e340c49e9aa`

## R1 closure-attempt safe stop

The first closure helper (`D4_B2B_CLOSE_PRESERVE_COMPLEMENTARITY_R1.py`) stopped
during package verification before any artifact copy or Git mutation.

Cause:
the helper incorrectly hard-coded an inferred full-precision value for the mean
class-set Jaccard even though the governed audit had established the displayed
8-decimal value.

R2 corrects verifier precision policy only. Scientific evidence and package contents
are unchanged.

## Evidence scope

Read-only analysis of:
- Split-B validation metadata;
- preserved D2 validation ground truth;
- preserved D2 baseline predictions;
- preserved D2 SCConv-Early predictions.

No raw images or raw label files were opened.
No new inference or training occurred.
Split-B test access remained NONE.

## Pair scope

- exact AP/LAT pairs: 1402
- fully readable pairs for model analysis: 1401
- excluded because one member is the governed unreadable validation image: 1

## Annotation complementarity

- pair-class units with GT in at least one view: 2761
- view-exclusive units: 317
- view-exclusive fraction: 0.11481347
- mean AP/LAT class-set Jaccard: 0.91440799
- exact AP/LAT class-set match fraction: 0.79814551

Interpretation:
The views are predominantly similar but not redundant. About 11.5% of supported
pair-class units are annotated in only one projection.

## Model-error complementarity at confidence 0.25

Baseline:
- shared-GT pair/class units: 2441
- one-view-only TP units: 91
- one-view-only TP fraction: 0.03727980

SCConv-Early:
- shared-GT pair/class units: 2441
- one-view-only TP units: 106
- one-view-only TP fraction: 0.04342483

Fracture:
- baseline one-view-only TP fraction: 0.05929304
- Early one-view-only TP fraction: 0.07411631

## Decision

Predeclared complementarity tier:

`MODERATE`

Multi-view remains a serious method-design candidate.

However, the evidence does not support treating every paired case as requiring heavy
fusion. The overall one-view-only TP fraction is modest, while the fracture class shows
stronger asymmetry.

The next design stage should therefore prefer a selective/conditional cross-view
mechanism over unconditional full-network fusion, subject to prior-art review.

## Critical limits

- no same-lesion correspondence across views is claimed;
- no cross-view geometric box mapping is assumed;
- no architecture is selected;
- implementation and GPU training remain unauthorized.

## Next

`D4-B2C_MULTIVIEW_METHOD_NOVELTY_AND_MATHEMATICAL_DESIGN`
