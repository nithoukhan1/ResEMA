# D4-B2C2 VGRA Novelty Boundary

## Status

`TARGETED SEARCH COMPLETE FOR CANDIDATE FREEZE`

This is not a legal/patent novelty opinion and does not prove absolute absence of prior art.

## Established adjacent work

### Generic paired-view fracture/radiograph fusion

Already established:
- paired AP/lateral scaphoid feature fusion;
- multi-radiograph study-level fracture aggregation;
- recent dual-view orthopedic radiograph models.

Therefore:
`MULTI_VIEW_FUSION_ITSELF_IS_NOT_NOVEL`

### Adaptive/gated multi-view fusion

Already established:
- gated CC/MLO mammography fusion;
- confidence/correlation adaptive multi-view fusion;
- evidential/uncertainty-based conflict gating;
- agreement/disagreement-aware dual-view X-ray fusion.

Therefore:
`GATING_OR_UNCERTAINTY_ALONE_IS_NOT_NOVEL`

### Multi-view distillation

Already established:
- hybrid fusion + mutual distillation;
- teacher/student multi-view learning;
- medical multi-view knowledge distillation.

Therefore:
`MULTIVIEW_DISTILLATION_ALONE_IS_NOT_NOVEL`

### View-specific labels

General multi-view multi-label literature already includes explicit learning of
view-specific labels, including:
- Zhao et al., Applied Soft Computing 2022, DOI 10.1016/j.asoc.2022.109071;
- Zhao et al., IEEE Transactions on Multimedia 2023,
  DOI 10.1109/TMM.2022.3219650.

Therefore:
`VIEW_SPECIFIC_LABELS_AS_A_GENERAL_IDEA_ARE_NOT_NOVEL`

### View-aware radiographic detection

Recent YOLO-style radiographic work also uses auxiliary view classification to create
view-dependent representations.

Therefore:
`AUXILIARY_VIEW_CLASSIFICATION_ALONE_IS_NOT_NOVEL`

## Current targeted-search result

No directly matching peer-reviewed pediatric wrist object detector was identified in
the targeted search that combines all of the following:

1. paired AP/lateral wrist studies;
2. ground-truth-derived per-pathology four-state projection visibility
   (`00`,`10`,`01`,`11`);
3. target-specific neutral/assist/suppress semantics derived from those states;
4. low-rank cross-view semantic residual applied only to the detection classification logits;
5. zero-initialized bounded residual;
6. no cross-view box transfer/alignment;
7. exact single-view fallback.

This is the candidate novelty boundary.

The manuscript must phrase novelty conservatively, for example:

"We investigate a projection-visibility-supervised residual assistance mechanism for
paired pediatric wrist detection..."

Do not write:
"the first multi-view X-ray detector" or "the first view-aware detector."

## Why the four-state target matters

B2B showed legitimate view-exclusive pathology support.

A binary agreement objective can collapse:
- AP-only;
- LAT-only;
- absent-both

into the same "not shared" category.

VGRA preserves these distinct semantics and uses target-specific coefficients so an
AP-only class does not positively leak into the LAT detector and vice versa.

## Why the residual is not generic feature fusion

The companion image does not overwrite the target representation.

Its influence is:
- low rank;
- class conditioned;
- bounded;
- target-local-feature anchored;
- visibility-state controlled;
- classification-only;
- zero initialized.

This is the exact candidate distinction to test experimentally.

## Novelty status

`VGRA_NOVELTY_STATUS = PLAUSIBLE_CANDIDATE / NOT YET EMPIRICALLY VALIDATED`

Publication novelty remains contingent on:
- correct implementation;
- positive controlled ablation;
- meaningful effect size;
- no later-discovered direct collision.
