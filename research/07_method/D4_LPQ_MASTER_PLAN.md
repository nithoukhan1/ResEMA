# D4 LPQ Master Plan

## Goal

Develop one problem-driven detector improvement for GRAZPEDWRI-DX based on the
residual error demonstrated by D3, without returning to uncontrolled module shopping.

## Governing hypothesis

SCConv-Early supplies useful feature refinement and false-positive suppression but does
not provide a robust net increase in lesion recovery. The next method should improve
prediction-quality consistency and duplicate competition while preserving useful
SCConv-Early behavior.

## Phase map

### D4-A — transition/context freeze
Documentation and provenance only.
No model-code changes and no training.

### D4-B — mathematical specification + novelty gate
Freeze:
- DAR inputs, targets and reliability formulation;
- GDS grouping, primary-candidate definition and score-separation formulation;
- how DAR/GDS interact with standard YOLO11 classification/box/DFL paths;
- loss terms and coefficients;
- inference behavior;
- exact ablation matrix;
- promotion criteria;
- novelty relationship to GFL, VarifocalNet, TOOD, one-to-one/end-to-end YOLO,
  uncertainty-aware localization, and recent wrist-X-ray detectors.

### D4-C — implementation
Implement LPQ additively.
Stock YOLO11 behavior must remain recoverable unchanged.
No training during implementation.

### D4-D — structural/synthetic verification
Required:
- parser/import tests;
- forward shapes;
- backward gradients;
- finite loss;
- checkpoint save/load;
- pretrained transfer;
- baseline-regression protection;
- deterministic DAR synthetic tests;
- deterministic GDS duplicate-group tests.

### D4-E — source freeze + experiment contract
Freeze exact scientific source and exact first-run training protocol.
Training requires separate authorization after this gate.

### D4-F — first primary training
Primary candidate:
`YOLO11s + SCConv-Early + LPQ`

Use B-TRAIN only for fitting and B-VAL only for development evaluation.
B-TEST remains NONE.

### D4-G — residual diagnostic
Evaluate not only AP/mAP but:
- TP/FN preservation;
- duplicate/background/localization FP;
- fixed-threshold behavior;
- per-class behavior;
- confidence behavior;
- same-GT transitions;
- patient-level changes;
- efficiency.

### D4-H — attribution ablations
Only if the full candidate is promising.
Expected minimum scientific controls:
- baseline reference;
- Early reference;
- LPQ-only if architecture permits clean isolation;
- Early + DAR;
- Early + GDS;
- Early + full LPQ;
- standard one-to-one/end-to-end duplicate-control reference if technically comparable.

### D4-I — optional attention gate
Attention is not pre-authorized.
Only if post-LPQ diagnostics demonstrate a residual feature-representation problem.
Choose one mechanism after a novelty/fit review, then isolate it by ablation.

### D4-J — final architecture freeze
Freeze all architecture, loss, training and inference choices before final-confirmation work.

### D5 — publication-grade robustness
Multiseed, uncertainty, efficiency, convergence and broader robustness.

### D6 — comparator preparation
Train/freeze selected reproducible SOTA/control methods under the same Split-B protocol.

### D7 — final-test contract
Pre-register checkpoints, hashes, models, metrics and evaluator before test access.

### D8 — one sealed-test transaction
Evaluate the pre-registered model set on B-TEST once.

### D9 — manuscript evidence package
Same-test fair comparison + separate cross-paper contextual comparison.

## Permanent firewalls

- no B-test access during D4/D5/D6;
- no architecture changes after final architecture freeze;
- no random module shopping;
- no claim of LPQ novelty until D4-B novelty review closes;
- no training before source freeze + explicit authorization;
- no promotion from one aggregate metric alone.
