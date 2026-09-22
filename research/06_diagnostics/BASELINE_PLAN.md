# Fresh Baseline Diagnostic Plan

Use fresh governed validation-selected checkpoints only.

Measure:
- P/R/F1/mAP50/mAP50-95
- per-class AP50/AP50-95/recall
- size-conditioned recall/error
- false positives / false negatives
- confidence distributions
- center vs border errors
- head/mid/tail summaries
- convergence and best-epoch trajectory

Use these results to decide:
- whether 100 epochs is sufficient
- exact SC placement
- DySample role
- attention vs NONE
- loss-family shortlist

No test-set selection.
