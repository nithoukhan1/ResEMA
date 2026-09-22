# Current Project State

## Active phase
Baseline refresh before final architecture implementation.

## Completed
- historical V1/V2/V3 reconstruction
- C3k2_SC topology/semantic audit
- DySample reference audit
- ResEMA/EMA reassessment
- Split-B provenance/freeze
- EDA-00
- EDA-01
- EDA-02 to EDA-10
- baseline training-parameter review
- repository preflight and workspace inventory

## Current task
Install and remote-close the active baseline-refresh framework on `research/baseline-refresh`.

## Next
1. Freeze Split A augmented, Split B original, Split B augmented bindings.
2. Freeze official YOLO11s COCO checkpoint and SHA256.
3. Implement reusable baseline trainer.
4. Implement reusable Kaggle resume helper.
5. Train `BASE-B-ORG-PT-S42`.
6. Register result and continue remaining five baselines.
7. Run validation-only baseline diagnostics.
8. Freeze architecture/loss.
9. Implement final method.

## Test policy
Neither Split A test nor Split B test is used for development decisions.
