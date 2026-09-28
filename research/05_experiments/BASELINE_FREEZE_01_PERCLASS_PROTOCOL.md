# BASELINE-FREEZE-01 Standardized Validation-Only Per-Class Protocol

## Status

`READY_FOR_EXECUTION`

## Purpose

Produce one standardized, machine-readable validation-only record for all six frozen
baseline `best.pt` checkpoints.

This closes the remaining BASELINE-FREEZE-01A evidence asymmetry before final
repository synchronization.

## Important distinction

Two different metric families are intentionally preserved:

1. **selection metrics**
   - taken from the epoch row in the canonical training `results.csv` that maximizes
     validation mAP50-95;
   - already frozen by BASELINE-FREEZE-01A.

2. **checkpoint revalidation metrics**
   - produced here by a new standardized validation-only pass over the frozen
     `best.pt` checkpoint;
   - used for per-class metrics, support counts, and standardized qualitative plots.

Checkpoint revalidation does **not** replace the selection metrics.

Small numerical differences between the training-time selected row and the new
checkpoint revalidation are allowed and must be preserved rather than hidden.

## Runtime identity

Each experiment is evaluated with the exact historical execution checkout associated
with its canonical training lineage:

- A2/S2 group: execution commit
  `9fe475175d3963a083d7afc29426f07d86c1887d`
- A3/S3 group: execution commit
  `fe95e2d51c4d545111ae2fa7be70c8e1c8e77487`

The orchestration/evaluation tool itself is copied from the current
`research/baseline-refresh` governance commit and then executed in child Python
processes whose import path is bound to the exact A2 or A3 checkout.

## Standard validation settings

- split: `val`
- imgsz: `1024`
- batch: `16`
- device: `0` for the standardized pass
- expected available GPUs: two Tesla T4
- workers: `4`
- rect: `false`
- conf: Ultralytics validation default (`None -> 0.001`)
- IoU/NMS threshold: `0.7`
- max_det: `300`
- half: `false`
- plots: `true`
- save_json: `false`
- seed: `42`
- deterministic: `true`

A single GPU is used intentionally so every checkpoint is evaluated under one common
validation execution shape. These standardized results are not a deployment-latency
benchmark.

## Dataset discovery and firewall

The evaluator reuses the exact DATA-01 membership logic from the corresponding
historical checkout.

It discovers train/validation membership by count + frozen membership hash while
pruning any test-like directory from discovery.

The runtime YAML contains only:

- `train`
- `val`
- `nc`
- `names`

It contains no `test` key.

Model inference is run only with `split=val`.

## Required outputs per experiment

- `PER_CLASS_METRICS.csv`
- `VALIDATION_ONLY_SUMMARY.json`
- `VALIDATION_ARTIFACT_MANIFEST.csv`
- native Ultralytics validation plots (where generated)

Per-class rows cover all nine configured classes.

For a class with zero validation support:

- support_images = 0
- support_instances = 0
- status = `NO_VALIDATION_SUPPORT`
- precision/recall/F1/AP50/AP50-95 = null

This is especially relevant to Split-B `foreignbody`, which has no frozen validation
support and must not be reported as AP=0.

## Aggregate outputs

Record:

- mean precision
- mean recall
- F1 derived from aggregate P/R
- mean class F1
- mAP50
- mAP50-95
- evaluation class count
- expected frozen validation membership
- expected operational readable validation count

## Test firewall

Forbidden:

- `split=test`
- test predictions
- test metrics
- test annotation/content analysis
- architecture/loss/epoch/model selection from test outcomes

## Acceptance

The standardized per-class freeze passes when:

- all six exact `best.pt` SHA256 values are matched exactly once;
- all six experiments use the expected execution commit;
- all six DATA-01 validation bindings are verified;
- all six passes complete;
- aggregate and per-class records are written;
- no test split is accessed;
- no training occurs.

After review, BASELINE-FREEZE-01B may synchronize the final repository experiment
ledger.
