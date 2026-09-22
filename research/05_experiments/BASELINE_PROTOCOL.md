# Baseline Reverification Protocol

## Six seed-42 runs
1. `BASE-B-ORG-PT-S42`
2. `BASE-B-ORG-SCR-S42`
3. `BASE-B-AUG-PT-S42`
4. `BASE-B-AUG-SCR-S42`
5. `BASE-A-AUG-PT-S42`
6. `BASE-A-AUG-SCR-S42`

## Comparisons
- Transfer: pretrained vs scratch within the same condition.
- Split-B augmented-recipe effect: augmented vs original under the same initialization.
- Split robustness: independently trained Split A augmented vs Split B augmented.

Equal epochs do not equal equal image exposures for original vs augmented data. If augmentation appears beneficial, use an exposure-matched control before attributing the gain to augmentation content.

## Selection
Validation only.
Primary metric: validation mAP50-95.
Report P, R, F1, mAP50, mAP50-95 and per-class metrics.

## Epoch calibration
Initial budget: 100 epochs.
If the best epoch is near the boundary and validation mAP50-95 is still materially rising, register an explicit longer-budget experiment rather than silently extending the completed run.
