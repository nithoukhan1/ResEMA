# Protocol-Aware SOTA Comparison Plan V8

## Rule
There is no single reliable GRAZPEDWRI-DX leaderboard because published studies use different splits, task subsets, preprocessing, resolutions, pretraining and augmentation.

## Comparison tiers

### Tier 1 — Direct
Same:
- task/classes
- exact split manifest
- patient-level partition
- evaluation code/metric definition

Only Tier-1 comparisons can support a strong "better than" claim.

### Tier 2 — Partially comparable
Same dataset/task but different patient-level split or training protocol.
Use language:
- "competitive with"
- "contextually compared with"
Do not claim direct superiority.

### Tier 3 — Context only
Examples:
- fracture-only subset
- image-level/random split
- unknown/nonpublic split
- different class set
- externally augmented test logic
These belong in literature context, not a numerical leaderboard claim.

## Required SOTA table columns
- Study / year
- Baseline/model
- Detection task and class set
- Split level
- Split ratio
- Split manifest/public?
- Input resolution
- Pretrained?
- Epochs
- Offline/online augmentation
- Parameters
- GFLOPs
- Inference hardware / latency
- mAP50
- mAP50-95
- External validation
- Comparability tier

## Current literature landmarks to include
- Ju & Cai YOLOv8
- YOLOv9
- Kid-YOLO
- X-YOLO
- FracDet-v11
- MFSD-YOLO
- WFYOLO
- SDG-YOLO
- FCE-YOLO / other full-task studies
- fracture-only studies such as PEYOLO separately labeled

## Wording policy
Preferred:
"Our model achieved X under the released patient-disjoint Split-B protocol and demonstrated a paired improvement of Y over a matched YOLO11s baseline."

Avoid:
"Our model is SOTA because its mAP is numerically higher than papers using different test patients."

If a Tier-1 literature split is later reproduced and won, a stronger claim becomes possible.
