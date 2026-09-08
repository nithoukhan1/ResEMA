# Project Status — 2026-09-08

## Historical architecture lesson
The original C3k2_SC + DySample + ResEMA hypothesis is no longer the final method.

- ResEMA: weak/negative in controlled historical ablations.
- DySample: negative standalone and non-reproducible incremental effect.
- SC-e50: small/weak overall benefit, although Core-6 showed a modest positive signal.
- SC+DySample: seed-42 gain did not reproduce at seed 43.

## Strongest recipe lesson
COCO-pretrained YOLO11s is the robust baseline. Pretraining produced a substantially larger benefit than any old custom module.

## Frozen development baseline
A0/B3:
- YOLO11s
- COCO pretrained
- Split-B V2
- imgsz 1024
- batch 16
- SGD
- cls 0.5
- mixup 0
- 150 epochs
- deterministic seed control

Independent validation:
- seed42 mAP50-95: 0.413305
- seed43 mAP50-95: 0.424578

## New method family
Anatomy-conditioned, transfer-preserving, patient-effective YOLO11:
1. TP-CDA
2. APCF
3. PELT

All three are hypotheses and must pass individual screening.

## Current phase
Phase 0 — repository/provenance consolidation.

## Next phase
D00 no-training diagnostic audit before any new full experiment.

## Test policy
Split-B test remains sealed until method lock and grouped robustness are complete.
