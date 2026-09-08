# 01 — Project Charter

## Primary objective
Develop a lightweight, reproducible pediatric wrist abnormality detector that improves a strong COCO-pretrained YOLO11s baseline under a patient-disjoint evaluation protocol and yields a technically defensible contribution suitable for a reputable SCIE/JCR Q2–Q3 journal.

## Scientific principles
1. Strong baseline first.
2. One scientific question per experiment.
3. No hidden hyperparameter/model shopping.
4. Patient-disjoint development and test separation.
5. Split-B test remains sealed until method lock.
6. Every full run comes from a committed Git SHA.
7. Every run has a predeclared success/failure rule.
8. Every result is registered even if negative.
9. Large checkpoints live outside Git but are referenced by immutable path + SHA256.
10. Manuscript claims are limited to what the evidence supports.

## Current frozen baseline
- Model: YOLO11s
- Initialization: COCO pretrained
- imgsz: 1024
- batch: 16
- optimizer: SGD
- lr0: 0.01
- lrf: 0.01
- momentum: 0.937
- weight decay: 0.0005
- cosine LR: true
- box: 7.5
- cls: 0.5
- dfl: 1.5
- mosaic: 1.0
- mixup: 0.0
- copy_paste: 0.0
- close_mosaic: 10
- 150 epochs
- deterministic seed control

## Frozen dataset policy
- Split-B V2 remains the development/test protocol.
- Test patients remain sealed.
- Known corrupt validation PNG remains documented; operational validation uses 3,049 readable images / 7,110 objects.
- Foreignbody is N/A on held-out metrics where no positives exist.

## Current historical decisions
- DySample: negative standalone; no reproducible positive incremental effect.
- SC-e50: small/weak positive context only; not sufficient headline novelty.
- SC+DySample: not locked.
- ResEMA: historical weak/negative branch.
- No generic module shopping.

## Reporting standards
The project will be organized so that CLAIM 2024 requirements can be satisfied:
- explicit data sources and partitions;
- patient-level disjointness;
- preprocessing;
- software/hardware versions;
- initialization strategy;
- complete training hyperparameters;
- model selection;
- uncertainty and subgroup analysis;
- software/repository availability.
