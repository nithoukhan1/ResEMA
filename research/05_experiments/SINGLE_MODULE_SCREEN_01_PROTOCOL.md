# SINGLE-MODULE-SCREEN-01 — Controlled Corrected-Module Ablation

## Status

`REGISTERED — TRAINING NOT AUTHORIZED`

Branch:

`research/single-module-screen-01`

Architecture parent closure:

`dbeb2f7efdf080c0c15b72c0f89fe1f38b84e3b3`

The architecture audit is closed. This phase does not redesign the modules.
It evaluates the already technically verified candidates under one controlled
development condition.

## Scientific question

Which individually corrected architecture modules improve the frozen pretrained
YOLO11s reference under the controlled Split-B Original development condition?

## Human-facing naming

To keep experiment names clear:

- `C3k2_TPSC` -> **SCConv-Early**
- `C3k2_TPSCG4` -> **SCConv-4Stage**
- `DySample` -> **DySample**
- `C3k2_TPEMA` -> **Canonical EMA**

The internal Python class names remain unchanged because those implementations
have already passed technical verification.

## Frozen screen condition

All candidate runs use:

- dataset: Split-B Original;
- data binding: `DATA01:B-ORG:v1`;
- initialization: official pretrained `yolo11s.pt`;
- seed: 42;
- epochs: 100;
- image size: 1024;
- global batch: 16;
- optimizer: SGD;
- cosine learning-rate schedule;
- deterministic execution;
- frozen baseline augmentation/loss/training recipe;
- validation-only model selection;
- Split-B test access: NONE.

## Frozen reference

Reference experiment:

`BASE-B-ORG-PT-S42`

Canonical training-time selection metrics:

- best epoch: 44;
- precision: 0.68900;
- recall: 0.61491;
- F1: 0.6498500510004525;
- mAP50: 0.65301;
- mAP50-95: 0.41704.

Standardized checkpoint revalidation is preserved separately and must not be
silently substituted for the canonical training-time selection metric.

## Registered candidates

### 1. YOLO11s + SCConv-Early

Experiment:

`BORG-PT-S42-SCCONV-EARLY-E100`

YAML:

`ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml`

Expected parameters:

`9,570,093`

Added parameters over baseline:

`138,818`

### 2. YOLO11s + SCConv-4Stage

Experiment:

`BORG-PT-S42-SCCONV-4STAGE-E100`

YAML:

`ultralytics/cfg/models/11/yolo11s-tpsc-g4-v1.yaml`

Expected parameters:

`10,021,679`

Added parameters over baseline:

`590,404`

### 3. YOLO11s + DySample

Experiment:

`BORG-PT-S42-DYSAMPLE-E100`

YAML:

`ultralytics/cfg/models/11/yolo11s-dysample-v2.yaml`

Expected parameters:

`9,455,915`

Added parameters over baseline:

`24,640`

DySample's retained implementation has passed focused technical verification.
Its exact pretrained-transfer behavior will be checked again by the governed
screen preflight before authorization.

### 4. YOLO11s + Canonical EMA

Experiment:

`BORG-PT-S42-CANONICAL-EMA-E100`

YAML:

`ultralytics/cfg/models/11/yolo11s-tpema-head-v1.yaml`

Expected parameters:

`9,435,423`

Added parameters over baseline:

`4,148`

## Selection rule

The primary comparison metric is validation mAP50-95.

Secondary evidence includes:

- validation mAP50;
- precision;
- recall;
- F1;
- per-class metrics;
- convergence behavior;
- parameter count;
- FLOPs;
- latency;
- GPU memory.

No module advances solely because it performs well on a different split or
initialization regime.

## Combination firewall

No module combination is authorized during SINGLE-MODULE-SCREEN-01.

Combination experiments may be designed only after all registered single-module
validation evidence is complete and reviewed.

## Later six-condition study

After a final proposed architecture is selected and frozen, that exact architecture
will be evaluated under the same six baseline conditions:

- Split-B Original pretrained;
- Split-B Original scratch;
- Split-B Augmented pretrained;
- Split-B Augmented scratch;
- Split-A Augmented pretrained;
- Split-A Augmented scratch.

This later robustness matrix does not alter the current architecture-selection
condition.

## Split-A restriction

Split-A may be used later for reporting/robustness comparison, but it may not drive
Split-B architecture selection because its training population overlaps Split-B
partitions.

## Test firewall

Split-B test remains sealed.

No test predictions, metrics, threshold tuning, error analysis or model selection
are permitted during this screen.

## Authorization state

Experiment registration does not authorize GPU training.

Before training:

1. dedicated governed screen runner must be implemented;
2. model/pretrained-transfer preflight must pass;
3. static tests must pass;
4. scientific source must be frozen in an immutable Git commit;
5. a later authorization record must bind that source commit;
6. Kaggle runtime preflight must pass.

Current state:

`TRAINING_AUTHORIZED=FALSE`
