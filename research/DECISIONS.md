# Active Decision Log

Append-only project-level decisions for the active research phase.

## 2026-09-21 - Baselines before final implementation

Fresh controlled YOLO11s baselines and validation-only model diagnostics will be completed before freezing the final architecture or long-tail loss.

## 2026-09-21 - Six baseline conditions

The initial seed-42 matrix contains:
- Split B original: pretrained and scratch;
- Split B augmented: pretrained and scratch;
- Split A augmented: pretrained and scratch.

Split A original is not maintained as an active training condition.

## 2026-09-21 - Kaggle execution model

Kaggle is the training platform.
A single experiment may span multiple Save-Version sessions.
Resume sessions must preserve the same Git SHA and scientific configuration.

## 2026-09-21 - External checkpoints

Model checkpoints remain outside Git and are registered by immutable reference and SHA256.

## 2026-09-21 - Test policy

Split A and Split B test partitions remain outside architecture, loss, epoch-budget and training-recipe selection.
