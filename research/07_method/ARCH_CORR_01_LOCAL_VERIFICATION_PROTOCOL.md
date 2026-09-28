# ARCH-CORR-01 — Local Verification Protocol

## Purpose

Reproduce the architecture contracts on a separate local Git worktree without
switching or contaminating the canonical baseline/E200 workspace.

## Required topology

Main workspace:
`ResEMA` -> `research/baseline-refresh`

Architecture worktree:
`ResEMA-ARCH-CORR` -> exact `research/arch-corr-01b` verification head

## Inputs

- exact Git verification head;
- official locked `yolo11s.pt`;
- checkpoint SHA256:
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.

No dataset is required.
No training is performed.
No test split is accessed.

## Local gates

1. clean architecture worktree;
2. Python imports exact worktree Ultralytics source;
3. focused TPSC, EMA and DySample tests pass;
4. TPSC transfer audit passes with locked checkpoint;
5. EMA transfer audit passes with locked checkpoint;
6. all audit JSON files are hashed;
7. a master local verification JSON records Git head, runtime, checkpoint identity and artifact hashes.

Local evidence must be preserved before the final ARCH-CORR-01 closure commit.
