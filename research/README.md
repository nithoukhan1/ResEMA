# GRAZPEDWRI-DX Publication Research Workspace

This directory is the authoritative research record for the publication-oriented detection project.

## Current state

The fresh six-run YOLO11s baseline matrix and BASELINE-FREEZE-01 are complete.

Two parallel workstreams are active:

1. `research/baseline-refresh`
   - governed scratch E200 epoch-budget calibration;
   - Split-B test remains sealed.

2. `research/arch-corr-01b`
   - historical C3k2_SC / ResEMA forensic closure;
   - transfer-preserving TPSC candidates;
   - retained DySample;
   - canonical EMA / transfer-preserving TPEMA;
   - implementation verification complete;
   - corrected-module training remains locked until ARCH-CORR-01 closure attestation.

The final architecture and long-tail loss are not yet frozen.

Start here:
1. `CURRENT.md`
2. `DECISIONS.md`
3. `07_method/ARCH_CORR_01_MASTER_TRACKER.md`
4. `07_method/ARCH_CORR_01_MODULE_DISPOSITION.md`
5. `05_experiments/BASELINE_FREEZE_01_CLOSURE.md`
6. `05_experiments/EXPERIMENTS.csv`
7. `01_provenance/ARTIFACTS.csv`

## Workflow

Local VS Code -> Git -> GitHub immutable commit -> governed Kaggle exact commit ->
Save Version / resume if needed -> external checkpoint storage -> compact result
registration in Git -> documented scientific decision.

## Existing history

The V7/V8 governance documents and `02_history/` are preserved as historical
evidence. `07_method/METHOD_BLUEPRINT_V7.md` is explicitly paused and is not the
active architecture specification.
