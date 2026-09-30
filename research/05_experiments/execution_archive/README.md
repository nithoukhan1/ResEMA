# Experiment Execution Archive

This directory preserves the exact Kaggle-side execution wrappers and session provenance used for scientific experiments.

## Design rules

1. Canonical scientific runners remain under `research/runtime/`; they are the source of truth.
2. Files here document how a specific experiment was launched or resumed on Kaggle.
3. Do not duplicate canonical runner implementations into experiment folders.
4. Each archived wrapper is bound to an exact repository commit and canonical runner path.
5. Historical code must be labeled as one of:
   - `EXACT_CAPTURE`
   - `RECONSTRUCTED_FROM_EVIDENCE`
   - `INCOMPLETE_HISTORICAL_RECORD`
6. Planned scripts may be committed before execution, but must remain labeled `PLANNED_NOT_EXECUTED` until actually used.
7. Large checkpoints and raw Kaggle run directories remain outside Git and are referenced by provenance records/hashes.
8. Split-B test access remains forbidden during development/model selection unless a later explicit governance phase authorizes it.

## Per-experiment layout

```
<EXPERIMENT-ID>/
  README.md
  EXECUTION_BINDING.json
  SESSIONS.csv
  kaggle/
    001_FRESH_LAUNCH.py
    002_RESUME_SESSION_01.py
    ...
```

The execution archive is documentation. It does not replace the governed runtime or authorization controls.
