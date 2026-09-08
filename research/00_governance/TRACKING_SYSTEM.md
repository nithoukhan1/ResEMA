# 09 — Tracking System

## Authoritative trackers
### PROJECT_TRACKER.csv
One row per phase/work package.

### EXPERIMENT_REGISTRY.csv
One row per scientific run.
Never delete historical rows.

### ARTIFACT_REGISTRY.csv
One row per external artifact/checkpoint set.

### DECISION_LOG.md
Append-only chronological decisions.

### PROGRESS_LOG.md
Short dated work diary.

### RESULTS_MASTER.csv
Canonical machine-readable table used for paper tables/plots.

## Status vocabulary
Use only:
- NOT_STARTED
- IN_PROGRESS
- BLOCKED
- COMPLETE
- REJECTED
- SUPERSEDED
- LOCKED

## Weekly checkpoint
At least once per week:
- update project tracker
- update experiment registry
- update decision log
- commit/push
- write `WEEKLY_STATUS_YYYY-MM-DD.md`

## Chat handoff rule
Future assistance should work from the latest repository tracker, decision log and experiment registry whenever available, rather than conversation memory alone.
