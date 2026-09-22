# 09 - Tracking System

## Active trackers

### `research/CURRENT.md`
Human-readable current phase, completed work, next work and blocked work.

### `research/DECISIONS.md`
Append-only project-level scientific and governance decisions.

### `research/05_experiments/EXPERIMENTS.csv`
One row per active scientific experiment.
Never delete completed or negative rows.

### `research/01_provenance/ARTIFACTS.csv`
One row per externally stored artifact or checkpoint identity.

### `research/05_experiments/records/<EXPERIMENT-ID>/`
Compact per-experiment configuration, runtime, session, metric, artifact and decision records.

## Historical trackers

The following files under `research/02_history/` are frozen historical evidence from earlier project phases:
- `PROJECT_TRACKER.csv`
- `EXPERIMENT_REGISTRY.csv`
- `ARTIFACT_REGISTRY.csv`
- `DECISION_LOG.md`
- `PROGRESS_LOG.md`
- `RESULTS_MASTER.csv`

Do not rewrite historical rows to imitate the active framework.

## Status vocabulary

Use only:
- `NOT_STARTED`
- `IN_PROGRESS`
- `BLOCKED`
- `COMPLETE`
- `REJECTED`
- `SUPERSEDED`
- `LOCKED`

## Update rule

At every meaningful milestone:
- update `CURRENT.md`;
- update `EXPERIMENTS.csv` when experiment state changes;
- update `ARTIFACTS.csv` when an external artifact is created or used;
- append a project-level decision to `DECISIONS.md` when required;
- commit and push compact evidence.

## Chat handoff rule

Future assistance should reconstruct project state from the latest repository current-state file, active registries, decisions and experiment records before relying on conversation memory.
