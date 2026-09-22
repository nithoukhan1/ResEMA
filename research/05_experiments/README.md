# Experiments

This folder contains the active governed experiment record.

## Main files

- `EXPERIMENTS.csv` — one row per scientific experiment.
- `BASELINE_PROTOCOL.md` — current six-run baseline study.
- `TRAINING.yaml` — shared baseline training contract.
- `records/` — compact per-experiment records.

## Per-experiment records

Each experiment will use:

`records/<EXPERIMENT-ID>/`

Typical committed files:

- `README.md`
- `config.yaml`
- `runtime_manifest.json`
- `sessions.csv`
- `epoch_metrics.csv`
- `validation_summary.yaml`
- `per_class.csv`
- `artifact_manifest.json`
- `decision.md`

Large checkpoint files such as `best.pt` and `last.pt` stay outside Git
and are registered in `research/01_provenance/ARTIFACTS.csv`.

Kaggle raw output directories are execution artifacts, not Git research records.
