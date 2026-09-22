# Artifact Policy

## Active registry

The active external-artifact registry is:

`research/01_provenance/ARTIFACTS.csv`

The historical registry under `research/02_history/ARTIFACT_REGISTRY.csv` remains frozen historical evidence.

## Do not commit large model artifacts

Do not commit:
- `best.pt`;
- `last.pt`;
- full Kaggle run directories;
- datasets;
- large prediction dumps;
- caches.

## Register scientifically used artifacts

For every checkpoint or run artifact used for a scientific claim, record as applicable:
- experiment ID;
- artifact role;
- logical name;
- immutable provider/path/version;
- SHA256;
- file size;
- source Git SHA;
- best epoch;
- primary validation metric;
- registration time;
- status.

For Kaggle run completion, also preserve hashes for:
- `best.pt`;
- `last.pt`;
- `results.csv`;
- `args.yaml`.

The complete run directory should be persisted externally when needed for resume or provenance.

## Commit small evidence

Commit small CSV, YAML, JSON, Markdown summaries and selected figures required to reproduce conclusions.
