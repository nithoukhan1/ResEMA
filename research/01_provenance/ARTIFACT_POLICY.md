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
- downloaded Kaggle ZIP archives;
- datasets;
- large prediction dumps;
- caches.

The local-only external-artifact mirror is:

`artifacts_external/`

It is ignored at repository root and may contain complete scientific run archives without becoming Git content.

## Preserve complete external evidence

For completed governed experiments, preserve the complete native run tree externally when available, including:

- weights;
- `args.yaml`;
- `results.csv`;
- native YOLO curves/plots;
- confusion matrices;
- label/training/validation visualization images;
- governance preflight/runtime manifests;
- numbered resume-session manifests;
- final validation summaries;
- execution logs when available.

Do not delete an earlier saved session merely because a later resumed session completed the run. Earlier sessions are part of the resume lineage.

## Register scientifically used artifacts

For every checkpoint or run artifact used for a scientific claim, record as applicable:
- experiment ID;
- artifact role;
- logical name;
- immutable provider/path/version;
- SHA256;
- file size;
- source Git SHA;
- execution Git SHA when different;
- best epoch;
- primary validation metric;
- registration time;
- status.

For Kaggle run completion, preserve hashes for at least:
- `best.pt`;
- `last.pt`;
- `results.csv`;
- `args.yaml`.

BASELINE-FREEZE-01 additionally creates a full file manifest for every registered downloaded ZIP/session so that native plots and auxiliary evidence can be authenticated without committing the binaries.

## Raw versus derived

Canonical external run archives are read-only evidence.

Any later:
- comparison figure;
- montage;
- cropped visualization;
- reformatted confusion matrix;
- combined convergence plot;
- manuscript graphic;

must be stored as a derived analysis artifact and must not overwrite the canonical raw copy.

## Commit small evidence

Commit small CSV, YAML, JSON, Markdown summaries and selected intentionally curated figures required to reproduce conclusions.

Do not commit native heavy run outputs merely for convenience.
