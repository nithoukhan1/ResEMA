# BASELINE-FREEZE-01 Protocol

## Status

`PREPARED_FOR_READ_ONLY_INGESTION`

## Purpose

BASELINE-FREEZE-01 closes the six-run YOLO11s baseline refresh without committing heavy model artifacts to Git.

The phase has two parts:

1. **BASELINE-FREEZE-01A — read-only artifact ingestion and audit**
2. **BASELINE-FREEZE-01B — compact repository synchronization after 01A passes**

No new model training, architecture change, loss change, or test-set evaluation is authorized by this protocol.

## Six canonical experiments

- `BASE-B-ORG-PT-S42`
- `BASE-B-ORG-SCR-S42`
- `BASE-B-AUG-PT-S42`
- `BASE-B-AUG-SCR-S42`
- `BASE-A-AUG-PT-S42`
- `BASE-A-AUG-SCR-S42`

All six have reached 100/100 epochs in external Kaggle evidence. Repository experiment rows are intentionally not rewritten until the complete freeze audit is reconciled.

## External artifact rule

Large run outputs remain outside Git.

The local repository may contain a **local-only** archive at:

`artifacts_external/baseline_freeze_01/`

The entire `/artifacts_external/` tree is ignored by Git.

Retain every downloaded Kaggle ZIP, including partial parent sessions used for resume. The baseline history contains 11 ZIP archives for six scientific experiments.

### Local archive layout

```text
artifacts_external/
└── baseline_freeze_01/
    ├── raw_zips/
    ├── sessions/
    │   └── <ZIP-STEM>/
    │       ├── extracted/
    │       └── file_manifest.csv
    ├── manifests/
    │   ├── ZIP_ARCHIVES.csv
    │   ├── SESSION_SUMMARY.csv
    │   ├── CANONICAL_FINAL_SUMMARY.csv
    │   └── SHA256SUMS.txt
    └── freeze_report/
        └── BASELINE_FREEZE_01_LOCAL_INGEST.json
```

The raw ZIPs and extracted trees are immutable evidence after successful ingestion.

## Session-chain authority

The registered session chain is stored in:

`research/05_experiments/BASELINE_FREEZE_01_INPUTS.json`

Canonical final sessions are:

- `BASE-B-ORG-SCR-S42-1.zip`
- `BASE-B-ORG-PT-S42-2.zip`
- `BASE-A-AUG-PT-S42-2.zip`
- `BASE-A-AUG-SCR-S42-2.zip`
- `BASE-B-AUG-PT-S42-2.zip`
- `BASE-B-AUG-SCR-S42-2.zip`

All other registered ZIPs are preserved as resume-parent/session evidence and must not be deleted merely because a later complete session exists.

## BASELINE-FREEZE-01A requirements

For every registered ZIP/session:

- record ZIP filename;
- record provider reference;
- SHA256 the ZIP;
- record ZIP byte size;
- inventory every member;
- safely extract to the ignored local archive;
- SHA256 every extracted file;
- preserve complete native YOLO output;
- locate the experiment run directory;
- verify required run artifacts when present.

For each canonical final run establish:

- experiment ID;
- expected source and execution lineage;
- data binding;
- initialization;
- completed epochs;
- final epoch;
- best epoch from `results.csv`;
- best-row precision;
- best-row recall;
- derived F1 = `2PR/(P+R)`;
- best-row mAP50;
- best-row mAP50-95;
- `best.pt` SHA256;
- `last.pt` SHA256;
- `results.csv` SHA256;
- `args.yaml` SHA256;
- governance preflight/runtime presence;
- resume-session inventory;
- test-firewall evidence;
- full external archive/file manifest;
- canonical Kaggle dataset reference.

The local ingest script does not use Split-A test or Split-B test data and does not generate model predictions.

## Per-class metrics and visualization

Native YOLO plots and images are retained in the local archive, including where present:

- `BoxF1_curve.png`
- `BoxPR_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `labels.jpg`
- `results.png`
- training-batch images
- validation label/prediction images

These are archival evidence, not the repository source of numeric claims.

Per-class AP/recall/support must be frozen from machine-readable or standardized validation-only evidence. If the canonical run output does not contain machine-readable per-class values, a separate governed **validation-only** pass over the frozen `best.pt` may be registered. It must use the corresponding validation split only and must not access any test partition.

## Convergence audit

The freeze report must retain epoch-by-epoch `results.csv` evidence.

Special review is required for any experiment whose best epoch is near the 100-epoch boundary. In particular, `BASE-B-ORG-SCR-S42` previously recorded its best validation mAP50-95 at epoch 100; BASELINE-FREEZE-01 must inspect the late trajectory before deciding whether a separately registered longer-budget experiment is scientifically necessary.

No completed run is silently extended.

## Split-A firewall

Split-A augmented experiments are reporting/reconstruction comparators.

Because Split-A training overlaps Split-B validation/test membership, Split-A outcomes must not drive Split-B architecture, loss, training-recipe, epoch-budget, or final-model selection.

## Test firewall

Split-B test predictions, metrics, annotation-level error analysis, and test-driven decisions remain forbidden during BASELINE-FREEZE-01.

## BASELINE-FREEZE-01B repository synchronization

Only after 01A passes:

- update `research/05_experiments/EXPERIMENTS.csv`;
- add six compact JSON experiment records under `research/05_experiments/records/`;
- register scientifically used external artifacts in `research/01_provenance/ARTIFACTS.csv`;
- update `research/CURRENT.md`;
- append the freeze decision to `research/DECISIONS.md`;
- commit only small CSV/JSON/Markdown evidence;
- never commit checkpoints, full run folders, ZIP archives, or native YOLO image collections;
- push and verify Research Integrity CI on the exact commit.

## Freeze acceptance

BASELINE-FREEZE-01 is complete only when:

- all 11 registered session ZIPs are accounted for;
- all six canonical final runs pass the 100-epoch/provenance/firewall audit;
- final checkpoint/result hashes are recorded;
- external artifact references are recorded;
- repository metadata matches external experimental reality;
- GitHub CI succeeds on the freeze commit;
- Split-B test remains sealed.
