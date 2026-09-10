# Decision Log

## 2026-09-08 — Workflow
Adopt local VS Code -> GitHub immutable commit -> Kaggle execution -> artifact registration as the only full-experiment workflow.

## 2026-09-08 — Scientific direction
TP-CDA, APCF and PELT are hypotheses to screen independently, not assumed final components.

## 2026-09-08 — Repository
Continue in existing `nithoukhan1/ResEMA`; create a new publication research branch only after framework approval.

## 2026-09-08 — Test policy
Split-B test remains sealed until final method lock and robustness evaluation.

## 2026-09-08 — Research CI strategy

Use two research-specific CI layers:

1. lightweight research-integrity checks for governance, provenance,
   trackers and artifact-policy validation;

2. model-smoke checks only when model YAML/source files change.

The inherited Ultralytics CI is not treated as the primary research CI.
Feature-branch pull-request usage will be reviewed separately because
the inherited workflow can run on pull requests.

## 2026-09-10 — P0-08 publication text-encoding policy

All publication-governed research text files and research-specific GitHub
workflow YAML files must decode as strict UTF-8.

The research integrity suite enforces this policy.

Legacy Windows-1252 byte `0x97` in three historical Markdown files was
replaced by the intended Unicode em dash, and those targeted files were
normalized to LF-only line endings.

This is a repository-integrity repair. It does not change scientific
results, dataset membership, model configuration, experiment outcomes,
or the sealed Split-B test policy.
