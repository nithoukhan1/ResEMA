# Phase-0 Provenance Audit — 2026-09-08

## Repository state

Publication research branch:
research-v7-publication

Frozen base commit:
5a511729bc21c1b0998b7e6f4a84108fb4e309bf

## Completed

- publication branch created from frozen code base
- research directory structure created
- V7 execution framework imported
- V8 final research protocol imported
- raw GRAZPEDWRI literature library imported
- B0-C2 experiment registry imported
- results master created
- historical checkpoint registry imported
- A0/B3 seed42 checkpoint provenance added
- known B2 checkpoint hash preserved
- unresolved historical artifact gaps explicitly registered

## Remaining Phase-0 work

P0-05:
- create research-specific CI/tests
- verify research workspace
- make first publication-research commit
- push publication branch
- verify GitHub state

## Scientific state

No new model has been trained.

Split-B test remains sealed.

Next scientific phase after Phase 0:
D00 diagnostic audit.

## Phase 0 closure

P0-05 subsequently completed on 2026-09-08.

Verification:
- research integrity: 8/8 passed
- model smoke tests: 4/4 passed

Phase 0 is therefore complete.

Next phase:
P1 / D00 diagnostics.

No new scientific training occurred during Phase 0.
Split-B test remains sealed.

## P0-08 — publication text-encoding hardening (2026-09-10)

A post-closure repository-integrity audit detected legacy Windows-1252
byte `0x97` in three publication-governed historical Markdown files.

Corrective action:

- replace the three legacy `0x97` bytes with the intended Unicode em dash;
- normalize only the affected Markdown files to LF-only line endings;
- add a persistent strict-UTF-8 research-integrity regression test;
- make no scientific, model, dataset-membership, or experiment-result change.

Local verification evidence before the documentation append:

- strict UTF-8 audit: 47 governed text files checked, 0 invalid;
- targeted text normalization: PASS;
- repository-state verifier with `--allow-dirty`: PASS;
- research integrity suite: 9/9 PASS;
- `git diff --check`: PASS.

P0-08 remote closure requires:

- commit of the reviewed P0-08 scope;
- push to `origin/research-v7-publication`;
- equality of local and remote commit SHA;
- PASS of the GitHub Research Integrity workflow for that commit.

Split-B test remains sealed.
No scientific training occurred during P0-08.
