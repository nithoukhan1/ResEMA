# Progress Log

## 2026-09-08
- V7 execution framework drafted.
- No GitHub branch created yet.
- No new training launched.
- Next action after approval: Phase 0 repository consolidation.

## 2026-09-08 — Phase 0 execution update

- `research-v7-publication` created from frozen commit `5a511729bc21c1b0998b7e6f4a84108fb4e309bf`.
- Publication research workspace created.
- Historical B0-C2 registry and results imported.
- Historical artifact provenance reconciled without guessing missing fields.
- V7/V8 research and publication protocols imported.
- Raw 27-record literature library imported.
- Research-specific GitHub CI workflows created.
- Local research integrity suite passed: 8/8.
- Historical/baseline model smoke suite passed: 4/4.
- Phase 0 repository consolidation completed.
- No new model training was launched.
- Split-B test remains sealed.
- Next scientific phase: D00 diagnostics.

## 2026-09-08 — Phase 0 remote closure

- Publication commit: `57dee3efdaf234f5ac6eaf0950c3f3e8824b56af`
- Remote branch: `origin/research-v7-publication`
- Remote SHA independently verified equal to local SHA.
- Research Integrity GitHub Action: PASS.
- Research Model Smoke GitHub Action: PASS.
- Local Git `origin` changed from HTTPS to SSH because repeated HTTPS DNS/reset failures prevented push.
- SSH authentication verified for GitHub.
- SSH transport uses `ssh.github.com:443` through the local SSH config.
- `upstream` remains unchanged.
- Phase 0 complete.
- Next phase: P1 / D00 diagnostics.
- No new model training performed.
- Split-B test remains sealed.

## 2026-09-10 — P0-08 UTF-8 integrity hardening

- Repaired legacy Windows-1252 byte `0x97` in `DECISION_LOG.md`,
  `P0_PROVENANCE_AUDIT_2026-09-08.md`, and `PROGRESS_LOG.md`.
- Normalized those three targeted Markdown files to LF-only line endings.
- Added a persistent strict-UTF-8 regression test for publication-governed
  research text files and research-specific workflow YAML files.
- Strict UTF-8 audit before this documentation transaction:
  47 governed text files checked, 0 invalid.
- Targeted text normalization audit: PASS.
- Repository verifier with `--allow-dirty`: PASS.
- Research integrity suite: 9/9 PASS.
- `git diff --check`: PASS.
- No model training was launched and no scientific result was changed.
- Split-B test remains sealed.
- Local repair and verification are complete. Git commit, push,
  local/remote SHA equality, and GitHub Research Integrity CI remain
  the P0-08 remote-closure gate.

## 2026-09-10 — P0-08 remote closure

- P0-08 implementation commit:
  `67e3237d1cf1d918635e78680e6760d4da82720e`.
- Commit parent:
  `f4d6b08e5af20451e32d5f8fd5bc9264b3a22004`.
- Push to `origin/research-v7-publication`: PASS.
- Local/remote SHA equality: PASS.
- GitHub Research Integrity workflow run: `34456349763`.
- Workflow head SHA exactly matched the P0-08 implementation commit.
- GitHub Research Integrity status: completed.
- GitHub Research Integrity conclusion: success.
- P0-08 is therefore CLOSED.
- No model training or scientific-result modification occurred during P0-08.
- Split-B test remained sealed.
- Next scientific phase: P1 / D00 diagnostics.

## 2026-09-11 - P1/D00 input provenance freeze

- D00-PRE-A local source inventory: PASS.
- D00-PRE-B controlled provenance freeze construction: PASS.
- D00-PRE-C physical Kaggle-package binding: PASS.
- D00-PRE-D upload-provenance semantic audit: PASS.
- Original `dataset.csv` matched the authoritative frozen SHA256.
- Five authoritative source archives matched expected SHA256 values.
- Frozen Split-B internal file registry passed 37/37.
- Split-B candidate-v2 matched the embedded accepted-candidate audit 26/26.
- Frozen patient/image counts remain 4,264/914/913 and
  14,227/3,050/3,050.
- Foreignbody-positive patients are authoritatively 4/0/0;
  held-out foreignbody AP is N/A.
- Clean physical train/validation/test image memberships exactly match
  their frozen Split-B memberships.
- Physical image/label counts match exactly for train, validation, and test.
- User-reported Kaggle dataset ref:
  `nettokhan/grazpedwri-dx-split-b`.
- Authoritative clean package YAML is `split_B_original.yaml`
  with train=`images/train`.
- Historical `train_aug_historical` is non-authoritative for D00.
- Historical augmentation YAML provenance drift was detected:
  old snapshot SHA `df8cfe5d70b9127dce2713aa02a0166ff7df13c61b4d67eee3f7c0934b687986`;
  current SHA `d5e5336b401f88be6bfb8026e26e573738101462fc21d6b971a3d183819b7ff0`.
- Exact cause/time of the historical YAML change was not reconstructed.
- Kaggle runtime mount path remains unverified and must be checked in Kaggle
  before D00 execution.
- Individual test membership CSVs remain external to the development
  repository.
- No model training, architecture change, test prediction, or test metric
  evaluation occurred.
- Split-B test remains sealed until P10.
- Next scientific checkpoint: D00-A transfer coverage audit.

## 2026-09-12 - P1/D00-A0 runtime binding and D00-A implementation start

- D00-PRE remote closure commit:
  `072dfc47f52a14f3266eae1b66128efc329749bf`.
- GitHub Research Integrity run `34617725501`: PASS.
- D00-A0 Kaggle runtime root:
  `/kaggle/input/datasets/nettokhan/grazpedwri-dx-split-b/GRAZPEDWRI-DX Split B`.
- Five critical package fingerprints matched exactly.
- Eight physical clean/historical folder counts matched exactly.
- D00-A0 runtime binding SHA256:
  `eddffe25ded04c7df598d76769a3dd224a794efe542da040a25dadc3c16f90ef`.
- Clean runtime definition remains `images/train`, `images/val`,
  and sealed `images/test`.
- Historical `train_aug_historical` remains non-authoritative for D00.
- Split-B test label content, predictions, metrics, and errors were not accessed.
- D00-A transfer coverage implementation started.
- Transfer coverage is deterministic and architecture-level; seed duplication is
  not required.
- Eight historical architecture states are registered as a complete
  C3k2_SC ? DySample ? ResEMA factorial matrix.
- Primary transfer metric reproduces exact Ultralytics state-dict key + shape
  intersection.
- Additional same-top-level-module-type coverage is recorded to distinguish
  loader-compatible tensors from stricter architecture-consistent transfer.
- Non-Detect coverage is reported separately to control for the expected
  COCO-80 to GRAZPEDWRI-9 Detect-head mismatch.
- No source checkpoint is committed to Git; runtime checkpoint SHA256 will be
  recorded during D00-A execution.
- Independent D00-A1 logic review distinguished tensor-level unmatched
  evidence from genuine top-level layer aggregation.
- Persistent Research Model Smoke coverage was expanded to all eight registered
  D00-A architectures using the exact YOLO11s `s` scale and 9-class target
  construction path.
- No training or dataset model evaluation occurred.
