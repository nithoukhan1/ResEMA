# Active Decision Log

Append-only project-level decisions for the active research phase.

## 2026-09-21 - Baselines before final implementation

Fresh controlled YOLO11s baselines and validation-only model diagnostics will be completed before freezing the final architecture or long-tail loss.

## 2026-09-21 - Six baseline conditions

The initial seed-42 matrix contains:
- Split B original: pretrained and scratch;
- Split B augmented: pretrained and scratch;
- Split A augmented: pretrained and scratch.

Split A original is not maintained as an active training condition.

## 2026-09-21 - Kaggle execution model

Kaggle is the training platform.
A single experiment may span multiple Save-Version sessions.
Resume sessions must preserve the same Git SHA and scientific configuration.

## 2026-09-21 - External checkpoints

Model checkpoints remain outside Git and are registered by immutable reference and SHA256.

## 2026-09-21 - Test policy

Split A and Split B test partitions remain outside architecture, loss, epoch-budget and training-recipe selection.
## 2026-09-22 - Baseline refresh framework remotely closed

The active baseline-refresh framework was committed and pushed on `research/baseline-refresh`.

Closure evidence:
- framework commit: `0a7b6304cc4bd2f064618f873304bd849f82e3c0`;
- local and remote branch heads matched;
- local working tree was clean after push;
- local pre-push validation passed all 33 research tests;
- GitHub Research Integrity workflow run `35687683929` completed successfully;
- the GitHub integrity job passed repository-state verification and research integrity/D00 contract tests.

The active project state advances from framework installation to `DATA-01` dataset binding freeze.

## 2026-09-22 - DATA-01 active dataset bindings accepted for freeze

Read-only Kaggle verification established three active baseline data bindings:

- `DATA01:A-AUG:v1` -> `nithoukhan/grazpedwri-dx-aug`;
- `DATA01:B-ORG:v1` -> runtime-observed `utopianstar/grazpedwri-dx-split-b`;
- `DATA01:B-AUG-HIST:v1` -> the verified `train_aug_historical` condition inside the same Split-B package.

The historical D00 manifest is not rewritten. DATA-01 records the current runtime source separately and preserves D00 as historical provenance.

Split-B original is the primary development condition. Split-B historical augmentation is intentionally retained as a historical offline-augmentation comparator; its mounted membership and copied-label integrity are verified, while the previously documented unresolved historical generation-provenance drift remains explicit.

Split-A augmented is retained for baseline reconstruction and split-robustness reporting only. Its training membership overlaps Split-B train/validation/test by 9,852 / 2,171 / 2,181 images respectively. Therefore Split-A results must not influence Split-B architecture, loss, epoch-budget, training-recipe or final-model selection.

Baseline training remains locked until `INIT-01`, `TRAIN-01` and `RESUME-01` are completed and committed.

## 2026-09-22 - DATA-01 R3 cross-test-label diagnostic superseded and contained

A superseded read-only `DATA-01B-R3` helper compared Split-A training annotation bytes against canonical labels located through Split-B train/validation/test folders. Membership containment shows that 2,181 Split-A training images fall inside the Split-B test membership; therefore the helper would have opened those Split-B test label files for byte-equality checking.

No model predictions, test metrics, annotation semantics, architecture/loss decisions, model selection, training or persistent dataset/repository modification resulted from this diagnostic. Its cross-Split-B label-byte comparison is excluded from authoritative DATA-01 evidence and from all development logic.

`DATA-01B-R4` replaced that operation with a Split-A train-local `orig_` versus `aug_` annotation comparison and opened no Split-B label files. Going forward, pre-final-test Split-B handling is limited to source identity, membership/hash and aggregate split-integrity evidence.

## 2026-09-22 - DATA-01 remotely closed

The active dataset-binding freeze is remotely closed on `research/baseline-refresh`.

Closure evidence:
- DATA-01 commit: `4473ff57126e2427c6a6e6e5f24c40528f76e5f7`;
- local pre-push research validation passed all 40 tests;
- the commit contained exactly the five approved DATA-01 binding files;
- local and remote branch heads matched after push;
- the local working tree was clean after push;
- GitHub Research Integrity workflow run `35711226411` completed successfully on the exact DATA-01 commit;
- the `integrity` job passed repository verification and the research-integrity/D00 contract test step.

The frozen active binding IDs are:
- `DATA01:A-AUG:v1`;
- `DATA01:B-ORG:v1`;
- `DATA01:B-AUG-HIST:v1`.

No model training or model-architecture change occurred during DATA-01. The superseded R3 cross-Split-B label-byte comparison remains excluded from authoritative evidence and development logic.

The project advances to `INIT-01`. Baseline training remains locked until `INIT-01`, `TRAIN-01` and `RESUME-01` are complete and committed.
