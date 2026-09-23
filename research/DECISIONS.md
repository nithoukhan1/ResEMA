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

## 2026-09-23 - INIT-01 initialization contract frozen

The official YOLO11s initialization identity and controlled pretrained-versus-scratch contract are accepted for the fresh six-run baseline matrix.

Authoritative runtime evidence:
- report: `INIT01_CHECKPOINT_DISCOVERY_R3_OFFLINE.json`;
- report SHA256: `d233133814a0c1ad145b76a8f33c85a36349c4ff915d632aab3d8af60329e181`;
- exact repository commit restored and verified: `60abdfce6ede6a954ffcd804daa08c224856481a`;
- offline Git bundle SHA256: `77f9f9cd7380606a72a8c1855bcc2bbe3252d168792b8693531cb6edda8d9f83`;
- exact fork runtime: Ultralytics `8.4.7`, Python `3.12.13`, PyTorch `2.10.0+cu128`, Tesla T4 x2;
- no model training, dataset-content access, test-data access or remote repository modification occurred during INIT-01A-R3.

Frozen official checkpoint:
- source: `ultralytics/assets`;
- release: `v8.4.0`;
- asset: `yolo11s.pt`;
- bytes: `19,313,732`;
- SHA256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`;
- checkpoint remains external to Git.

Frozen 9-class initialization contract:
- contract ID: `INIT01:YOLO11S-9C-PTVSCR-S42:v1`;
- target: YOLO11s detect scale `s`, 9 classes, seed 42;
- target parameters: `9,431,275`;
- pretrained and scratch targets have identical architecture and identical seed-42 initial state before checkpoint transfer;
- `493 / 499` state items transfer exactly from the official checkpoint;
- the only six non-transferable target items are the three class-prediction conv weight/bias pairs:
  - `model.23.cv3.0.2.weight`;
  - `model.23.cv3.0.2.bias`;
  - `model.23.cv3.1.2.weight`;
  - `model.23.cv3.1.2.bias`;
  - `model.23.cv3.2.2.weight`;
  - `model.23.cv3.2.2.bias`;
- all six incompatible target tensors remain at their deterministic seed-42 initialization.

The source checkpoint's historical COCO training arguments are provenance only and are not the project baseline training recipe.

The three pretrained baseline rows are bound to checkpoint SHA256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`. The three scratch baseline rows remain checkpoint-free. All six experiments remain `NOT_STARTED`; their `source_commit` values remain blank until the governed training implementation is frozen.

The project advances to `TRAIN-01`. Training remains locked until both `TRAIN-01` and `RESUME-01` are complete and committed.
