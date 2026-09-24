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

## 2026-09-23 - TRAIN-01 governed baseline trainer design

TRAIN-01 uses a single experiment-ID-driven launcher plus a repository-installed `GovernedDetectionTrainer` for all six frozen seed-42 baseline conditions.

The trainer class is placed under the installed `ultralytics` package so the Ultralytics-generated DDP subprocess can import the exact governed class independent of notebook working directory.

A source-level audit of the exact fork found that `BaseTrainer` initializes outer rank RNG as `args.seed + 1 + RANK`; therefore configured seed 42 corresponds to outer rank-0 RNG seed 43 during DDP. To preserve the already-frozen INIT-01 seed-42 model-initialization contract without changing rank-specific data/augmentation randomness, model construction is executed in an isolated seed-42 RNG scope and the outer DDP RNG states are restored immediately afterward.

The actual training-time model construction rechecks the INIT-01 target parameter count, state-item count, 493 compatible pretrained transfers and six class-head-specific incompatible tensors.

The dataset firewall is structural: TRAIN-01 creates a runtime dataset YAML containing only `train` and `val`; it contains no `test` key. Dataset discovery prunes test-like paths and verifies only frozen train/validation membership and image/label stem equality.

TRAIN-01 fresh launch remains locked while `EXPERIMENTS.csv` `source_commit` is blank. RESUME-01 must be implemented and remotely closed before the source commit is bound and any model training is authorized.

## 2026-09-23 - RESUME-01 continuation and source-freeze governance

The exact fork's resume pathway was audited before implementation.

Continuation uses the native checkpoint-state pathway:
`YOLO(last.pt).train(trainer=GovernedDetectionTrainer, resume=True)`.

Scientific hyperparameters are not manually redefined during continuation. The prior
checkpoint must contain optimizer, scaler and EMA state, and its completed epoch must
match the last row of `results.csv`.

Resume sessions must use the exact `execution_commit` recorded by the original TRAIN-01
runtime manifest. A merely compatible descendant commit is not accepted for continuation.

Because upstream trainer construction rewrites `args.yaml`, the governed resume launcher
archives the pre-resume `args.yaml` inside the numbered append-only resume-session
governance directory before constructing the trainer.

The original TRAIN-01 `RUNTIME_MANIFEST.json` is immutable during continuation.
RESUME-01 adds numbered `RESUME_PREFLIGHT_NNN.json` and `RESUME_RUNTIME_NNN.json`
records instead of overwriting original provenance.

Source binding uses two commits to avoid circular self-reference:
- implementation commit `S`: contains the complete frozen TRAIN-01/RESUME-01 runtime;
- authorization commit `A`: descendant of `S`, writes `S` into all six
  `EXPERIMENTS.csv::source_commit` fields without modifying guarded scientific/runtime
  paths.

All six baseline experiment rows remain `NOT_STARTED` until actual governed execution,
and test predictions/metrics/error analysis remain forbidden during development.

## 2026-09-23 - RESUME-01 evidence-binding hardening

Before freezing the RESUME-01 implementation, the authorization exception for
`EXPERIMENTS.csv::source_commit` is constrained explicitly: every other matrix field
must remain identical to source commit `S`, and all six current source bindings must
equal `S`.

The preflight manifest SHA256 and runtime data-YAML SHA256 are rechecked before
Ultralytics opens train/validation data and again inside trainer setup. Resume also
cross-checks the persisted `args.yaml`, `results.csv`, and `best.pt` against checkpoint
and frozen-runtime evidence.

These checks close authorization drift and preflight-to-DDP time-of-check/time-of-use
gaps without changing the scientific recipe.

## 2026-09-23 - Frozen source commit S remotely accepted and authorization binding approved

The governed TRAIN-01/RESUME-01 implementation is frozen at:

- `S = 2954eab070368932ae68370532b73295feb3ca2d`;
- parent `6bf9764c8455815d57cd253be8284717d6711962`;
- GitHub Research Integrity run `35853996772` (run #13) completed with
  conclusion `success`;
- integrity job `107158063197` completed with conclusion `success`;
- both the repository-state verification step and the research-integrity/D00/RESUME-01
  test step completed successfully.

The six baseline rows may now be atomically bound to `S`.

Authorization rules:
- every baseline row must remain `NOT_STARTED`;
- every `source_commit` value must equal exactly `S`;
- no other `EXPERIMENTS.csv` field may differ from its value at `S`;
- no guarded scientific/runtime path may change in the authorization commit;
- model training remains locked until the authorization commit is remotely pushed and
  its exact GitHub Research Integrity run succeeds.

## 2026-09-24 - Supersede original S/A for execution after zero-epoch interface failure

The original source/authorization chain was governance-valid but is superseded for
execution after the first governed launch exposed an exact-fork interface defect before
epoch 1.

The corrective implementation verifies construction-time class identity from
`self.data["nc"]`, `model.yaml["nc"]`, and `model.model[-1].nc`, and forbids
direct `model.nc` access inside governed `get_model()`.

All six experiments remain `NOT_STARTED`. Their source bindings are cleared, and a new
remotely CI-verified `S2` followed by a separate remotely CI-verified `A2` is required
before any training.

## 2026-09-24 - Corrected source S2 remotely accepted; A2 authorization approved

The corrected governed TRAIN-01/RESUME-01 source is remotely accepted at:

- `S2 = 46f40838c1c24a8ced77a2b868dfa0f7f1037f9c`;
- GitHub Research Integrity run `35956458084` completed successfully;
- integrity job `107495713588` completed successfully;
- repository-state verification and research-integrity/D00/RESUME-01 tests all passed.

The six baseline rows may now be atomically bound to `S2`.

A2 authorization rules:
- every row remains `NOT_STARTED`;
- every `source_commit` equals exactly `S2`;
- no other `EXPERIMENTS.csv` field changes relative to `S2`;
- no guarded scientific/runtime path changes relative to `S2`;
- training remains locked until `A2` is pushed and its exact GitHub Research Integrity
  workflow succeeds.

## 2026-09-24 - Preserve B-AUG frozen membership; correct only operational readability

A train-only diagnostic established that `DATA01:B-AUG-HIST:v1` still contains the
exact frozen 28,454 image/label members with the frozen membership SHA256
`4799e44ace20ab724974984f4cab60740a3fd8ad5cad3e1570bc895f02eefded`.

Ultralytics deterministically excludes exactly two structurally unreadable PNGs at
runtime, leaving 28,452 operationally readable training images. The dataset is not
repaired, deleted, regenerated or relabeled. The frozen membership remains 28,454.

TRAIN-01 therefore distinguishes membership identity from operational readability for
B-AUG, matching the existing Split-B validation policy that distinguishes 3,050 frozen
members from 3,049 operationally readable images.

Existing A2 B-ORG/A-AUG execution and resume lineages remain valid and unchanged.
B-AUG A2 attempts are noncanonical because they failed before epoch 1 on the
operational-count guard. B-AUG fresh execution is blocked until a successor S3/A3
source/authorization chain is remotely CI-verified.
