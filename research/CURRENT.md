# Current Project State

## Active branch

`research/baseline-refresh`

## Active phase

Baseline data and initialization freeze before fresh YOLO11s reverification.

## Framework closure

The baseline-refresh research framework is remotely closed and CI-verified.

- framework commit: `0a7b6304cc4bd2f064618f873304bd849f82e3c0`
- framework closure bookkeeping commit: `76980ecd05416361cd4f133f869981d3af1bef8b`
- GitHub Research Integrity workflow: success
- local framework validation before push: 33/33 research tests passed
- model source changed during framework setup: no

## Completed

- historical V1/V2/V3 reconstruction
- historical implementation audit
- C3k2_SC topology/semantic defect confirmation
- DySample reference audit
- ResEMA/EMA reassessment
- Split-B provenance/freeze
- EDA-00
- EDA-01
- EDA-02 to EDA-10
- baseline training-parameter review
- six-run baseline design
- Kaggle Save-Version/resume governance
- active experiment/artifact tracking framework
- baseline-refresh branch remote closure
- Research Integrity CI activation for `research/**`
- DATA-01A read-only runtime discovery
- DATA-01B active binding verification for Split A augmented, Split B original and Split B historical augmented
- DATA-01 active dataset-binding freeze remotely closed and CI-verified
- INIT-01 official YOLO11s initialization identity and pretrained-versus-scratch transfer contract frozen
- TRAIN-01 governed baseline trainer remotely closed and CI-verified
- RESUME-01A exact-fork resume source/interface audit passed

## DATA-01 remote closure

DATA-01 is complete and remotely CI-verified.

- DATA-01 commit: `4473ff57126e2427c6a6e6e5f24c40528f76e5f7`
- GitHub Research Integrity workflow run: `35711226411`
- workflow status: completed
- workflow conclusion: success
- integrity job: success
- local pre-push validation: 40/40 research tests passed
- local/remote branch heads matched after push
- local working tree was clean after push
- model training during DATA-01: no
- model architecture change during DATA-01: no
- test model outcomes accessed during DATA-01: no

Frozen active bindings:

- `DATA01:A-AUG:v1`
- `DATA01:B-ORG:v1`
- `DATA01:B-AUG-HIST:v1`

## Current task

`DATA-01C / TRAIN-01D` - record the Split-B historical-augmentation operational-readability correction (28,454 frozen members; 28,452 operationally readable), then create and remotely close successor `S3 -> A3` for future B-AUG fresh execution. Existing A2 lineages continue/resume on exact A2.

INIT-01 is complete at the evidence/freezing level.

Frozen initialization contract:

- contract ID: `INIT01:YOLO11S-9C-PTVSCR-S42:v1`
- official asset: `ultralytics/assets` release `v8.4.0` / `yolo11s.pt`
- checkpoint bytes: `19,313,732`
- checkpoint SHA256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`
- exact repository parent: `60abdfce6ede6a954ffcd804daa08c224856481a`
- Ultralytics runtime: `8.4.7`
- target: YOLO11s detect, scale `s`, 9 classes, seed 42
- target parameters: `9,431,275`
- transferable state items: `493 / 499`
- non-transferable state items: `6 / 499`, limited to the three class-prediction conv weight/bias pairs
- pretrained and scratch targets share the same architecture and the same seed-42 initial state before transfer
- all 493 compatible tensors loaded exactly
- all 6 incompatible target tensors remained at their seed-42 initialization
- INIT-01 evidence report SHA256: `d233133814a0c1ad145b76a8f33c85a36349c4ff915d632aab3d8af60329e181`

The three pretrained baseline rows are now bound to the frozen checkpoint SHA256.
The three scratch rows remain explicitly checkpoint-free.
All six experiment rows remain `NOT_STARTED`.

Training remains locked until both `TRAIN-01` and `RESUME-01` are complete and committed.

### TRAIN-01 implementation substate

TRAIN-01A governed-trainer implementation is locally prepared for review.

The implementation:
- uses one experiment-ID-driven launcher for all six frozen baseline conditions;
- uses a repository-installed `GovernedDetectionTrainer` so Ultralytics DDP subprocesses import the exact governed trainer;
- isolates model construction under configured seed 42, then restores the normal rank-specific DDP RNG state;
- verifies the INIT-01 9-class parameter/transfer contract at actual model construction;
- creates a train/validation-only runtime YAML with no test key;
- verifies DATA-01 train/validation membership by count and frozen stem SHA256 before launch;
- requires exact Python/PyTorch/Ultralytics/T4x2 runtime identity;
- refuses existing fresh-run output directories;
- writes preflight and runtime manifests into the persisted run evidence;
- refuses any launch while the experiment `source_commit` field remains blank.

No training has been authorized or executed. `RESUME-01` remains required before source-commit binding and launch authorization.

## DATA-01 binding summary

### Split A augmented

- Kaggle source: `nithoukhan/grazpedwri-dx-aug`
- runtime root observed: `/kaggle/input/datasets/nithoukhan/grazpedwri-dx-aug/GRAZPEDWRI-DX/data`
- training folder: 28,408 images/labels = 14,204 `orig_` + 14,204 `aug_`
- validation: 4,094
- test: 2,029
- historical split CSVs exactly partition all 20,327 canonical images
- patients: 4,263 / 1,218 / 610 with zero cross-split patient overlap
- Split-A training overlaps Split-B train/validation/test by 9,852 / 2,171 / 2,181 images
- governance: Split-A baseline results are firewalled from Split-B architecture, loss, epoch-budget, training-recipe and final-model selection

### Split B original

- Kaggle source observed: `utopianstar/grazpedwri-dx-split-b`
- runtime root observed: `/kaggle/input/datasets/utopianstar/grazpedwri-dx-split-b/GRAZPEDWRI-DX Split B`
- train / validation / test: 14,227 / 3,050 / 3,050
- patients: 4,264 / 914 / 913
- patient overlap: zero
- train/validation/test membership hashes match the frozen Split-B contract exactly

### Split B historical augmented

- same frozen Split-B package
- `train_aug_historical`: 28,454 = 14,227 frozen originals + 14,227 `aug_` counterparts
- transformed bases equal frozen Split-B train exactly
- copied labels are byte-identical within the training condition
- validation/test base overlap: zero
- governance: historical offline-augmentation comparator only; unresolved historical generation-provenance drift remains explicit

## Contained DATA-01 diagnostic incident

`DATA-01B-R3` is superseded for cross-Split-B annotation comparison.

Its helper searched Split-B train/validation/test label folders while comparing Split-A training annotation bytes. Membership containment shows 2,181 Split-A train images fall inside Split-B test membership.

No model predictions, test metrics, model selection, architecture decision or persistent data/repository modification resulted from that diagnostic. The cross-Split-B label-byte result is excluded from authoritative evidence. `DATA-01B-R4` replaced it with a Split-A train-local `orig_`/`aug_` comparison and opened no Split-B labels.

## Next

1. complete TRAIN-01C local semantic review
2. create replacement frozen source commit `S2` with all six `source_commit` bindings blank
3. push `S2` and verify GitHub Research Integrity on the exact `S2` SHA
4. create replacement authorization commit `A2` binding all six rows to `S2`
5. push `A2` and verify GitHub Research Integrity on the exact `A2` SHA
6. start a fresh Kaggle working session and rerun `BASE-B-ORG-PT-S42` preflight
7. launch `BASE-B-ORG-PT-S42` only after the corrected preflight passes
8. preserve the complete run directory for Save-Version continuation
9. continue the remaining five seed-42 baseline conditions
10. perform fresh Split-B-validation-only model diagnostics
11. freeze architecture and loss
12. implement the final method

## Active baseline experiments

- `BASE-B-ORG-PT-S42`
- `BASE-B-ORG-SCR-S42`
- `BASE-B-AUG-PT-S42`
- `BASE-B-AUG-SCR-S42`
- `BASE-A-AUG-PT-S42`
- `BASE-A-AUG-SCR-S42`

All remain `NOT_STARTED`.

## Test policy

Neither Split A test nor Split B test is used for architecture, loss, training-recipe, epoch-budget or model-selection decisions.

For Split B, pre-final-test access is limited to governed membership/fingerprint and aggregate split-integrity evidence. Test annotation contents, predictions, metrics and error analysis are not used for development.

Because Split-A train overlaps Split-B validation/test membership, Split-A experiments are reporting-only comparators and are not inputs to the Split-B development or final-model-selection path.

## 2026-09-23 - RESUME-01 implementation freeze preparation

`RESUME-01A` read-only source/interface audit passed at repository HEAD
`6bf9764c8455815d57cd253be8284717d6711962`.

The audit confirmed:
- upstream `Model.train(resume=...)` binds continuation to the loaded checkpoint;
- checkpoint training arguments are restored by `BaseTrainer.check_resume`;
- optimizer, AMP scaler, EMA and best-fitness state are restored from `last.pt`;
- DDP preserves the existing save directory during resume;
- the current pre-RESUME `GovernedDetectionTrainer` was fresh-run-only and required an explicit resume-aware branch;
- upstream trainer construction rewrites `args.yaml`, so governed continuation must archive the pre-resume file before trainer construction.

`RESUME-01B` implements the governed continuation path:
- a dedicated `research/runtime/resume_runner.py`;
- strict prior-run discovery and full-run copy verification;
- exact `last.pt` SHA256 verification before and after copy;
- exact original execution-commit continuity across sessions;
- frozen scientific-argument verification;
- regeneration of the train/validation-only runtime YAML with exact original-session SHA256 equality;
- resume-aware 499/499 trained-state restoration in `GovernedDetectionTrainer`;
- append-only numbered resume preflight/runtime manifests;
- preservation of the original TRAIN-01 `RUNTIME_MANIFEST.json`;
- archival of the pre-resume `args.yaml`;
- no manual scientific hyperparameter override surface.

The baseline recipe status is now durable as `FROZEN_BASELINE_RECIPE_V2`, and the
TRAIN-01 trainer contract is durable as `IMPLEMENTED_RESUME_AWARE`.

Training remains fail-closed while `EXPERIMENTS.csv::source_commit` is blank.

The source-freeze protocol is two-stage:
1. remotely CI-verify and freeze the RESUME-01 implementation commit as source commit `S`;
2. create a separate descendant authorization commit that writes `S` into all six baseline `source_commit` fields.

No baseline training is authorized by the RESUME-01B implementation commit itself.

## RESUME-01B R3 hardening before source freeze

A post-R1 code-level audit identified governance surfaces that required strengthening before commit:

- `EXPERIMENTS.csv` is now frozen relative to source commit `S` for every field except the intentional atomic `source_commit` binding; all six current bindings must equal `S`.
- The preflight manifest SHA256 and runtime train/validation YAML SHA256 are bound into the actual trainer startup.
- The runtime YAML test-key absence and exact train/validation paths are verified **before Ultralytics opens any dataset**, then checked again during trainer setup.
- Resume cross-checks `args.yaml` against `last.pt::train_args`.
- Resume cross-checks `results.csv` against `last.pt::train_results`.
- `best.pt` must satisfy the same model, Git-commit, branch, and Ultralytics-runtime identity contract before continuation.

This is pre-commit hardening, not a rollback of TRAIN-01. Training remains locked and all six experiment rows remain `NOT_STARTED`.

## 2026-09-23 - RESUME-01 source freeze remotely verified

Frozen training-source commit:

- source commit `S`: `2954eab070368932ae68370532b73295feb3ca2d`
- parent: `6bf9764c8455815d57cd253be8284717d6711962`
- commit subject: `research: freeze governed resume runtime`
- GitHub Research Integrity run: `35853996772` (run #13)
- event: `push`
- exact workflow head SHA: `2954eab070368932ae68370532b73295feb3ca2d`
- workflow conclusion: `success`
- integrity job: `107158063197` / `success`
- repository-state verification step: `success`
- research-integrity/D00/RESUME-01 test step: `success`

The source-freeze half of RESUME-01 is therefore remotely closed.

Authorization is intentionally separate. The authorization commit binds all six
`EXPERIMENTS.csv::source_commit` values to `S` while leaving every guarded
scientific/runtime path byte-identical to `S`. All six experiment statuses remain
`NOT_STARTED`.

Model training remains locked until the authorization commit itself is pushed and its
GitHub Research Integrity workflow succeeds.

## 2026-09-24 - TRAIN-01C zero-epoch exact-fork interface correction

The first governed `BASE-B-ORG-PT-S42` launch used source commit
`2954eab070368932ae68370532b73295feb3ca2d` and authorization commit
`4b07d32846188cf9d6d5d0368f1f6db0d9bc32f0`. The governed preflight
passed twice with identical SHA256
`3b3a6a8f8e507f04d457c3364b3cac1659c7d4a6d739254142a79ebebeb61360`,
with `TEST_ACCESS=NONE`.

Execution constructed the correct nine-class YOLO11s target
(9,431,275 parameters) but failed before epoch 1 because
`GovernedDetectionTrainer.get_model()` read `model.nc` immediately after
`DetectionModel(...)` construction.

Exact-fork source review confirmed:
- construction-time class count is available as `model.yaml["nc"]`;
- the Detect head exposes `model.model[-1].nc`;
- direct `model.nc` is attached only later by
  `DetectionTrainer.set_model_attributes()` during `_setup_train()`.

Disposition:
- epochs started: 0;
- training metrics/model-selection evidence produced: none;
- Split-B test predictions/metrics/error analysis accessed: none;
- old `S/A` remain historical provenance but are superseded for execution;
- all six experiments remain `NOT_STARTED`;
- old source bindings are cleared before replacement source freeze;
- corrected execution requires new remotely CI-verified `S2` and `A2`;
- the failed Kaggle run directory must not be resumed.

## 2026-09-24 - Replacement source freeze S2 remotely verified

Corrected frozen training source:

- replacement source commit `S2`: `46f40838c1c24a8ced77a2b868dfa0f7f1037f9c`;
- commit subject: `research: freeze corrected governed trainer source`;
- GitHub Research Integrity run: `35956458084` (run #15);
- event: `push`;
- exact workflow head SHA: `46f40838c1c24a8ced77a2b868dfa0f7f1037f9c`;
- workflow conclusion: `success`;
- integrity job: `107495713588` / `success`;
- repository-state verification step: `success`;
- research-integrity/D00/RESUME-01 test step: `success`.

`S2` contains the exact-fork construction-time class-count correction and the
zero-epoch incident record. All six baseline rows remain `NOT_STARTED` and all six
`source_commit` values remain blank at `S2`.

Authorization remains intentionally separate. `A2` may change only:
- `EXPERIMENTS.csv::source_commit` by binding all six rows to exact `S2`;
- project bookkeeping in `CURRENT.md` and `DECISIONS.md`.

No guarded scientific/runtime path may differ from `S2`.

## 2026-09-24 - B-AUG operational-readability correction preparation

A train-only read-only diagnostic of `DATA01:B-AUG-HIST:v1` confirmed:

- frozen image membership: 28,454;
- frozen label membership: 28,454;
- frozen membership SHA256: `4799e44ace20ab724974984f4cab60740a3fd8ad5cad3e1570bc895f02eefded`;
- operationally readable images: 28,452;
- structurally unreadable PNGs: exactly 2;
- diagnostic JSON SHA256: `e24d3b2f7fc123360ab1513508a958c5a0dc25ceaa4b108ea5778c5d5a10dbc0`;
- test access: none;
- dataset modification: none.

The two recorded unreadable files are:

1. `6015_0845856403_01_WRI-R2_M017.png`
   - SHA256 `eed6b57b8d1d0a7279d7366978d9cdb9c1ccc1fbbd0cfff773b48b75b9f87a23`
2. `aug_1355_0485876132_01_WRI-L1_M014.png`
   - SHA256 `1963d76af0fffcc07f70dc87f869243ad30c13127916396019d35f1eb65ba5f0`

Resolution: preserve the frozen 28,454-member identity unchanged and set the
TRAIN-01 operational expectation to 28,452 for B-AUG only. No image is deleted,
repaired, regenerated or relabeled.

Existing B-ORG and A-AUG runs already started under A2 remain valid and must resume
on exact A2. The failed B-AUG A2 attempts produced no completed epoch and are not
canonical. Future B-AUG fresh execution requires a remotely CI-verified S3/A3
successor chain.

## 2026-09-24 - S3 remotely verified; A3 authorization preparation

The B-AUG operational-readability correction source freeze is remotely closed:

- source commit `S3`: `18e75338ae116beccf3f5e4a0481efede601026b`;
- parent: `A2 = 9fe475175d3963a083d7afc29426f07d86c1887d`;
- commit subject: `research: freeze B-AUG operational readability correction`;
- GitHub Research Integrity run: `36024485727` (run #17);
- workflow conclusion: `success`;
- integrity job: `107717414682` / `success`;
- repository-state verification step: `success`;
- research-integrity/D00/RESUME-01 test step: `success`.

`S3` preserves the frozen B-AUG membership at 28,454 while recording 28,452
operationally readable training images and exactly two immutable unreadable PNGs.
No dataset repair, deletion, regeneration or relabeling occurred.

Authorization remains separate. `A3` may change only:
- `EXPERIMENTS.csv::source_commit`, atomically binding all six rows to exact `S3`;
- authorization bookkeeping in `CURRENT.md` and `DECISIONS.md`.

All six experiment rows remain `NOT_STARTED`. No guarded scientific/runtime path may
differ from `S3`. Existing runs that originally executed at `A2` retain their exact A2
resume lineage; `A3` authorizes future fresh execution from the S3 source, including the
two previously blocked B-AUG conditions.
