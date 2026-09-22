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

`INIT-01` - freeze the exact official YOLO11s pretrained initialization identity before any baseline training.

INIT-01 must record the official checkpoint source/reference, runtime acquisition path, exact file size and SHA256, checkpoint/model identity, relevant Ultralytics/runtime version evidence, and the transfer-versus-scratch initialization contract.

The six baseline experiment rows remain `NOT_STARTED`.
Training remains locked until `INIT-01`, `TRAIN-01` and `RESUME-01` are complete and committed.

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

1. `INIT-01` - freeze official YOLO11s pretrained checkpoint identity and SHA256
2. `TRAIN-01` - implement reusable governed baseline trainer
3. `RESUME-01` - implement Kaggle Save-Version/resume helper
4. launch `BASE-B-ORG-PT-S42`
5. register validation/per-class/convergence evidence
6. continue the remaining five seed-42 baseline conditions
7. perform fresh Split-B-validation-only model diagnostics
8. freeze architecture and loss
9. implement the final method

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
