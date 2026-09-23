# Kaggle Run SOP

## Core rule
One scientific experiment may span multiple Kaggle Save-Version sessions.

The initial launch must use the governed fresh launcher. Every continuation must use the governed RESUME-01 launcher. Do not call Ultralytics resume directly from an ad-hoc notebook cell.

## Before initial launch
- [ ] experiment registered in `research/05_experiments/EXPERIMENTS.csv`
- [ ] non-empty frozen `source_commit` bound by the RESUME-01 authorization commit
- [ ] exact execution Git SHA recorded
- [ ] config committed
- [ ] dataset binding verified
- [ ] test evaluation disabled
- [ ] pretrained checkpoint SHA256 verified when applicable

## Setup
- [ ] clone repo
- [ ] checkout the exact authorized execution SHA
- [ ] assert clean status
- [ ] editable install with `--no-deps` unless documented otherwise
- [ ] verify imported Ultralytics source
- [ ] record runtime versions
- [ ] verify dataset counts
- [ ] verify model topology

## Initial training session
Use the fixed experiment ID as the run name through the governed fresh launcher:

```bash
python research/runtime/baseline_runner.py \
  --experiment-id <EXPERIMENT-ID> \
  --mode preflight-only
```

After the preflight passes, execute with the same checkout and inputs:

```bash
python research/runtime/baseline_runner.py \
  --experiment-id <EXPERIMENT-ID> \
  --mode execute
```

Preserve the complete run folder.

## Save Version
Before a session ends verify that the persisted complete run folder contains:
- [ ] `args.yaml`
- [ ] `results.csv`
- [ ] `weights/best.pt`
- [ ] `weights/last.pt`
- [ ] `governance/PRETRAIN_PREFLIGHT.json`
- [ ] `governance/RUNTIME_MANIFEST.json`
- [ ] any existing `governance/resume_sessions/` lineage files

Recommended saved-session names:
`<EXPERIMENT-ID>-SESSION01`, `SESSION02`, ...

## Resume
Attach only the latest successfully saved complete run for the experiment, plus the same frozen datasets required by the original run.

The repository checkout must equal the exact `execution_commit` recorded by the original TRAIN-01 runtime manifest. A descendant SHA is not sufficient for continuation, even if scientific files are unchanged.

First run the governed read-only/materialization-free preflight:

```bash
python research/runtime/resume_runner.py \
  --experiment-id <EXPERIMENT-ID> \
  --mode preflight-only
```

The preflight must verify:
- [ ] exactly one complete prior run directory
- [ ] same experiment ID
- [ ] same frozen training source commit
- [ ] exact same execution commit
- [ ] same data binding and original initialization identity
- [ ] same scientific training arguments
- [ ] regenerated train/validation-only runtime YAML matches the original session SHA256
- [ ] mounted `last.pt` SHA256 and checkpoint structure
- [ ] optimizer, scaler and EMA states are present
- [ ] `results.csv` last epoch matches `last.pt`
- [ ] test access remains disabled

Then execute:

```bash
python research/runtime/resume_runner.py \
  --experiment-id <EXPERIMENT-ID> \
  --mode execute
```

The governed resume launcher will:
- copy the complete previous run folder to the canonical `/kaggle/working/ResEMA_baseline_runs/<EXPERIMENT-ID>` path;
- verify an exact file-by-file copy before adding new lineage records;
- archive the pre-resume `args.yaml` before upstream Ultralytics rewrites it;
- append a numbered RESUME-01 preflight manifest;
- load `YOLO(last.pt)` and call `model.train(trainer=GovernedDetectionTrainer, resume=True)`;
- restore checkpoint training arguments and optimizer/scaler/EMA state through the exact frozen Ultralytics resume path;
- append a numbered RESUME-01 runtime manifest without overwriting the original TRAIN-01 runtime manifest.

Do **not** redefine optimizer, learning rate, seed, image size, batch, augmentations, loss, epoch budget or other scientific hyperparameters manually during resume.

Must remain unchanged across continuation sessions:
Git execution commit, experiment ID, dataset membership, model architecture, original initialization identity, seed, optimizer/LR, image size, batch, augmentation, loss and epoch budget.

## Final
- [ ] collect validation/per-class metrics
- [ ] hash `best.pt`, `last.pt`, `results.csv`, `args.yaml`
- [ ] retain all governance/resume lineage files
- [ ] keep `.pt` outside Git
- [ ] update `research/01_provenance/ARTIFACTS.csv`
- [ ] update `research/05_experiments/EXPERIMENTS.csv`
- [ ] document the decision

Neither Split A test nor Split B test is used for architecture, loss, epoch-budget, training-recipe or model-selection decisions.

## Source-binding and evidence-integrity hardening
- The authorization commit may change only `EXPERIMENTS.csv::source_commit` relative to frozen source commit `S`.
- At execution, every other experiment-matrix field must still equal the matrix stored in `S`, and all six `source_commit` cells must equal `S`.
- The launcher binds the preflight manifest SHA256 into the trainer environment; the trainer re-verifies it before any epoch starts.
- The trainer re-verifies the preflight SHA256, runtime train/validation YAML SHA256, test-key absence, and exact train/validation paths **before Ultralytics opens any dataset**, then rechecks them at trainer setup.
- On resume, `args.yaml` must agree with `last.pt::train_args`.
- On resume, `results.csv` must agree with `last.pt::train_results`.
- `best.pt` must match the frozen model/Git/runtime identity before continuation.
