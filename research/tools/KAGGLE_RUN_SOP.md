# Kaggle Run SOP

## Core rule
One scientific experiment may span multiple Kaggle Save-Version sessions.

## Before launch
- [ ] experiment registered in `research/05_experiments/EXPERIMENTS.csv`
- [ ] exact Git SHA recorded
- [ ] config committed
- [ ] dataset binding verified
- [ ] test evaluation disabled
- [ ] pretrained checkpoint SHA256 verified when applicable

## Setup
- [ ] clone repo
- [ ] checkout exact SHA
- [ ] assert clean status
- [ ] editable install with `--no-deps` unless documented otherwise
- [ ] verify imported Ultralytics source
- [ ] record runtime versions
- [ ] verify dataset counts
- [ ] verify model topology

## Train
- [ ] use fixed experiment ID as run name
- [ ] preserve the complete run folder

## Save Version
Before a session ends verify:
- [ ] `args.yaml`
- [ ] `results.csv`
- [ ] `weights/best.pt`
- [ ] `weights/last.pt`

Persist the complete run directory.

Recommended saved-session names:
`<EXPERIMENT-ID>-SESSION01`, `SESSION02`, ...

## Resume
- [ ] attach latest successfully saved run
- [ ] checkout the same Git SHA
- [ ] hash mounted `last.pt`
- [ ] copy full previous run folder to `/kaggle/working`
- [ ] verify copied `last.pt` hash
- [ ] inspect previous last epoch
- [ ] load `YOLO(last.pt)`
- [ ] call `model.train(resume=True)`
- [ ] do not redefine scientific hyperparameters manually
- [ ] persist the new complete run folder

Must remain unchanged:
Git commit, experiment ID, dataset membership, model, initialization, seed, optimizer/LR, image size, batch, augmentation, loss, epoch budget.

## Final
- [ ] collect validation/per-class metrics
- [ ] hash best.pt / last.pt / results.csv / args.yaml
- [ ] keep `.pt` outside Git
- [ ] update `research/01_provenance/ARTIFACTS.csv`
- [ ] update `research/05_experiments/EXPERIMENTS.csv`
- [ ] document the decision
