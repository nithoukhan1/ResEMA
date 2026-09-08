# Kaggle Run SOP

## Before launch
- [ ] experiment registered in Git
- [ ] exact commit SHA recorded
- [ ] config committed
- [ ] decision rule committed
- [ ] dataset attached
- [ ] test evaluation disabled

## Setup
- [ ] clone repo
- [ ] checkout exact SHA
- [ ] assert clean status
- [ ] editable install with `--no-deps` unless documented otherwise
- [ ] verify import source
- [ ] record runtime versions
- [ ] verify dataset counts
- [ ] verify model topology
- [ ] hash pretrained checkpoint

## Train
- [ ] preserve transfer console output
- [ ] run standard entrypoint
- [ ] save complete run folder

## Resume
- [ ] attach latest complete run
- [ ] restore full folder
- [ ] inspect last.pt optimizer state
- [ ] resume without redefining scientific settings

## Final
- [ ] independent validation
- [ ] per-class metrics
- [ ] diagnostic macros
- [ ] SHA256 critical artifacts
- [ ] publish immutable Kaggle Model version
- [ ] update Git registry/decision
