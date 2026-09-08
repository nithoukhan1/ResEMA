# 10 — Reproducibility Checklist

## Data
- [ ] source/version documented
- [ ] patient counts documented
- [ ] split manifest frozen
- [ ] patient overlap = 0
- [ ] class counts recorded
- [ ] corrupt/missing files recorded
- [ ] metadata mapping documented
- [ ] test sealed

## Model
- [ ] repository commit SHA
- [ ] model YAML/source hash
- [ ] parameter count
- [ ] module topology audit
- [ ] pretrained source + SHA256
- [ ] transferred tensor/parameter coverage

## Training
- [ ] config committed before run
- [ ] seed
- [ ] epochs/patience
- [ ] optimizer/LR
- [ ] loss weights
- [ ] augmentations
- [ ] runtime
- [ ] model-selection rule

## Kaggle
- [ ] exact commit checkout
- [ ] clean git status
- [ ] runtime manifest
- [ ] dataset counts asserted
- [ ] no test access

## Output
- [ ] best.pt hash
- [ ] last.pt hash
- [ ] results.csv hash
- [ ] args.yaml hash
- [ ] independent validation
- [ ] per-class metrics
- [ ] artifact version/path
- [ ] decision recorded

## Final reporting
- [ ] patient-disjoint split described
- [ ] initialization described
- [ ] mixed pretrained/random components described
- [ ] hyperparameter search described
- [ ] negative ablations included
- [ ] uncertainty reported
- [ ] software/hardware versions reported
- [ ] repository released
