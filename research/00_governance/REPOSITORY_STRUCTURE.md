# 05 — Repository Structure and Maintenance

## Existing repository
Continue in `nithoukhan1/ResEMA`.

Verified:
- active public repository;
- default branch `main`;
- existing historical research branches;
- repository is an Ultralytics fork with inherited CI/workflows.

## New publication branch
Recommended:
`research-v7-publication`

Base commit:
`5a511729bc21c1b0998b7e6f4a84108fb4e309bf`

Do not rewrite historical branches.

## New research tree
```text
research/
├── README.md
├── 00_governance/
├── 01_provenance/
├── 02_history/
├── 03_literature/
├── 04_data/
├── 05_experiments/
├── 06_diagnostics/
├── 07_method/
├── 08_robustness/
├── 09_final_test/
├── 10_manuscript/
├── tools/
└── tests/
```

## Research-specific CI
Add a small `.github/workflows/research-ci.yml` that runs:
- syntax/import checks
- research unit tests
- dummy model build
- CPU forward pass
- identity-init tests
- deterministic PELT sampler tests
- config validation
- no large data or training

## Environment policy
Maintain:
1. `LOCAL_DEV_ENVIRONMENT.md`
2. `KAGGLE_RUNTIME_LOCK.md`

Important existing discrepancy:
the pinned fork's `pyproject.toml` declares `torch<2.10`, while successful Kaggle runs used `torch 2.10.0+cu128`.

Do not let pip silently replace the Kaggle CUDA stack.
Use `pip install -e . --no-deps` in Kaggle unless testing proves another approach is safer.

## Git hygiene
Never commit:
- .pt weights
- full datasets
- Kaggle caches
- large run directories

Commit:
- source
- configs
- small CSV/JSON/YAML
- hashes
- selected plots
- decisions
- manuscript assets
