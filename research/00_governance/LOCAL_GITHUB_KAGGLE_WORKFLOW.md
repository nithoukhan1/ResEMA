# 04 — Local VS Code -> GitHub -> Kaggle Workflow

## Core rule
Scientific code lives in Git.
Kaggle executes an immutable Git commit.
Kaggle notebooks are runners, not the authoritative source of scientific logic.

## A. Local VS Code
1. Update research branch:
```bash
git checkout research-v7-publication
git pull
```

2. Create short-lived branch:
```bash
git checkout -b feat/tp-cda
# or feat/apcf, train/pelt, diagnostic/d00
```

3. Implement locally.

4. Run preflight:
```bash
python -m pytest research/tests -q
python research/tools/verify_repo_state.py
```

5. Commit one scientific change per commit where possible.

6. Push branch.

7. Use a PR even for solo work for diff review/provenance.

8. Merge only after tests pass.

## B. Register experiment BEFORE Kaggle
Create:
`research/05_experiments/<ID>/`

Required before run:
- README.md
- config.yaml
- run.py
- resume.py or standardized resume command
- verify.py
- predeclared decision rule

Commit everything.

Record:
```bash
git rev-parse HEAD
```

That SHA is the experiment code identity.

## C. Kaggle setup
Notebook runner should:
1. clone repo;
2. checkout exact registered commit;
3. assert clean Git status;
4. verify imported Ultralytics source;
5. record runtime versions;
6. verify dataset counts/manifests;
7. call committed experiment entrypoint.

Preferred:
```bash
git clone https://github.com/nithoukhan1/ResEMA.git
cd ResEMA
git checkout <EXPERIMENT_SHA>
pip install -e . --no-deps
```

Why `--no-deps`:
the pinned fork's pyproject declares `torch<2.10`, while successful Kaggle runs used `torch 2.10.0+cu128`. Do not let pip silently replace the working CUDA stack.

Then:
```bash
python research/05_experiments/M01/run.py   --config research/05_experiments/M01/config.yaml
```

## D. Runtime manifest
Before training save:
- remote
- commit SHA
- clean/dirty state
- Python
- Ultralytics
- PyTorch
- torchvision
- CUDA
- GPU
- GPU count
- dataset paths/counts
- config SHA256
- model YAML SHA256
- pretrained checkpoint SHA256

File:
`runtime_manifest.json`

## E. Resume
Resume uses:
- complete previous run folder
- latest last.pt
- same scientific config

Do not redefine scientific hyperparameters manually.

If a code change is required only for resume mechanics, document both original and resume commit SHA.

## F. Completion
Generate:
- best.pt
- last.pt
- results.csv
- args.yaml
- runtime_manifest.json
- artifact_manifest.json
- results_summary.yaml
- independent-validation CSV
- per-class CSV
- plots

Hash critical files.

Publish entire run folder as immutable Kaggle Model version.

## G. Return to local
1. download/copy small metadata/results;
2. update artifact registry;
3. update experiment registry;
4. write decision.md;
5. update project tracker;
6. commit/push.

Weights remain in Kaggle.

## H. Analysis-code rule
Any analysis that can alter a scientific conclusion must be version-controlled.
Paper tables should be generated from registered CSV/JSON whenever possible, not manual calculator transcription.
