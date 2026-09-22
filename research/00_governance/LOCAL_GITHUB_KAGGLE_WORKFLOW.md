# 04 - Local VS Code -> GitHub -> Kaggle Workflow

## Core rule

Scientific logic lives in Git.
GitHub is the source of truth.
Kaggle executes an immutable Git commit.

Kaggle notebooks are launchers and inspection surfaces, not the authoritative source of scientific logic.

## A. Local development

1. Start from the active branch recorded in `research/CURRENT.md`.
2. Verify the working tree is clean.
3. Implement one governed change at a time.
4. Run:
   - `python -m pytest research/tests -q`
   - `python research/tools/verify_repo_state.py`
5. Review the complete diff.
6. Commit and push.
7. Record the immutable Git SHA used for Kaggle.

## B. Register an experiment before Kaggle

Create:

`research/05_experiments/records/<EXPERIMENT-ID>/`

At minimum before launch, record:
- experiment ID;
- scientific question;
- dataset binding;
- initialization;
- seed;
- training configuration;
- primary metric;
- decision rule;
- exact Git SHA.

Reusable scientific execution code should live in version-controlled shared tools rather than being copied into every experiment directory.

Per-experiment records store configuration and provenance, not duplicate implementations.

## C. Kaggle setup

The Kaggle launcher should:

1. clone `nithoukhan1/ResEMA`;
2. checkout the exact registered Git SHA;
3. assert clean Git status;
4. install the local fork with `pip install -e . --no-deps` unless a documented runtime change is required;
5. verify the imported Ultralytics source;
6. record Python, Ultralytics, PyTorch, torchvision, CUDA, GPU and GPU-count information;
7. verify the dataset binding and counts;
8. verify the pretrained checkpoint SHA256 when applicable;
9. call the committed reusable training entrypoint.

Why `--no-deps`:
the pinned fork and successful Kaggle CUDA environment may have dependency-version differences. Do not allow pip to silently replace a working CUDA/PyTorch stack.

## D. Runtime record

Before training, record at minimum:
- remote;
- Git SHA;
- clean/dirty state;
- runtime versions;
- GPU model/count;
- dataset binding;
- dataset counts;
- config hash;
- model config/source hash;
- pretrained checkpoint hash when applicable.

## E. Kaggle Save Version and resume

One scientific experiment may span multiple Kaggle sessions.

Resume requires:
- the complete previous persisted run directory;
- the latest valid `last.pt`;
- the same Git SHA;
- the same dataset binding;
- the same scientific configuration.

Restore the full previous run directory into writable `/kaggle/working`, verify hashes, then resume from `last.pt` with `resume=True`.

Do not redefine scientific hyperparameters in a resume session.

If the previous Kaggle session was not successfully persisted, resume from the most recent successfully persisted version.

## F. Completion

Persist the final complete run folder externally.

Register compact evidence in Git:
- resolved configuration;
- epoch metrics;
- validation summary;
- per-class metrics;
- runtime/session history;
- artifact hashes;
- scientific decision.

Large `.pt` weights remain outside Git.

## G. Return to local

Update:
- `research/05_experiments/EXPERIMENTS.csv`;
- `research/01_provenance/ARTIFACTS.csv`;
- the relevant `records/<EXPERIMENT-ID>/` files;
- `research/CURRENT.md`;
- `research/DECISIONS.md` when a project-level decision is made.

Then commit and push.

## H. Analysis-code rule

Any analysis that can change a scientific conclusion must be version-controlled.
Paper tables should be generated from registered machine-readable evidence whenever possible.
