# Permanent Project Operating Rules

## Authority

These rules govern the Detection Project from 2026-10-05 onward unless
the user explicitly changes them.

They apply across chats, branches, experiments, diagnostics, manuscript
work and final evaluation.

The repository is the durable project memory.

Conversation memory may assist continuity but must never replace current
repository evidence.

For active state, use the latest committed:

- `research/CURRENT.md`;
- `research/DECISIONS.md`;
- `research/PROJECT_LOG.md`;
- active experiment registries and per-experiment records;
- immutable artifact hashes and execution archives.

Historical migration packages remain evidence for earlier phases but do
not override newer committed scientific decisions.

## Rule 1 ? Document every substantive project event

Every scientifically or operationally meaningful event must be recorded.

This includes:

- proposals and alternative ideas;
- hypotheses;
- accepted decisions;
- rejected and superseded decisions;
- experiment registrations;
- experiment results;
- negative/null results;
- implementation discoveries;
- failures;
- recovery actions;
- provenance corrections;
- literature/novelty conclusions;
- artifact creation;
- architecture freezes;
- evaluation decisions;
- changes to the next-step plan.

Do not rely on a previous chat as the sole record of an important fact.

Proposals must be clearly distinguished from accepted decisions.

Hypotheses must be clearly distinguished from demonstrated mechanisms.

Negative evidence must be preserved rather than silently discarded.

At every meaningful milestone, update the appropriate durable records
before moving to the next scientific phase.

## Rule 2 ? Preserve every experiment completely

Every training experiment must be reproducible and recoverable later.

### Before execution

The repository must contain the exact scientific implementation used by
Kaggle before training begins.

The run must be bound to:

- experiment ID;
- scientific question;
- exact Git source commit;
- dataset binding;
- split;
- initialization;
- seed;
- model configuration;
- training configuration;
- primary and secondary metrics;
- predefined decision rule;
- test-access state.

Reusable Kaggle launch, preflight, resume and analysis logic must be
version-controlled.

Do not treat an unrecorded notebook cell as authoritative scientific
implementation.

### During execution

Record:

- Git commit;
- runtime environment;
- GPU information;
- package versions;
- dataset identity/counts;
- initialization checkpoint identity;
- session lineage;
- resume lineage when applicable;
- failures and recoveries;
- runtime manifests.

### After execution

Preserve the complete run output externally/local where size makes Git
inappropriate.

Where produced, preserve at minimum:

- `best.pt`;
- `last.pt`;
- `results.csv`;
- `args.yaml`;
- training curves;
- precision/recall/F1/PR curves;
- confusion matrices;
- validation visualizations;
- per-class metrics;
- prediction-level outputs used for scientific analysis;
- runtime/governance manifests;
- relevant console/execution logs;
- launcher/wrapper provenance;
- session/resume history.

Large binary model artifacts remain outside Git when appropriate.

Git must contain compact evidence that makes the external artifact
recoverable:

- exact artifact name;
- external/local archive location;
- byte size where available;
- SHA256;
- experiment ID;
- source commit;
- scientific status;
- decision;
- relationship to parent/resume artifacts.

A complete raw experiment archive should also be preserved locally or in
the designated external storage so future tables, plots, diagrams,
re-analysis and manuscript revisions do not require rerunning the model.

Do not delete a negative experiment because it failed to improve metrics.

## Rule 3 ? Independent scientific judgment

The assistant must not automatically agree with the user's proposed
interpretation, architecture, module or next experiment.

For consequential scientific questions:

- inspect the available project evidence first;
- verify current repository state when relevant;
- perform current literature research when novelty or SOTA positioning is
  involved;
- compare alternatives rather than endorsing the first idea;
- identify methodological weaknesses and confounders;
- distinguish fact, source evidence, inference and hypothesis;
- explicitly state when evidence is insufficient;
- do not invent results, mechanisms, citations, files or provenance.

A higher validation metric alone is not enough to justify a final method.

Architecture changes must have a scientific rationale tied to a
demonstrated problem whenever possible.

Avoid uncontrolled module shopping and repeated validation-set probing.

## Rule 4 - Keep the repository simple and understandable

Scientific traceability must not create repository confusion.

Use the minimum number of canonical files needed to preserve complete
evidence.

### Canonical-file rule

Do not create a second tracker, decision file, artifact registry,
current-state file, governance system, or experiment registry when an
existing canonical file already serves that purpose.

Prefer extending the existing canonical structure over creating a parallel
one.

### Naming discipline

Names must be clear, stable, and understandable without relying on chat
memory.

Use consistent experiment-ID token order.
Use unique experiment IDs.
Use descriptive file names.
Use human-readable paper-facing model names.
Use clear and descriptive branch names.
Use the exact experiment ID in experiment records and archives where
practical.

Do not reuse an experiment ID for a different scientific condition.

Do not rename a completed experiment merely to improve presentation.

Do not silently mix internal implementation class names with paper-facing
names.

Define unavoidable abbreviations before using them broadly.

### Repository placement

Put files in the existing directory that matches their scientific purpose.

Do not create a new top-level research directory without a demonstrated
need.

Avoid duplicate copies of authoritative evidence in multiple active
locations.

### Superseded material

Historical evidence must not be deleted.

When something is superseded, mark it explicitly and identify the newer
authority.

Keep superseded historical material outside the active decision path.

### Large artifacts

Do not bloat Git with datasets, complete run directories, or large model
weights.

Keep large raw artifacts locally or in designated external storage.

Bind external artifacts to repository evidence using stable names, storage
locations, byte sizes where available, and SHA256 hashes.

### Simplicity principle

When two repository designs provide equivalent scientific traceability,
choose the simpler design.

## Primary scientific objective

The project goal is a technically strong, reproducible and defensible
pediatric wrist abnormality-detection study suitable for an SCI/JCR
Q1 or strong Q2 publication and the user's PhD research requirements.

The project should optimize the total scientific contribution, including:

- detection performance;
- methodological novelty;
- defensible problem formulation;
- reproducibility;
- robustness;
- uncertainty;
- per-class behavior;
- efficiency;
- generalization;
- clinical/modeling relevance;
- transparent negative evidence.

Performance improvement is important but is not the only objective.

## Test firewall

Split-B test remains sealed during development.

No architecture selection, module replacement, loss selection,
hyperparameter selection, diagnostic hypothesis selection or manuscript
claim may use Split-B test performance before the final method and final
evaluation protocol are frozen.

## Continuity rule

At the beginning of a new chat or after uncertainty about project state,
reconstruct authority from the repository and latest migration/handoff
before acting.

Do not silently reconstruct scientific facts from memory when exact
repository or artifact evidence is available.

## Rule 5 - Durable evidence at every governed milestone (2026-10-09)

**The chat transcript is never the only archive of a substantive project action.**
Before marking a gate complete or moving to the next consequential phase, persist:

1. The canonical repo-side facts in the existing `research/CURRENT.md`, `research/DECISIONS.md`, `research/PROJECT_LOG.md`, and relevant stage tracker/experiment registry **only where needed** (do not introduce duplicate trackers).
2. The exact reviewed and executed code, scripts, test definitions, failures, correction patches and scientific rationale in their existing purpose-matched repository locations where Git-sized and safe.
3. Full stdout/stderr, environment, workstation/Kaggle session logs, source snapshots while uncommitted, backup copies, large outputs, checkpoints and proof artifacts in a governed persistent external/local evidence vault, not just `Temp` or `Downloads`.
4. SHA256, byte size, path/storage binding, source branch/commit and candidate-file hashes for all off-repo evidence, registered in the existing provenance records where applicable.
5. A clear distinction between a successfully executed *test* and an independently accepted *scientific gate*; preserve expected negative controls and unresolved failure modes explicitly.
6. A cross-chat handoff that reconciles repository state, local uncommitted/staged changes, evidence manifest and hashes, authorizations used/unused, pending scientific questions, and the precise next allowed step.

For every incomplete operation, record where execution stopped, which files may have changed, what remains to verify and whether a rerun is authorized. Never silently erase, overwrite or relabel earlier failures.

**Provenance firewall:** When a tightly pinned implementation/test transaction requires an unchanged Git HEAD or exact worktree status, preserve evidence **immediately in the external vault** and prepare append-ready repository records; incorporate them into the canonical tracked files in a separately verified and governed documentation commit after the pinned test gate allows it. Do not mutate HEAD merely to improve recordkeeping and thereby invalidate the scientific preflight.

Repository commits, remote pushes, evaluation/training, source edits and documentation updates remain separately governed actions. Keeping a pending record in a chat or uncommitted local file is not equivalent to preserving it in the remote repository.
