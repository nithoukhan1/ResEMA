# Detection Project Log

This is an append-only chronological log of substantive project events.

It complements:

- `CURRENT.md` for active state;
- `DECISIONS.md` for accepted project-level decisions;
- experiment registries for experiment state;
- artifact registries for external evidence.

The log may contain proposals, hypotheses, failures and recoveries.
Those entries are not automatically accepted scientific conclusions.

For history before 2026-10-05, use the frozen migration/handoff packages,
historical registries and decision records.

## 2026-10-05 - A12 corrected-module-family closure

Type: `DECISION / GOVERNANCE`

Scientific closure commit before this governance transaction:

`70550e1e5390b3e0b714de6855703d18807b46cc`

State:

- corrected module family closed;
- SCConv-Early selected as corrected-family candidate;
- global final paper architecture NOT frozen;
- six-model residual diagnostic required;
- DySample diagnostic explicitly required;
- no corrected SCConv-Early + DySample run claimed;
- no corrected three-way SCConv + DySample + EMA run claimed;
- new GPU training not authorized;
- Split-B test access NONE.

Next scientific task remains:

`A12-D0_DIAGNOSTIC_ARTIFACT_INVENTORY`

## 2026-10-05 - Permanent operating rules requested

Type: `GOVERNANCE`

The user established three permanent project requirements:

1. preserve substantive proposals, decisions, experiments, failures,
   recoveries and rationale so the project can be reconstructed later;
2. preserve exact experiment code/provenance before training and complete
   outputs/artifacts after training so results can be reused without
   rerunning;
3. require independent evidence-based scientific judgment rather than
   automatic agreement, with deep research for consequential methodology
   and novelty decisions and the Q1/Q2 publication objective kept central.

These requirements are codified in:

`research/00_governance/PROJECT_OPERATING_RULES.md`

## 2026-10-05 - Repository simplicity and naming discipline

Type: GOVERNANCE

A fourth permanent project requirement was adopted:

Keep the repository clean, consistently named, minimally duplicative, and
easy to reconstruct.

Operational consequences:

- use one canonical file for each governance or tracking purpose;
- use stable and understandable experiment IDs;
- use clear human-readable paper-facing model names;
- avoid unnecessary branches and duplicate trackers;
- place artifacts in the existing directory matching their purpose;
- keep large raw outputs outside Git and bind them by SHA256;
- explicitly mark superseded material;
- prefer the simpler design when scientific traceability is equivalent.

Canonical authority:

research/00_governance/PROJECT_OPERATING_RULES.md

## 2026-10-05 - Rule-4 freeze verification stopped and forward-recovered

Type: FAILURE / RECOVERY

The first Python Rule-4 freeze attempt successfully wrote Rule 4 and the
repository-simplicity event, then stopped before staging because its
verification expected the publication objective as one contiguous string.

The existing operating-rules file expresses the same objective across line
breaks as `SCI/JCR` and `Q1 or strong Q2`.

Scientific impact:

- none;
- no training occurred;
- no dataset was accessed;
- Split-B test remained sealed;
- no file was staged by the stopped attempt;
- no commit was created;
- no push occurred.

Recovery:

- preserve the already-written Rule 4 and project-log event;
- verify the publication objective semantically rather than by one fragile
  contiguous string;
- continue with the same two canonical governance files only.

## 2026-10-06 - A12-D0 diagnostic artifact inventory

Type: DIAGNOSTIC / PROVENANCE

Read-only A12-D0 inspected the six required residual-diagnostic models.

Result:

- expected models: 6;
- canonical model artifacts found: 6;
- repository files modified by D0: none;
- training: none;
- dataset access: none;
- Split-B test access: none;
- new GPU training remains unauthorized.

The canonical final archives are:

- BASE-B-ORG-PT-S42-2.zip;
- BORG-PT-S42-SCCONV-EARLY-E100-2.zip;
- BORG-PT-S42-SCCONV-4STAGE-E100-2.zip;
- BORG-PT-S42-DYSAMPLE-E100-1.zip;
- BORG-PT-S42-CANONICAL-EMA-E100-2.zip;
- BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100-2.zip.

Each canonical archive was identified by frozen member hashes rather than
filename alone.

A12-D0A then copied both COMB session ZIPs into the governed external
artifact store.

COMB canonical final archive:

`BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100-2.zip`

SHA256:

`d0de1f91d0a914c2b0b334fc622c17266fa65c310dd10d01e004d495dfa76f01`

D0A local archive manifest SHA256:

`0144fb6096d7c3044a218566445159b9b06929570ff03be1a91c1ab52bbc4b53`

A12-D0B records the previously missing corrected single-module and COMB
artifacts in the existing canonical `research/01_provenance/ARTIFACTS.csv`
rather than creating a parallel registry.

The exact executed A12-D0 inventory script is preserved at:

`research/tools/a12_d0_diagnostic_artifact_inventory.py`

Exact script SHA256:

`5e71f41e90281a4912a41784bbc64bd74c2e6a6519e8b516787901185e2e3c88`

Next action:

`A12-D1_STANDARDIZED_VALIDATION_DIAGNOSTIC_PREFLIGHT`

## 2026-10-06 - A12-D0B status-parser failure and forward recovery

Type: FAILURE / RECOVERY

The first A12-D0B recording transaction successfully:

- captured the exact executed A12-D0 inventory tool;
- verified all five corrected-model/combination final archives and members;
- verified the COMB local archive manifest;
- added exactly 21 intended rows to the canonical artifact registry;
- updated CURRENT.md;
- appended the A12-D0 project-log event.

It then stopped before tests, staging, commit, or push because its
`git status --porcelain` helper stripped the leading status-space before
path parsing. This caused the displayed path
`research/01_provenance/ARTIFACTS.csv` to appear as
`esearch/01_provenance/ARTIFACTS.csv`.

Scientific impact:

- none;
- no training occurred;
- no dataset was accessed;
- Split-B test remained sealed;
- no file was staged;
- no commit was created;
- no push occurred.

Recovery:

- preserve the four already-written intended files;
- verify tracked and untracked memberships separately without porcelain
  substring parsing;
- verify the artifact-registry delta against the pre-D0B Git parent;
- run governance tests;
- freeze exactly the same four-file A12-D0B transaction.

## 2026-10-06 - A12-D1 standardized validation diagnostic protocol candidate

Type: DIAGNOSTIC DESIGN / IMPLEMENTATION

After A12-D0/D0B established six canonical diagnostic checkpoints, the next
phase was defined as a common validation-only diagnostic rather than a new
training experiment.

The D1 design decision is:

- revalidate all six frozen `best.pt` checkpoints;
- use one current source-frozen evaluation checkout for all six;
- preserve historical training-time selection metrics separately;
- reuse the governed BASELINE-FREEZE-01 standardized validation runtime;
- export low-confidence post-NMS predictions in A12-D2 so all later
  diagnostic thresholds can be applied offline without rerunning models;
- use authoritative Split-B validation metadata for patient linkage;
- freeze class, confidence, localization, size, patient and error-transition
  diagnostic rules before seeing D2 outcomes.

Scientific restraint:

- outcome-level DySample evidence may identify scale/localization-associated
  failure but cannot prove internal cross-scale-alignment causality;
- the SCConv-Early + Canonical EMA result must not be labeled causal module
  interference without a separate mechanism-level test;
- no replacement module is selected before the residual diagnostic is
  complete.

A12-D1 implementation transaction itself performs no dataset access,
checkpoint loading, validation inference, prediction, or training.

Split-B test access remains NONE.

A12-D2 remains unauthorized pending D1 source freeze and runtime-preflight
review.
## 2026-10-06 - A12-D1 runtime preflight PASS and evidence preservation

Type: DIAGNOSTIC / PROVENANCE

The source-frozen A12-D1 standardized validation diagnostic preflight was
executed on Kaggle against scientific source commit:

`e2367057e3a4ffabcb5cde1c2ff569df264625ef`

Runtime contract verified:

- Python 3.12.13;
- PyTorch 2.10.0+cu128;
- Ultralytics 8.4.7 imported from the governed checkout;
- Tesla T4 x2;
- clean governed worktree.

Dataset binding verified:

- `DATA01:B-ORG:v1`;
- frozen validation membership: 3,050 images;
- operational readable validation images: 3,049;
- validation patients: 914;
- runtime YAML contains no test key.

All six canonical `best.pt` checkpoints resolved exactly once and were loaded
only for structural verification of class count, class names, parameter count
and corrected-module signature.

No validation inference, prediction, training or Split-B test analysis was
performed.

Preserved evidence:

- `A12_D1_PREFLIGHT.json` SHA256:
  `d33ead0f712aa432e4afdd67aa89f6a4481acc543ae0486b8424a95761909de8`;
- `runtime_data_train_val_only.yaml` SHA256:
  `b2114397eefc1ac377352fb4e0429df0fe6fc954a307e4b0df89139b4bed82c4`;
- `A12_D1_RUNTIME_PREFLIGHT_PRESERVATION.json` SHA256:
  `217d67b401afbeca76033a7eef381649ef36e4f0f3b829c76b47216a8cc6039d`;
- `A12_D1_RUNTIME_PREFLIGHT_E2367057.zip` SHA256:
  `f9f47b7816caec3556fd1700aee79f8e716a8efa866bb9030e571a947b4d9de8`.

A12-D1 preflight status:

`PASS`

A12-D2 execution remains unauthorized pending a separate implementation,
source-freeze and authorization transaction.

Split-B test access remains NONE.
## 2026-10-06 - Governance chronology forward correction

Type: `PROVENANCE CORRECTION / GOVERNANCE`

A chronology audit identified that four governance events which occurred on
2026-10-05 had been recorded one calendar day late as 2026-10-06.

Corrected historical event dates:

- A12 corrected-module-family closure: 2026-10-06 -> 2026-10-05;
- permanent operating rules requested: 2026-10-06 -> 2026-10-05;
- repository simplicity and naming discipline: 2026-10-06 -> 2026-10-05;
- Rule-4 freeze verification stopped and forward-recovered:
  2026-10-06 -> 2026-10-05.

The operating-rules authority date was corrected consistently from
2026-10-06 to 2026-10-05.

The project-log historical cutoff was corrected from "before 2026-10-06" to
"before 2026-10-05" so the now-recorded 2026-10-05 events remain inside the
canonical log chronology.

Scope and scientific impact:

- this is a forward-only provenance correction;
- no existing Git commit was amended, reset or force-pushed;
- no experiment ID, architecture, metric, result or artifact identity changed;
- A12-D0 and A12-D1 events dated 2026-10-06 remain unchanged because they
  occurred on 2026-10-06;
- no dataset was accessed;
- no checkpoint was loaded;
- no validation inference or prediction occurred;
- no training occurred;
- Split-B test access remained NONE;
- A12-D2 remained unauthorized during this correction.

Next action:

`A12-D2_STANDARDIZED_VALIDATION_EXECUTION_IMPLEMENTATION_AND_FREEZE`
## 2026-10-06 - A12-D2 standardized validation execution candidate implemented

Type: `DIAGNOSTIC DESIGN / IMPLEMENTATION`

A12-D2 execution source was implemented locally after A12-D1 runtime-preflight
closure and the governance chronology correction.

The candidate runner is:

`research/runtime/a12_d2_standardized_validation.py`

The candidate static contract test is:

`research/tests/test_a12_d2_standardized_validation_static.py`

Frozen execution intent:

- evaluate the same six canonical Split-B Original pretrained checkpoints;
- use one current source-frozen checkout;
- reuse the D1-governed Python/PyTorch/Ultralytics/T4 runtime;
- use `DATA01:B-ORG:v1` validation only;
- preserve native validation plots and confusion matrices;
- export post-NMS predictions once with `save_txt=True`, `save_conf=True` at
  the frozen validation confidence floor;
- record aggregate P/R/F1/mAP50/mAP75/mAP50-95;
- record per-class P/R/F1/AP50/AP75/AP50-95 and support;
- preserve `foreignbody` as N/A because Split-B validation support is zero;
- preserve historical training-time selection metrics separately;
- defer all fixed-threshold error taxonomy, size, patient and transition
  analysis to later offline diagnostics using the frozen prediction exports.

Scientific safeguards:

- no module is selected from D2 aggregate score alone;
- no prediction threshold is tuned after seeing D2 outcomes;
- no causal architectural-mechanism claim is inferred from outcome metrics;
- no training is authorized;
- Split-B test remains sealed.

This implementation transaction itself performs no dataset access, checkpoint
loading, validation inference, prediction or training.

A12-D2 execution remains unauthorized pending source-freeze review.
## 2026-10-06 - A12-D2 candidate review hardening

Type: `DIAGNOSTIC DESIGN / GOVERNANCE HARDENING`

Review of the first local A12-D2 implementation candidate identified that the
execution runner should itself enforce the project's separate-authorization
rule rather than relying only on operator procedure.

The candidate was therefore hardened before source freeze.

Hardening decisions:

- require a separate source-bound
  `A12_D2_EXECUTION_AUTHORIZATION.json` before dataset access, checkpoint
  loading or validation inference;
- bind authorization to the exact frozen D2 runner SHA256 and source commit;
- verify D1 source-freeze, D1 closure and chronology commits are ancestors of
  the execution checkout;
- preserve a validation image index with authoritative patient IDs;
- preserve normalized ground truth, continuous normalized area and the frozen
  small/medium/large size bin;
- preserve a per-model `PREDICTIONS.csv` in addition to raw Ultralytics TXT
  prediction exports;
- preserve an explicit failure manifest for interrupted authorized execution;
- document native Ultralytics confusion matrices as native visualizations at
  effective confidence 0.25 / IoU 0.45, separate from the later frozen offline
  matching/error taxonomy;
- keep AP75 extraction from the ten-threshold AP matrix at index 5;
- keep all fixed confidence, IoU, size, patient and transition analyses outside
  D2 so they are performed later on frozen outputs without rerunning models.

This hardening transaction remains source-only:

- dataset access: NONE;
- checkpoint loading: NONE;
- validation inference: NONE;
- prediction: NONE;
- training: NONE;
- Split-B test access: NONE;
- A12-D2 execution authorization: FALSE.
## 2026-10-06 - A12-D2 authorization-status parser hardening after safe source-freeze stop

Type: `DIAGNOSTIC DESIGN / SAFE RECOVERY / GOVERNANCE HARDENING`

The first A12-D2 source-freeze attempt stopped safely before staging, commit or
push.

Cause:

- the source-freeze script used a raw substring guard for
  `A12-D2 execution authorization: TRUE`;
- the protocol already contained that exact phrase only as explanatory text
  describing a future authorization requirement;
- the guard therefore produced a false positive even though A12-D2 execution
  remained unauthorized.

Review also identified the same ambiguity in the R2 execution runner, whose
protocol authorization check used raw substring presence.

Recovery/hardening:

- restore the protocol status from the uncommitted partial source-freeze state
  to `A12_D2_IMPLEMENTATION_CANDIDATE_PENDING_SOURCE_FREEZE`;
- restore the candidate-stage FALSE authorization statement;
- rewrite the explanatory future-authorization bullet so it does not contain
  a machine authorization flag that can be confused with current state;
- make the D2 runner parse the exact Markdown `## Status` field and require it
  to equal `A12_D2_EXECUTION_AUTHORIZED`;
- retain the separate authorization JSON, source-commit binding and runner
  SHA256 checks;
- add static tests proving explanatory prose cannot satisfy the authorization
  gate.

The failed source-freeze attempt performed:

- staging: NONE;
- commit: NONE;
- push: NONE;
- dataset access: NONE;
- checkpoint loading: NONE;
- validation inference: NONE;
- prediction: NONE;
- training: NONE;
- Split-B test access: NONE.

A12-D2 execution authorization remains FALSE.

Next action:

`A12-D2_SOURCE_FREEZE_REVIEW_R3`
## 2026-10-06 - A12-D2 R3 standardized validation source freeze

Type: `DIAGNOSTIC SOURCE FREEZE / GOVERNANCE`

The R3 A12-D2 standardized validation execution candidate passed review after
the safe R2 source-freeze stop and authorization-status parser hardening.

Frozen runner:

`research/runtime/a12_d2_standardized_validation.py`

Frozen runner SHA256:

`9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250`

The source-freeze preserves these execution safeguards:

- execution requires a separate authorization JSON;
- authorization must bind the exact frozen source commit and runner SHA256;
- the runner parses the exact protocol `## Status` field;
- explanatory prose cannot satisfy the authorization gate;
- authorization is checked before output-root creation, dataset access,
  checkpoint discovery/loading and validation inference;
- D1 source-freeze, D1 closure and chronology commits remain required
  ancestors;
- Ultralytics/model source must remain unchanged relative to the frozen D1
  source;
- the six canonical checkpoints remain SHA-bound;
- validation remains `DATA01:B-ORG:v1` only;
- normalized validation ground truth, authoritative patient mapping,
  continuous box area, fixed size bins, raw prediction TXT, per-model
  `PREDICTIONS.csv`, aggregate/per-class metrics and artifact manifests remain
  part of the preservation contract;
- native Ultralytics confusion matrices remain visualization artifacts and do
  not replace the later frozen offline matching/error taxonomy;
- no offline threshold/error taxonomy analysis occurs inside D2;
- historical training-time selection metrics remain separate.

This source-freeze transaction performs:

- dataset access: NONE;
- checkpoint loading: NONE;
- validation inference: NONE;
- prediction: NONE;
- training: NONE;
- Split-B test access: NONE.

A12-D2 execution authorization remains FALSE.

Next action:

`A12-D2_EXECUTION_AUTHORIZATION_FREEZE`
