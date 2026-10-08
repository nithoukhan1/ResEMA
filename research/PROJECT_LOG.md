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
## 2026-10-06 - A12-D2 validation-only execution authorization freeze

Type: `EXECUTION AUTHORIZATION / GOVERNANCE`

The source-frozen A12-D2 R3 standardized validation runner was explicitly
authorized for one governed validation-only execution.

Authorization record:

`research/06_diagnostics/A12_D2_EXECUTION_AUTHORIZATION.json`

Authorization record SHA256:

`8cbb1e999d3ffb2e7e4b2e30d9882c5510c076d116468609962cd06e36b3dcb1`

Authorized source commit:

`758f0643e2bceb8d996d8e9603fb836d9319d8eb`

Authorized runner SHA256:

`9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250`

Authorized scope:

- six frozen Split-B Original pretrained checkpoints;
- `DATA01:B-ORG:v1`;
- validation split only;
- standardized validation inference;
- low-confidence post-NMS validation prediction export;
- validation metric/artifact generation.

Still forbidden:

- new training or weight updates;
- Split-B test access, prediction or metrics;
- architecture promotion from D2 aggregate score alone;
- threshold tuning inside D2;
- interpreting native confusion-matrix settings as the later offline error
  taxonomy.

This authorization-freeze transaction itself performs no dataset access,
checkpoint loading, validation inference, prediction or training.

Next action:

`A12-D2_KAGGLE_EXECUTION_GATE`
## 2026-10-06 - A12-D2D safe closure recovery and standardized validation closure

Type: `SAFE RECOVERY / DIAGNOSTIC PRESERVATION / CLOSURE`

The first A12-D2D closure attempt stopped safely before staging, commit or
push because the raw marker `A12_D2_EXECUTION_AUTHORIZED` occurred twice in
the protocol: once as the actual `## Status` and once in explanatory prose.

Before the stop, the intended partial local actions were completed:

- nine A12-D2 evidence rows were written to `ARTIFACTS.csv` (77 -> 86);
- `A12_D2_STANDARDIZED_VALIDATION_CLOSURE.json` was written with SHA256
  `aaecb31afc29fd36a53455e5cabaaf59b93fab6bb5dbb38e13eaa6b27ac9b034`.

The protocol itself was not changed by the failed replacement. Recovery
verified the exact two-file partial worktree state, verified remote HEAD
`791566cb04c91258eadbbb8ee6120edac9cc4c0f`, verified the nine A12-D2 registry rows, verified the closure
record and external evidence, then replaced only the exact Markdown `## Status`
value.

A12-D2 evidence:

- execution checkout: `791566cb04c91258eadbbb8ee6120edac9cc4c0f`;
- source commit: `758f0643e2bceb8d996d8e9603fb836d9319d8eb`;
- runner SHA256: `9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250`;
- authorization SHA256: `8cbb1e999d3ffb2e7e4b2e30d9882c5510c076d116468609962cd06e36b3dcb1`;
- preservation archive SHA256: `419ac71b1e42b791168a5ea24cf87d71b8a3d1eba9333d183a73009733056b93`;
- preservation review SHA256: `9805009b0fb757bcd3e84ef7ec28b471cced7da0b9a700d3ad56cac6932a84bd`;
- 18,407 / 18,407 artifact-manifest rows verified;
- 3,050 frozen validation images / 3,049 operational readable images;
- 914 validation patients;
- 7,113 frozen GT boxes / 7,110 operational diagnostic GT boxes;
- unreadable-image GT boxes: 3;
- offline diagnostics: NOT YET EXECUTED;
- new GPU training: NOT AUTHORIZED;
- Split-B test access: NONE.

The one governed D2 authorization is consumed by the completed execution.
Any D2 rerun requires a new explicit authorization transaction.

Next action:

`PROJECT_MIGRATION_AFTER_A12_D2_CLOSURE`

After migration:

`A12-D3_OFFLINE_DIAGNOSTIC_IMPLEMENTATION_AND_FREEZE`

## 2026-10-07 - A12-D3 offline diagnostic implementation source freeze

Type: `DIAGNOSTIC SOURCE FREEZE / GOVERNANCE`

A12-D3 offline diagnostic implementation was reviewed and source-frozen before any real D2 diagnostic execution.

Scientific source-freeze commit:

`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

Frozen implementation:
- global same-class candidate-pair matching ordered by descending IoU;
- one-to-one primary match threshold IoU >= 0.50;
- fixed confidence thresholds 0.05 / 0.10 / 0.25 / 0.50;
- fixed unmatched-prediction taxonomy priority;
- GT-size and FP-prediction-size analyses kept separate;
- confidence-distribution outputs frozen;
- eight predefined model-comparison contrasts frozen;
- paired patient-level changes frozen;
- patient bootstrap fixed at seed 42, 10,000 replicates and two-sided 95% percentile intervals.

Final pre-freeze synthetic review:
- engine: 13/13 PASS;
- runner: 20/20 PASS.

Firewalls during implementation/source freeze:
- preserved D2 archive read: NONE;
- checkpoint loading: NONE;
- validation inference/rerun: NONE;
- training: NONE;
- Split-B test access: NONE.

The source freeze does not authorize D3 execution.
A separate source-bound execution authorization is required before the runner may open the preserved D2 archive.

Source-freeze record:
`research/06_diagnostics/A12_D3_SOURCE_FREEZE.json`

Source-freeze record SHA256:
`24b34050764342d0ac2c2398ffdb80aedc59a4c151206765b203d01361ba4e19`

Next action:
`A12_D3_OFFLINE_DIAGNOSTIC_EXECUTION_AUTHORIZATION`

## 2026-10-07 - A12-D3 one offline diagnostic execution authorized

Type: `EXECUTION AUTHORIZATION / GOVERNANCE`

A12-D3 scientific source was already frozen at:

`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

One governed offline diagnostic execution is now authorized from the canonical preserved A12-D2 CSV evidence.

Authorization commit:

`fe342dd9da4c1a79746d43c3628fe4f628c97013`

Authorization record:
`research/06_diagnostics/A12_D3_EXECUTION_AUTHORIZATION.json`

Authorization JSON SHA256:
`c537450066515940fc3a49e10bea870bb3cfb9395ca144250efd411dac8ba600`

Execution gate:
`research/06_diagnostics/A12_D3_AUTHORIZED_EXECUTION_GATE.py`

Execution gate SHA256:
`5d8ccfc5fb1c967f36d32b7f673f3bfd48c569938a90aac3534010335bd6ebb8`

Frozen runner SHA256:
`7edae668a571f31274ca02be8f73d97ef49a1c0b4266c5f2a7a4464ece4116c4`

The authorization is limited to the predeclared offline diagnostic scope and does not authorize:
- checkpoint loading;
- model validation/inference;
- training;
- threshold tuning;
- Split-B test access;
- architecture promotion during execution.

The authorization transaction itself opened no D2 archive and produced no diagnostic result.

A separate authorization-preflight must PASS before the one execution attempt.

Next action:
`A12_D3_AUTHORIZATION_PREFLIGHT_THEN_ONE_OFFLINE_EXECUTION`

## 2026-10-07 - A12-D3 offline diagnostic execution preserved and closed

Type: `DIAGNOSTIC PRESERVATION / AUTHORIZATION CONSUMPTION / CLOSURE`

The source-frozen and separately authorized A12-D3 offline diagnostic executed successfully against the canonical preserved A12-D2 CSV evidence.

Execution checkout:
`d696330f2b2a8cbe0dfe0b6f3d7c9fe512f1a9ea`

Scientific source-freeze commit:
`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

Authorization commit:
`fe342dd9da4c1a79746d43c3628fe4f628c97013`

Independent post-execution audit verified:
- 58/58 output files;
- 9/9 summary tables;
- 48/48 object-event files;
- full TP/FP/FN and event-link conservation;
- 914-patient metric lattice;
- 8 predefined comparison lattice;
- 128 patient-bootstrap rows with seed 42 / 10,000 replicates;
- no checkpoint loading;
- no validation rerun;
- no training;
- Split-B test access NONE.

Canonical preservation archive SHA256:
`8c3545fd772b56984d87090819ba265bc98dbfdf15c456b4abb6f0a02fd513af`

Independent preservation review SHA256:
`7fb6038dbf6c3f6b0b82d06c5631a9fb3b0b96e1410bf338b0e758c63298a3c1`

Execution manifest SHA256:
`0c078833b0f3675d41cccce7bb09545267947064fa986fe756d51d8c7689ba9b`

Output-tree digest SHA256:
`701d0cd105f86525bb24e2847c6a6e907eaf2dca6923ad8e54f42d014e504ff7`

The one-use A12-D3 execution authorization is consumed by the successful execution.
Any rerun requires a new explicit authorization transaction.

No scientific interpretation or architecture-promotion decision was made inside execution/preservation.

Next action:
`A12_D3_E_SCIENTIFIC_INTERPRETATION_AND_RESIDUAL_NOVELTY_DECISION`

## 2026-10-07 - D4 LPQ clean method-development phase opened

Type: `SCIENTIFIC TRANSITION / CONTEXT FREEZE / PROVENANCE`

The governed D3 residual-error analysis has been carried forward into a new,
clean method-development phase.

Source:
- branch `research/combination-screen-01`;
- HEAD `227918fa1c2a496023e39a992bfaedd953bcc8f9`.

New branch:
`research/lpq-method-01`

D3-E1 evidence package was preserved under the external artifact store and registered:
- package SHA256 `d75afd61c0549c6ae805a57b364e926bc7750bfb46fec520e69002976b147cb9`;
- manifest SHA256 `34d79b264f7a26e051ab78947d59a3680afdf03356cd76ac47cf4d36460dd36e`;
- scientific report SHA256 `02abfeb20c59aa16e3a2251d2dd48d55ab3b2a914146adad7ea98d3003bc84db`.

Path B is selected.
SCConv-Early is a reference component, not the final architecture.
LPQ/DAR/GDS remain concepts only.

No model code changed.
No training occurred.
No validation inference occurred.
No Split-B test access occurred.

Next action:
`D4-B_LPQ_MATHEMATICAL_SPECIFICATION_NOVELTY_AND_PROMOTION_FREEZE`

## 2026-10-07 - D4-B1 prior-art collision gate opened redesign

Type: `NOVELTY GATE / SAFE REDESIGN`

A literature/code review was completed before LPQ implementation.

The original DAR/GDS sketch was found too close to established quality-estimation and
duplicate-ranking methods and is not promoted to implementation.

No model code changed.
No dataset access occurred.
No checkpoint loading occurred.
No validation inference occurred.
No training occurred.
Split-B test access remained NONE.

Next action:
`D4-B2_REVISED_METHOD_HYPOTHESIS_PRIOR_ART_AND_MATHEMATICAL_FREEZE`

## 2026-10-07 - D4-B2A multi-view feasibility audit preserved

Type: `READ-ONLY DATA FEASIBILITY / PRESERVATION`

Source HEAD:
`78b90280e9e4a889cf32e3709d21a40ce22dd66f`

Package SHA256:
`755d7d0cae382caec95495e1e8e2bad262b1e55a52277ca681aed87d1cc747b6`

Result:
`STRONG`

No images, object labels, checkpoints or test files were opened.
No inference or training occurred.
No architecture decision was made.

Next action:
`D4-B2B_MULTIVIEW_OBJECT_LEVEL_COMPLEMENTARITY_AND_ERROR_FEASIBILITY`

## 2026-10-07 - D4-B2B complementarity audit preserved

Type: `READ-ONLY MULTI-VIEW COMPLEMENTARITY / PRESERVATION`

Source HEAD:
`4963f8e81264cb42d2ef81c06aefe65e4ffe7fe2`

Package SHA256:
`21d738e9b269c791fcc953eeab6fb2c2e1252ca42255d1bd9c4b4e340c49e9aa`

Result:
`MODERATE`

Closure helper history:
- R1 safe-stopped during pre-mutation package verification because of an over-strict
  inferred floating-point precision check;
- R2 corrected verifier precision policy only;
- evidence package and scientific results were unchanged.

The audit used preserved validation GT/predictions and performed no new inference.
No raw images or raw label files were opened.
Split-B test remained sealed.

Next action:
`D4-B2C_MULTIVIEW_METHOD_NOVELTY_AND_MATHEMATICAL_DESIGN`

## 2026-10-07 - D4-B2C1 novelty landscape and candidate shortlist

Type: `METHOD DESIGN / PRIOR-ART NARROWING`

Generic dual-view fusion, uncertainty/conflict gating and mutual-distillation approaches
were reviewed and rejected as novelty cores.

VGRA was promoted to the next mathematical-design gate.

No model code changed.
No inference or training occurred.
Split-B test remained sealed.

Next action:
`D4-B2C2_VGRA_MATHEMATICAL_ARCHITECTURE_AND_PROMOTION_FREEZE`

## 2026-10-07 - D4-B2C2 VGRA mathematical candidate freeze

Type: `METHOD SPECIFICATION / NOVELTY BOUNDARY / PROMOTION CONTRACT`

VGRA candidate V1 equations, parameter budget, initialization, missing-view behavior,
ablation plan and first-screen promotion criteria were frozen before code implementation.

No model code changed in this transaction.
No inference or training occurred.
Split-B test remained sealed.

Next action:
`D4-C_VGRA_IMPLEMENTATION_SCAFFOLD_AND_UNIT_TESTS`

## 2026-10-08 - D4-C0 VGRA implementation execution plan
Type: `IMPLEMENTATION GOVERNANCE / WORKTREE ISOLATION`. Source design `research/lpq-method-01 @ a5d260ca83236c774e1920ce729e08eaacfcf5d4`. Implementation branch `research/vgra-impl-01`. Documentation/repository isolation only; no model/data/training code changed. Next: `D4-C1_VGRA_CORE_MATH_IMPLEMENTATION`.

## 2026-10-08 - D4-C1 VGRA mathematical core

Type: `IMPLEMENTATION / CPU UNIT VERIFICATION`

Added standalone VGRA core and focused tests; re-ran existing TPSC tests.

R1 history:
- source compilation PASS;
- 17 focused CPU tests PASS;
- safe stop during documentation tracker update due to exact wording mismatch;
- pre-commit rollback clean;
- R2 corrects tracker matching only.

No data, inference, GPU training or B-test access occurred.

Next:
`D4-C2_VGRA_PAIR_AND_VISIBILITY_MANIFEST`

## 2026-10-08 - D4-C2 VGRA pair/visibility manifest

Type: `DATA CONTRACT / METADATA-ONLY`

Generated reproducible TRAIN/VAL pair and image-accounting manifests.
Frozen the four-state detection-label target schema without reading labels.

No raw images, YOLO labels, inference, training or B-test content was accessed.

Next:
`D4-C3_VGRA_HEAD_MODEL_INTEGRATION`

## 2026-10-08 - D4-C3 VGRA head/model integration

Type: `MODEL INTEGRATION / CPU VERIFICATION`

Added VGRADetect, parser registration and VGRA candidate YAML.
Focused tests verify native single-view fallback, zero-residual identity, active-residual
classification change, exact box invariance, full YAML parsing and parameter budget.

No data loader, training, inference experiment or B-test access occurred.

Next:
`D4-C4_VGRA_PAIR_AWARE_DATA_AND_BATCHING`

## 2026-10-08 - D4-C4 VGRA pair-aware data/batching

Type:
`DATA PIPELINE IMPLEMENTATION / METADATA-ONLY AUDIT`

Implemented VGRA-specific dataset, pair sampler, collate metadata and pair-safe loader
builder without changing stock dataset/build code.

R1 compiled and passed all 38 focused CPU tests but safely stopped before
commit because direct launching of the independent audit script did not resolve
the local `ultralytics.data.vgra` import. Pre-commit rollback was clean.

R2 changes only audit launch/import-path verification and documents the R1 failure;
the VGRA module, tests and audit implementation are unchanged.

Full frozen assignment manifest was audited for pair co-batching and exact image
accounting.

No image, YOLO label, training, inference experiment or B-test access occurred.

Next:
`D4-C5_VGRA_CRITERION_TRAINER_VALIDATOR_INTEGRATION`

## 2026-10-08 - D4-C5A VGRA visibility supervision

Type: `METHOD INTEGRATION / SYNTHETIC OBJECTIVE TESTS`

Implemented target mapping, strict companion validation, TRAIN-only visibility
counting/weight utility, and weighted CE. No real images or label files opened.
No GPU training, B-TEST access, stock loss or trainer/validator modifications.

Recorded the signed-beta semantic-inversion limitation as an OPEN pre-training gate.

Next:
`D4-C5B_VGRA_PAIRED_RUNTIME_AND_LOSS_INTEGRATION`

## 2026-10-08 - D4-C5B paired runtime/native loss

Type: `VGRA RUNTIME / SYNTHETIC CPU LOSS INTEGRATION`

Added a dedicated mixed-batch VGRA DetectionModel subclass and additive criterion.
Reused untouched native YOLO box/DFL and detection loss.
Confirmed fork-specific four-component loss behavior and exact singles fallback.

No stock source changes, real data access, GPU training or B-TEST access.

Next: `D4-C5C_VGRA_TRAINER_VALIDATOR_INTEGRATION`

## 2026-10-08 - D4-C5C trainer validator CPU verification

Type: `PAIR-AWARE TRAINER / VALIDATOR / SYNTHETIC CPU TESTS`

Implemented dedicated VGRA trainer and validator without stock source changes.
R1 safe-stop: 66 tests passed, two failed (all-zero-input non-finite
backward gradient and incorrect indexing of the eval-mode native tuple).
Pre-commit rollback clean and no commit created. R2 changes test input to
fixed-seed nonconstant synthetic pixels, corrects eval raw-output indexing,
and preserves strict gradient- and parameter-finiteness requirements.
The all-zero-input gradient case remains open for D4-D3 diagnosis.

Verifies synthetic mixed-view optimizer step, native box firewall,
paired validation inference/metrics, four losses and fail-closed guards.

No real images, YOLO label files, GPU experiment or B-TEST access.
Next: `D4-C6_VGRA_IMPLEMENTATION_CLOSURE`

## 2026-10-08 - D4-C6 VGRA implementation closure

Type: `SOURCE / GATE / FAILURE-HISTORY INVENTORY`

Created reproducible C0-C5C audit, full SHA256 inventory, formal closure,
D4-D gate checklist, gate record, and updated trackers.
Re-ran full focused synthetic CPU regression suite.
No raw data, YOLO labels, GPU training, or B-TEST access.

Next: `D4-D1_VGRA_INDEPENDENT_STATIC_UNIT_VERIFICATION`

## 2026-10-08 - D4-D1R confirmed findings and patch

Independent audit confirmed two production-path integration issues.
Corrected under strict source allowlist, ran full focused CPU regression and documented both.
No patient images, YOLO labels, GPU training or B-TEST read.

## D4-D1 formal independent acceptance — 2026-10-08

Type: `INDEPENDENT EVIDENCE REPLAY / STATIC CLOSURE`

Independent D1B replay: PASS; original two D1 blockers verified repaired.
75 focused CPU tests: PASS. Frozen VAL: 3,049 images,
1,401 pairs and 247 singles; 191 pair-preserving batches at size 16.
Full read-only audit output and SHA256 preserved in the D1 acceptance record.
No raw images/labels, GPU experiments or sealed TEST accessed.
D4-D2 transfer verification is next; D4-D3 remains gated.

Next: `D4-D2_VGRA_PRETRAINED_TRANSFER_AND_ZERO_GATE_IDENTITY`

## D4-D2 formal pretrained-transfer acceptance — 2026-10-08

D4-D2A/B/C rerun as immutable SHA-pinned independent synthetic CPU checks.
Canonical pretrained SCConv-Early checkpoint matched its frozen archive;
537/537 Early tensors transferred unchanged; zero-rho class/box/DFL identity
passed in TRAIN and EVAL, and synthetic TRAIN buffers/EMA/best-checkpoint
restoration plus gradient reachability passed.

- Evidence: `research/07_method/D4_D2_INDEPENDENT_REPLAY_EVIDENCE.txt` (SHA256 `ee23f2f7ae0bcb9775bf9b7e36aef8b77b911a926ea324233cca7c8988b2ee83`).
- Formal decision: `research/07_method/D4_D2_FORMAL_ACCEPTANCE.md`; gate: `research/07_method/D4_D2_GATE_RECORD.json`.
- D4-D3 all-zero gradient numerical issue, persistent dataloader/complete
  synthetic epoch, signed-beta interpretation and MV00/MV01 recipe parity: OPEN.
- No GPU training, source freeze, model architecture change or B-TEST access.

Next: `D4-D3_VGRA_INTEGRATED_SYNTHETIC_NUMERICAL_VERIFICATION`
