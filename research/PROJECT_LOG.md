# Detection Project Log

This is an append-only chronological log of substantive project events.

It complements:

- `CURRENT.md` for active state;
- `DECISIONS.md` for accepted project-level decisions;
- experiment registries for experiment state;
- artifact registries for external evidence.

The log may contain proposals, hypotheses, failures and recoveries.
Those entries are not automatically accepted scientific conclusions.

For history before 2026-10-06, use the frozen migration/handoff packages,
historical registries and decision records.

## 2026-10-06 ? A12 corrected-module-family closure

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

## 2026-10-06 ? Permanent operating rules requested

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

## 2026-10-06 - Repository simplicity and naming discipline

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

## 2026-10-06 - Rule-4 freeze verification stopped and forward-recovered

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
