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
