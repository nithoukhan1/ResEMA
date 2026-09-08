# 03 — Master Execution Plan

## Phase 0 — Research-state consolidation
### Objective
Create one authoritative repository state containing all historical evidence and all future tracking infrastructure.

### Local tasks
- create publication branch from pinned commit;
- add `research/` structure;
- import B0–C2 summaries, hashes and decisions;
- import literature library and curated matrix;
- freeze environment notes and data provenance;
- add templates and trackers;
- add lightweight research-specific tests/CI.

### Training
None.

### Exit criteria
A new researcher can determine:
- what has been done;
- what code produced it;
- where the weights live;
- why each branch was accepted/rejected.

Estimated time: 2–3 working days.

---

## Phase 1 — D00 diagnostic audit
### Objective
Measure actual failure mechanisms before implementing the new method.

No full training.

### Work packages
D00-A transfer coverage
- exact state-dict tensor and parameter transfer from YOLO11s.pt to A0/A1/A3
- per-layer unmatched tensors
- total parameter-coverage percentage

D00-B localization
- AP50, AP75 and higher-IoU profile where technically valid
- matched-IoU distribution
- localization-error analysis

D00-C object geometry
- normalized area
- aspect ratio
- class x size
- rare geometry bins

D00-D metadata/view
- exact projection-code mapping from authoritative metadata
- AP/PA/LAT/other counts by train/val/test
- performance by projection
- verify axis annotation availability and parser

D00-E patient sensitivity
- positive-patient counts by class
- rare patient contribution
- leave-one-positive-patient-out sensitivity for bonelesion/boneanomaly/softtissue

D00-F paired-study feasibility
- number of studies with usable AP/LAT pairs
- pair completeness by split/class
- whether paired-view branch is viable

### Exit criteria
`DIAGNOSTIC_CONCLUSIONS.md` ranks bottlenecks and explicitly selects implementation order.

Estimated time: 3–5 days.

---

## Phase 2 — TP-CDA implementation and screening
### Local implementation
Branch: `feat/tp-cda`
- module
- parser/YAML support
- unit tests
- identity-at-init test
- parameter budget test
- CPU forward smoke test
- pretrained transfer audit

### Kaggle
M01: baseline + TP-CDA, one screening seed.

### Gate
Pass if:
- overall mAP50-95 improves >=0.5 pp, OR
- prespecified detail/localization metric improves >=1.0 pp with overall strict AP loss <=0.2 pp;
- no meaningful fracture collapse;
- effect is not explained by one rare patient.

Estimated: 3–5 implementation days + training/resume window.

---

## Phase 3 — PELT implementation and screening
### Local implementation
Branch: `train/pelt`
- patient-index builder
- positive-patient class statistics
- capped sampler
- deterministic tests
- fixed epoch length
- exposure logging by class/patient

### Kaggle
M02: baseline + PELT, one seed.

### Gate
Pass if:
- prespecified tail metric improves materially;
- overall strict AP does not decrease >0.2 pp;
- fracture remains stable;
- exposure audit confirms no single patient dominates.

Estimated: 3–5 days.

---

## Phase 4 — APCF implementation and screening
### Local implementation
Branch: `feat/apcf`
- metadata parser
- axis target parser
- tiny axis head
- projection embedding
- conditioned neck gates
- identity initialization
- unknown-projection fallback
- model forward tests

### Kaggle
M03: baseline + APCF, one seed.

### Gate
Pass if:
- overall strict AP improves >=0.5 pp, OR
- projection-stratified metric improves >=1.0 pp with overall loss <=0.2 pp;
- axis prediction is stable;
- AP/PA and LAT groups show no pathological tradeoff.

Estimated: 5–8 days.

---

## Phase 5 — Combine only passing mechanisms
No failed mechanism is included for paper symmetry.

Expected experiments:
- M10 best pair
- M11 second pair only if justified
- M12 full method only if components earned their place

Maximum: 2–3 full runs.

Exit: one preferred candidate and at most one runner-up.

---

## Phase 6 — Second-seed confirmation
Repeat:
- final candidate at seed43
- runner-up only if ranking is ambiguous

Expected: 1–2 runs.

---

## Phase 7 — Optional paired-view branch
Only if D00 shows:
- enough AP/LAT pairs;
- complementary errors;
- clean matched-cohort design.

Maximum development budget: 3–5 runs.

---

## Phase 8 — Method lock
Freeze:
- source
- model config
- sampler/loss
- metadata handling
- training recipe
- selection rule
- evaluation script

Create release-candidate tag.

---

## Phase 9 — 3-fold grouped robustness
Train:
- baseline folds 1–3
- proposed method folds 1–3

6 runs.

Report:
- fold values
- mean ± SD
- paired fold deltas
- class-wise robustness
- confidence intervals where appropriate

---

## Phase 10 — Final sealed Split-B test
Evaluate exactly:
- frozen baseline
- frozen proposed model

No redesign after test.

Use patient-level bootstrap uncertainty where feasible.

---

## Phase 11 — External/comparability and efficiency
After method lock:
- FracAtlas external fracture evaluation if mapping is valid
- optional literature-comparable GRAZ split
- parameters/GFLOPs
- batch-1 latency after warmup
- memory
- failure cases

---

## Phase 12 — Manuscript
Assemble:
- split/data audit
- baseline calibration
- failed historical modules
- new method rationale
- ablations
- seed robustness
- grouped CV
- sealed test
- external evidence
- uncertainty
- efficiency
- limitations
- CLAIM checklist
