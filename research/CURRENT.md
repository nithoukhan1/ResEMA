# Current Project State

## Active branch

`research/combination-screen-01`

## Active phase

`CORRECTED MODULE FAMILY CLOSED -> RESIDUAL-ERROR + NOVELTY DECISION GATE`

## Frozen baseline reference

Primary controlled pretrained reference:

`BASE-B-ORG-PT-S42`

- validation mAP50-95: 0.41704
- test access: NONE

## Corrected module-family results

| Candidate | validation mAP50-95 | Delta vs baseline |
|---|---:|---:|
| SCConv-Early | 0.43130 | +0.01426 |
| SCConv-4Stage | 0.42587 | +0.00883 |
| DySample | 0.41525 | -0.00179 |
| Canonical EMA | 0.42796 | +0.01092 |

Family winner:

`YOLO11s + SCConv-Early`

## Combination result

`YOLO11s + SCConv-Early + Canonical EMA`

- best epoch: 47
- precision: 0.61407
- recall: 0.63296
- F1: 0.6233719272
- mAP50: 0.64357
- mAP50-95: 0.41839
- delta vs baseline: +0.00135
- delta vs SCConv-Early: -0.01291
- strict complementarity: FALSE
- decision: DO_NOT_PROMOTE_COMBINATION_PRIMARY_METRIC

Execution archive:

`research/experiment-execution-archive`

Archive HEAD:

`3acd95ea046b9027c857f9068bb276ef4b2aa0ee`

## Controlling scientific decision

`CORRECTED_MODULE_FAMILY_SCREEN_CLOSED=TRUE`

`SELECTED_FAMILY_CANDIDATE=YOLO11s + SCConv-Early`

`GLOBAL_FINAL_PAPER_ARCHITECTURE_FROZEN=FALSE`

`REPLACEMENT_RESEARCH_PENDING_RESIDUAL_DIAGNOSIS=TRUE`

`NEW_GPU_TRAINING_AUTHORIZED=FALSE`

`TEST_ACCESS=NONE`

Canonical EMA remains an individually positive corrected ablation.
The combination result does not establish a causal explanation for its
negative interaction with SCConv-Early.

## Current task

A12-D1 standardized validation diagnostic preflight is complete and preserved.

Source-frozen D1 implementation:

`e2367057e3a4ffabcb5cde1c2ff569df264625ef`

Runtime preflight result:

- status: PASS;
- six canonical `best.pt` checkpoints resolved exactly once;
- all six checkpoints loaded structurally without validation inference;
- corrected-module signatures and parameter counts verified;
- data binding: `DATA01:B-ORG:v1`;
- frozen validation membership: 3,050 images / 914 patients;
- operational readable validation images: 3,049;
- runtime YAML contains train and val only;
- validation inference: NONE;
- prediction: NONE;
- training: NONE;
- Split-B test access: NONE.

Preserved runtime-preflight evidence:

- `A12_D1_PREFLIGHT.json` SHA256:
  `d33ead0f712aa432e4afdd67aa89f6a4481acc543ae0486b8424a95761909de8`;
- runtime YAML SHA256:
  `b2114397eefc1ac377352fb4e0429df0fe6fc954a307e4b0df89139b4bed82c4`;
- preservation manifest SHA256:
  `217d67b401afbeca76033a7eef381649ef36e4f0f3b829c76b47216a8cc6039d`;
- preservation ZIP SHA256:
  `f9f47b7816caec3556fd1700aee79f8e716a8efa866bb9030e571a947b4d9de8`.

Next:

`PROJECT_MIGRATION_AFTER_A12_D2_CLOSURE`

A12-D2 standardized six-model validation is complete, fully preserved,
registered and closed.

Governed evidence:

- execution checkout: `791566cb04c91258eadbbb8ee6120edac9cc4c0f`;
- frozen source commit: `758f0643e2bceb8d996d8e9603fb836d9319d8eb`;
- frozen runner SHA256: `9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250`;
- authorization SHA256: `8cbb1e999d3ffb2e7e4b2e30d9882c5510c076d116468609962cd06e36b3dcb1`;
- canonical preservation archive SHA256:
  `419ac71b1e42b791168a5ea24cf87d71b8a3d1eba9333d183a73009733056b93`;
- preservation review SHA256:
  `9805009b0fb757bcd3e84ef7ec28b471cced7da0b9a700d3ad56cac6932a84bd`;
- global artifact manifest: 18,407 verified rows;
- data binding: `DATA01:B-ORG:v1`;
- frozen validation: 3,050 images / 914 patients;
- operational validation: 3,049 readable images;
- GT boxes: 7,113 frozen / 7,110 operational / 3 on unreadable image;
- training: NONE;
- Split-B test access: NONE.

Standardized diagnostic revalidation mAP50-95:

- Baseline: 0.41765881
- SCConv-Early: 0.42588415
- SCConv-4Stage: 0.42452888
- DySample: 0.41427802
- Canonical EMA: 0.42663323
- SCConv-Early + Canonical EMA: 0.41308765

This standardized diagnostic metric family does not replace the historical
training-time selection metric family and does not by itself change the
selected family candidate.

The one governed D2 authorization is consumed by the completed execution.
Any rerun requires a new explicit authorization transaction.

Offline diagnostics have not yet been executed.
New GPU training remains unauthorized.
Split-B test remains sealed.

After the migration/archive handoff, the next scientific transaction is:

`A12-D3_OFFLINE_DIAGNOSTIC_IMPLEMENTATION_AND_FREEZE`

Required comparison:

- frozen YOLO11s baseline;
- SCConv-Early;
- SCConv-4Stage as a placement/intensity control;
- DySample;
- Canonical EMA;
- SCConv-Early + Canonical EMA.

Required questions:

- what errors does SCConv-Early fix?
- why does SCConv-4Stage change the precision-recall behavior?
- why does DySample underperform the baseline in this frozen condition?
- does DySample fail by class, object size, confidence, localization,
  cross-scale alignment, or because upsampling is not the bottleneck?
- what errors does Canonical EMA fix?
- which fixes disappear in the SCConv-Early + Canonical EMA combination?
- what residual errors remain after SCConv-Early?
- should the next mechanism target upsampling, feature refinement,
  localization, rare-class behavior, or another demonstrated bottleneck?

Do not choose a replacement module before this diagnostic.

Do not start new GPU training.

Split-B test remains sealed.

## A12-D3 offline diagnostic source freeze — 2026-10-07

Status:

`SOURCE_FROZEN_PENDING_EXECUTION_AUTHORIZATION`

Scientific source-freeze commit:

`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

Frozen D3 implementation:
- primary matcher: global same-class prediction/GT candidate pairs ordered by descending IoU;
- primary match threshold: IoU >= 0.50;
- frozen confidence thresholds: 0.05 / 0.10 / 0.25 / 0.50;
- frozen unmatched-prediction taxonomy: duplicate -> class_confusion -> localization -> background -> other_overlap;
- frozen comparison lattice: 8 predefined contrasts;
- patient bootstrap: seed 42 / 10,000 replicates / two-sided 95% percentile interval;
- engine synthetic regressions: 13/13 PASS;
- runner synthetic regressions: 20/20 PASS;
- D2 preservation archive read during implementation/source freeze: NONE;
- checkpoint loading: NONE;
- validation rerun: NONE;
- training: NONE;
- Split-B test access: NONE.

Source-freeze record:

`research/06_diagnostics/A12_D3_SOURCE_FREEZE.json`

SHA256:

`24b34050764342d0ac2c2398ffdb80aedc59a4c151206765b203d01361ba4e19`

Execution is **not authorized** by this source freeze.

Next governed transaction:

`A12_D3_OFFLINE_DIAGNOSTIC_EXECUTION_AUTHORIZATION`

## A12-D3 offline diagnostic execution authorization — 2026-10-07

Status:

`AUTHORIZED_PENDING_ONE_OFFLINE_EXECUTION`

Scientific source-freeze commit:

`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

Authorization commit:

`fe342dd9da4c1a79746d43c3628fe4f628c97013`

Frozen runner SHA256:

`7edae668a571f31274ca02be8f73d97ef49a1c0b4266c5f2a7a4464ece4116c4`

Execution gate SHA256:

`5d8ccfc5fb1c967f36d32b7f673f3bfd48c569938a90aac3534010335bd6ebb8`

Authorization JSON SHA256:

`c537450066515940fc3a49e10bea870bb3cfb9395ca144250efd411dac8ba600`

Authorized scope:
- one governed offline diagnostic execution from the preserved A12-D2 CSV evidence;
- six frozen models;
- four frozen confidence thresholds;
- global same-class candidate-pair IoU-descending matcher;
- eight predefined comparisons;
- paired patient bootstrap already source-frozen.

Still forbidden:
- checkpoint loading;
- validation rerun/inference;
- training;
- threshold tuning;
- architecture selection during execution;
- Split-B test access.

No D2 archive was opened during this authorization transaction.

Next:
`A12_D3_AUTHORIZATION_PREFLIGHT_THEN_ONE_OFFLINE_EXECUTION`

## A12-D3 offline diagnostic execution closure — 2026-10-07

Status:

`EXECUTION_COMPLETE_PRESERVED_AUTHORIZATION_CONSUMED`

Execution checkout:

`d696330f2b2a8cbe0dfe0b6f3d7c9fe512f1a9ea`

Scientific source-freeze commit:

`8d3cfee47bb4de37b98eb78964f58a953ba4fd36`

Authorization commit:

`fe342dd9da4c1a79746d43c3628fe4f628c97013`

Governed preservation:
- execution manifest SHA256: `0c078833b0f3675d41cccce7bb09545267947064fa986fe756d51d8c7689ba9b`;
- output-tree digest SHA256: `701d0cd105f86525bb24e2847c6a6e907eaf2dca6923ad8e54f42d014e504ff7`;
- preservation archive SHA256: `8c3545fd772b56984d87090819ba265bc98dbfdf15c456b4abb6f0a02fd513af`;
- preservation review SHA256: `7fb6038dbf6c3f6b0b82d06c5631a9fb3b0b96e1410bf338b0e758c63298a3c1`;
- output files verified: 58/58;
- summary tables verified: 9/9;
- object-event files verified: 48/48;
- patients: 914;
- comparisons: 8;
- patient-bootstrap rows: 128.

Execution firewalls:
- checkpoint loading: NONE;
- validation rerun: NONE;
- training: NONE;
- Split-B test access: NONE.

The one governed A12-D3 execution authorization is consumed.
`D3_RERUN_AUTHORIZED=FALSE`.

No scientific model-selection interpretation was performed during execution or preservation.

New GPU training remains unauthorized.
Split-B test remains sealed.

Next scientific transaction:

`A12_D3_E_SCIENTIFIC_INTERPRETATION_AND_RESIDUAL_NOVELTY_DECISION`

## D4 LPQ method-development transition — 2026-10-07

Status:

`PATH_B_SELECTED_LPQ_CONCEPT_PHASE`

Authoritative D4 branch:

`research/lpq-method-01`

D4 branch parent:

`227918fa1c2a496023e39a992bfaedd953bcc8f9`

D3-E2 decision:
- SCConv-Early remains a useful corrected-family reference but is not frozen as the final paper architecture;
- broad SCConv/DySample/EMA module-combination search is closed;
- the next method is problem-driven around true-positive preservation, score/reliability consistency and duplicate control.

Working concept:
- LPQ = Lesion-Preserving Quality;
- DAR = Distribution-Aware Reliability;
- GDS = Groupwise Duplicate Separation.

Current firewalls:
- LPQ implemented: FALSE;
- LPQ training authorized: FALSE;
- attention authorized: FALSE;
- final architecture frozen: FALSE;
- B-VAL: development/selection;
- B-TEST access: NONE.

D4 canonical starting document:

`research/07_method/D4_LPQ_CONTEXT_INDEX.md`

Next:

`D4-B_LPQ_MATHEMATICAL_SPECIFICATION_NOVELTY_AND_PROMOTION_FREEZE`

## D4-B1 prior-art collision gate — 2026-10-07

Status:

`PRIOR_ART_COLLISION_FOUND_REDESIGN_REQUIRED`

D4-A is complete and remote-verified.

Novelty review found:
- original DAR overlaps materially with GFLV2 distribution-guided quality prediction;
- generic localization-aware score fusion overlaps with GFL/VFNet/TOOD/TAL;
- original GDS best-vs-secondary score-gap concept overlaps materially with
  hybrid/dual-assignment and ranking-based duplicate suppression.

Therefore:
- D3 residual problem: RETAINED;
- LPQ working name: temporary only;
- original DAR novelty core: REJECTED;
- original GDS novelty core: REJECTED;
- implementation authorized: FALSE;
- training authorized: FALSE;
- attention authorized: FALSE;
- final architecture frozen: FALSE;
- B-test access: NONE.

Next:
`D4-B2_REVISED_METHOD_HYPOTHESIS_PRIOR_ART_AND_MATHEMATICAL_FREEZE`

## D4-B2A multi-view feasibility — STRONG — 2026-10-07

Read-only TRAIN/VAL metadata audit completed and preserved.

- TRAIN pair-capable group fraction: 0.85665349
- TRAIN patient pair fraction: 0.93761726
- VAL pair-capable group fraction: 0.86436498
- VAL patient pair fraction: 0.93763676

Disposition:
`MULTIVIEW_FEASIBILITY_TIER=STRONG`

This does not yet prove annotation/model-error complementarity.
Architecture decision remains FALSE.

Next:
`D4-B2B_MULTIVIEW_OBJECT_LEVEL_COMPLEMENTARITY_AND_ERROR_FEASIBILITY`

Firewalls:
- model code changed: FALSE;
- training authorized: FALSE;
- attention authorized: FALSE;
- B-test access: NONE.

## D4-B2B complementarity — MODERATE — 2026-10-07

Read-only paired-view annotation/model-error audit completed and preserved.

- annotation view-exclusive fraction: 0.11481347
- baseline one-view-only TP fraction @0.25: 0.03727980
- Early one-view-only TP fraction @0.25: 0.04342483
- fracture baseline one-view-only TP fraction @0.25: 0.05929304
- fracture Early one-view-only TP fraction @0.25: 0.07411631

Disposition:
`COMPLEMENTARITY_EVIDENCE_TIER=MODERATE`

Multi-view remains a serious candidate, but architecture selection remains FALSE.
The next design should focus on selective/conditional assistance and must pass a fresh
prior-art review.

Next:
`D4-B2C_MULTIVIEW_METHOD_NOVELTY_AND_MATHEMATICAL_DESIGN`

Firewalls:
- model code changed: FALSE;
- new inference: NONE;
- training authorized: FALSE;
- B-test access: NONE.

## D4-B2C1 multi-view novelty landscape — 2026-10-07

Status:

`VGRA_WORKING_CANDIDATE_SELECTED_FOR_MATHEMATICAL_DESIGN`

The following are rejected as D4 novelty cores:
- generic dual-view fusion;
- generic uncertainty/conflict gating;
- generic multi-view distillation.

Working candidate:
`Visibility-State-Gated Cross-View Residual Assistance (VGRA)`

VGRA is not frozen and no novelty claim is made yet.

Current firewalls:
- architecture frozen: FALSE;
- implementation authorized: FALSE;
- GPU training authorized: FALSE;
- attention authorized: FALSE;
- B-test access: NONE.

Next:
`D4-B2C2_VGRA_MATHEMATICAL_ARCHITECTURE_AND_PROMOTION_FREEZE`

## D4-B2C2 VGRA candidate V1 frozen — 2026-10-07

Status:

`VGRA_CANDIDATE_V1_SPEC_FROZEN_FOR_IMPLEMENTATION`

Working novelty status:
`PLAUSIBLE_CANDIDATE_NOT_EMPIRICALLY_VALIDATED`

Frozen candidate constants:
- visibility states: 00 / AP-only / LAT-only / both;
- r_d=32;
- pair hidden=128;
- cross-view rank=16;
- beta_max=2.0;
- rho init=0;
- lambda_vis=0.25;
- added parameter cap=300,000.

Implementation authorized: TRUE.
GPU training authorized: FALSE.
Attention authorized: FALSE.
Global final architecture frozen: FALSE.
B-test access: NONE.

Next:
`D4-C_VGRA_IMPLEMENTATION_SCAFFOLD_AND_UNIT_TESTS`

## D4-C0 authoritative implementation state — 2026-10-08
Active implementation branch: `research/vgra-impl-01`. Parent design authority: `research/lpq-method-01 @ a5d260ca83236c774e1920ce729e08eaacfcf5d4`. VGRA V1 is frozen; implementation authorized; GPU training not authorized; B-test NONE. Next: `D4-C1_VGRA_CORE_MATH_IMPLEMENTATION`.

## D4-C1 VGRA core implementation — 2026-10-08

Status: `COMPLETE / CPU-UNIT-VERIFIED`
Core source: `ultralytics/nn/modules/vgra.py`
Reference core parameters: `102567`

Head/data/trainer integration: FALSE.
GPU training authorized: FALSE.
B-test access: NONE.

Next:
`D4-C2_VGRA_PAIR_AND_VISIBILITY_MANIFEST`

## D4-C2 VGRA pairing manifest — 2026-10-08

Status: `COMPLETE / DETERMINISTIC / METADATA-ONLY`

TRAIN exact AP/LAT pairs: `6496`
VAL exact AP/LAT pairs: `1402`

Visibility state schema is frozen, but actual state values remain deferred to B-TRAIN
YOLO-label processing in D4-C5.

No images, YOLO labels or B-test files were opened.
GPU training remains unauthorized.

Next:
`D4-C3_VGRA_HEAD_MODEL_INTEGRATION`

## D4-C3 VGRA head/model integration — 2026-10-08

Status: `COMPLETE / CPU-VERIFIED`

VGRA-specific head:
`VGRADetect`

Full candidate parameter count:
`9672660`

Box/DFL cross-view modification:
`FALSE`

Pair-aware data/trainer integration:
`FALSE`

GPU training:
`NOT AUTHORIZED`

B-test:
`NONE`

Next:
`D4-C4_VGRA_PAIR_AWARE_DATA_AND_BATCHING`

## D4-C4 VGRA pair-aware data/batching — 2026-10-08

Status:
`COMPLETE / CPU-VERIFIED / METADATA-AUDITED`

Train pair units: `6496`
Train singles: `1235`
Val usable pair units: `1401`
Val singles/fallbacks: `247`

Mosaic/MixUp/CutMix/Copy-Paste:
`DISABLED FOR PAIR-AWARE PIPELINE`

Pair-aware rect validation:
`FALSE`

GPU training:
`NOT AUTHORIZED`

B-test:
`NONE`

Next:
`D4-C5_VGRA_CRITERION_TRAINER_VALIDATOR_INTEGRATION`

## D4-C5A visibility objective — 2026-10-08

Status: `COMPLETE / SYNTHETIC_CPU_VERIFIED`

Target source: collated nine-class YOLO GT.
State weighting source: B-TRAIN exact-pair labels only, when bound.
Actual TRAIN state counts: not yet computed.
Signed-beta interpretation issue: OPEN pre-training gate.

GPU training remains unauthorized. B-TEST remains sealed.

Next:
`D4-C5B_VGRA_PAIRED_RUNTIME_AND_LOSS_INTEGRATION`

## D4-C5B mixed runtime/loss — 2026-10-08

Status: `COMPLETE / SYNTHETIC_CPU_VERIFIED`

Paired/single dispatch: implemented.
Native box/DFL: untouched.
Native loss vector: preserved.
Visibility: fourth loss component weighted by 2 * valid pairs.
Trainer and validator: NOT integrated.
Actual TRAIN visibility counts/weights: NOT computed.
Signed-beta interpretation gate: OPEN.
GPU training: NOT AUTHORIZED.
B-TEST: SEALED.

Next: `D4-C5C_VGRA_TRAINER_VALIDATOR_INTEGRATION`

## D4-C5C paired trainer and validator — 2026-10-08

Status: `COMPLETE / SYNTHETIC CPU VERIFIED`

Trainer: `VGRADetectionTrainer`
Validator: `VGRADetectionValidator`

Pair-aware VAL predicted scores and native YOLO NMS/metrics: synthetic-verified.
B-TRAIN real weights not yet bound/generated in an actual dataset run.
Signed-beta semantic gate: OPEN.
GPU training: NOT AUTHORIZED.
B-TEST access: NONE.

Next: `D4-C6_VGRA_IMPLEMENTATION_CLOSURE`

## D4-C6 VGRA implementation complete

Status: `IMPLEMENTATION COMPLETE / INDEPENDENT VERIFICATION PENDING`

Frozen source and real training: NO.
Last full focused CPU suite: PASS.
Next: `D4-D1_VGRA_INDEPENDENT_STATIC_UNIT_VERIFICATION`.
B-TEST: SEALED.

## D4-D1R guarded remediation — 2026-10-08

Two confirmed blockers remediated and CPU-tested, but D4-D1 independent exit not yet granted.
Signed-beta and all-zero-input risks remain OPEN. GPU/B-TEST forbidden.

## D4-D1 formal independent acceptance — 2026-10-08

D4-D1: `COMPLETE / INDEPENDENT SYNTHETIC CPU VERIFIED`.
Best-checkpoint pair-aware final evaluation and frozen unreadable VAL member
exclusion independently verified and captured in repository evidence.
75 CPU regressions passed.

D4-D2 transfer verification: NEXT, not yet complete.
Signed-beta and all-zero-input numerical gates: OPEN.
D4-E source freeze, GPU training, and B-TEST: NOT AUTHORIZED.
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

## D4-E2 R3 state and R3-04 scientific HOLD — 2026-10-09

VGRA branch: `research/vgra-impl-01`; accepted pinned HEAD `48cc059cfa133ae5fbfcec607f15daeda843d3ba` (not yet updated by the R3 local candidate).

D4-E2 R2 five-file source candidate remains **local and uncommitted**. R3-01 static source preflight PASS; R3-02 original 13/13 synthetic units PASS; R3-03 initial gradient-accumulation counterexample reproduced; R3-03B post-accumulation hooks 7/7 diagnostic PASS; R3-03C two-file correction applied locally, corrected suite 17/17 PASS. R3-04 13/13 tiny synthetic CPU tests passed, but **preexisting EMA corruption survived pre-update and optimizer.step happened before post-update EMA rejection** in the tested synthetic EMA stub.

**Current scientific gate:** `D4_E2_R3_04_SCIENTIFIC_HOLD_PREUPDATE_EMA`.

Next permitted scope: review the pre-update EMA guard gap and propose/source-review a narrow correction; no automatic patch application, full native AMP training, real dataset access, Git commit/push, or R4 freeze. D4-F remains locked; B-TEST sealed.

Evidence: see external/local `DETECTION_PROJECT_D4_E2_R3_04_DURABLE_RECORD_PACKET_2026-10-09.zip`. The captured Windows log vault's exact ZIP path, byte size and SHA256 are recorded in the next section and in `research/01_provenance/ARTIFACTS.csv`. This evidence remains local, not independently downloaded from the user's workstation.

## D4-E2 R3-04 evidence vault capture verified — 2026-10-09

External/local ZIP: `E:\PhD\Admitted\Research\Project 1\Detection_Evidence_Vault\D4_E2_R3_04_EVIDENCE_20261009_012330.zip`; bytes: `77695`; SHA256: `f621cd797aea1684887e8667eac1fa3420ee45e596ea5e97f11c892611a15ee0`. Captured 25 evidence files with no missing optional logs. The R1 collector incorrectly searched stderr for the R3-04 summary; R2 corrected the routing to stdout and successfully captured the original evidence without rerunning tests. The post-accumulation correction passed 17/17 synthetic unit tests, while R3-04 EMA pre-update timing remains scientific **HOLD**. The original five-file candidate remains uncommitted in the pinned scientific worktree; this documentation-only worktree is not the source freeze.

## D4-E2 R3-04B sealed local synthetic correction — 2026-10-09

**Local scientific candidate only:** one-file pre-update EMA finiteness guard applied to `ultralytics/models/yolo/detect/vgra_fail_closed.py`, source LF-normalized SHA256 `e1947b645202f3d73f6db035ffa673350837bd162ddd2822cc5ee6cd5bde082f`. The scientific checkout remains `research/vgra-impl-01 @ 48cc059cfa133ae5fbfcec607f15daeda843d3ba` with five candidate files uncommitted; historical `ultralytics/settings.json` remains untracked.

**Two Windows CPU synthetic suites PASS:** 17/17 targeted EMA regressions (real SGD, stub EMA/scaler), and 17/17 pre-existing guarded-unit tests, each exit 0. Pre-existing nonfinite EMA is now rejected before the optimizer step **in this synthetic path**. Original R3-04 negative control remains preserved, not rewritten. No native ModelEMA/AMP/trainer acceptance is claimed.

**Evidence sealed locally:** `D4_E2_R3_04B_EVIDENCE_20261009_120417.zip`; 56195 bytes; SHA256 `0cdb58ffec2f55bbe58064b5fe1d6739bdc748f23eaea670ab1469a141da629a`; 24 ZIP members; archive path `E:\PhD\Admitted\Research\Project 1\Detection_Evidence_Vault\D4_E2_R3_04B_EVIDENCE_20261009_120417.zip`. R1 sealer stopped on transcript capture semantics; R2 verified four of six native markers plus complete six-step host chronology, separately hashed raw test logs/source and backup. The original failed R1 sealer and recovery record are archived.

**Scientific gate:** `HOLD_NATIVE_MODEL_EMA_AMP_TRAINER_PENDING`. Next allowed stage: `D4_E2_R3_05_NATIVE_INTEGRATION_READONLY_AUDIT` without training or B-TEST. D4-E source freeze and D4-F remain locked.

## D4-E2 R3-05A static source audit and local vault seal — 2026-10-09

Windows exact-source read-only audit: 10/10 static checks PASS; `research/vgra-impl-01 @ 48cc059cfa133ae5fbfcec607f15daeda843d3ba`, five normalized LF candidate hashes exact, pinned three native Git blobs exact, no staging and expected six-path worktree status. R1 evidence sealer failed on UTF-16-BOM audit log decoding; corrected R2 reported exit 0, preserved both R1 logs, and sealed `D4_E2_R3_05A_EVIDENCE_20261009T091906Z.zip` (34,000 bytes; 16 members; SHA256 `da8fc99357c0aaa3317ec641d51f545d288d73b019026894f837ae3b6e0ef4e1`) outside Git. ZIP identity is Windows-reported; member bytes have not been independently retrieved here.

Source audit verified ModelEMA and optimizer/EMA control ordering only; native ModelEMA/GradScaler and actual production trainer calls have not run. `SCIENTIFIC_GATE=HOLD_NATIVE_MODEL_EMA_AMP_TRAINER_PENDING`. Next R3-05B requires separate scoped authorization. No source freeze, scientific commit, training, or B-TEST.
