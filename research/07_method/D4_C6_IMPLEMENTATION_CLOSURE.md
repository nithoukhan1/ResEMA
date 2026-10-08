# D4-C6 — VGRA V1 Implementation Closure

## Authority and scope
Branch: `research/vgra-impl-01`
Parent: `fc991546d6d360905b9b1af33e736612d0379e31`
Design: `a5d260ca83236c774e1920ce729e08eaacfcf5d4`

**VGRA_IMPLEMENTED=TRUE** (engineering source complete; C0–C5C CPU-tested).
**VGRA_SOURCE_FROZEN=FALSE**.
**VGRA_GPU_TRAINING_AUTHORIZED=FALSE**.
**B_TEST_ACCESS=NONE**.

This marks engineering implementation closure, NOT verified physical-data
functionality, source freeze, final architecture, or experimental success.

## Complete implementation chain

- C0 governance and dedicated worktree.
- C1 mathematical VGRA core.
- C2 frozen TRAIN/VAL pairing and target schema.
- C3 VGRADetect class-head integration and parameter cap.
- C4 pair-aware dataset/sampler/collation; full metadata batching audit.
- C5A visibility targets and TRAIN-only state-weight utilities.
- C5B whole-batch mixed-view runtime and four-component loss.
- C5C pair-aware trainer and validator; synthetic forward/optimizer/metric tests.

## Source inventory

Reproducible paths and SHA256 hashes are stored in:

`research/07_method/D4_C6_IMPLEMENTATION_INVENTORY.json`

Inventory file entries: `44`.

Protected stock files unchanged since design authority:
`[]`.

Candidate trainable parameters: 9,672,660.
Frozen upper budget: 9,870,093.
TRAIN exact pairs: 6,496.
VAL usable pairs: 1,401.

## Permanent failure history

- C1 R1 failed after 17 passing CPU tests: tracker string mismatch, no commit, rollback clean.
- C4 R1 passed 38 CPU tests but audit import path failed, no commit, rollback clean.
- C5C R1 passed 66/68 tests: wrong eval tuple handling and all-zero input
  nonfinite-gradient incident; no commit, rollback clean.
- C5C R2 subsequently passed 68/68 on seeded nonconstant synthetic inputs.
  That does not dismiss the all-zero gradient incident.

## Outstanding gates, NOT complete

The source inventory retains an explicit unresolved-gate matrix. In
particular:
1. Signed `beta = 2*tanh(rho)` polarity and interpretation.
2. All-zero-input nonfinite gradient cause and stability.
3. Exact Early checkpoint transfer, optimizer/EMA/save-load.
4. TRAIN-only nine-class visibility state counts and frequency weights
   computed from real labels under a separate governed contract.
5. Full pair-aware training/validation epoch behavior.
6. Explicit standalone pair-aware evaluation interface for the final
   separately authorized sealed test, not allowed now.

## Testing performed in C6

All C1–C5C focused CPU tests and historical TPSC regression tests rerun.
C6 source inventory cross-checks frozen gates, source hash records, protected
stock-code diff, dataset manifests, failure evidence, and forbidden accesses.

No new model training was performed and no patient image, YOLO label file
or B-TEST material was opened.

## Next authoritative action

`D4-D1_VGRA_INDEPENDENT_STATIC_UNIT_VERIFICATION`

D4-D2/D3, D4-E source freeze and GPU training remain locked.
