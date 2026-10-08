# D4-D2 — Formal pretrained Early transfer and synthetic CPU acceptance

## Authority
- Branch: `research/vgra-impl-01`; verified parent: `f1392c74b27ff8c3cccec7f61153cdc45bb4a735`.
- Canonical SCConv-Early `best.pt` SHA256: `620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1`.
- Containing run archive SHA256: `4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396`.
- Independent execution evidence: `research/07_method/D4_D2_INDEPENDENT_REPLAY_EVIDENCE.txt`.
- Captured stdout/stderr evidence SHA256: `ee23f2f7ae0bcb9775bf9b7e36aef8b77b911a926ea324233cca7c8988b2ee83`.
- Original D2A/D2B/D2C scripts pinned by exact SHA256 in `research/07_method/D4_D2_GATE_RECORD.json`.

## Acceptance scope: D4-D2=COMPLETE_CPU_PRETRAINED_TRANSFER_VERIFIED
All three pinned scripts reran successfully against the exact frozen D1 parent:

1. D2A verified selected Early checkpoint registry, ZIP and internal `best.pt` member hashes.
2. D2B verified 537/537 Early state tensors match exactly after actual model `.load`;
   zero absent keys and zero incompatible shapes/dtypes; exactly 35 additive VGRA state tensors.
3. D2B compared raw class logits, box/DFL and decoded outputs with Early at `rho=0`
   in synthetic TRAIN/EVAL, valid paired, mixed paired/single and all-single paths.
4. 9,570,093 pretrained Early parameters vs 9,672,660 VGRA parameters;
   102,567 additional trainable parameters, below the frozen cap.
5. D2C verified synthetic TRAIN-state binding before EMA, float and bool buffer
   preservation, finite nonzero pretrained gradients in native box, native class,
   visibility predictor and residual `rho`.
6. D2C verified a synthetic EMA update, real checkpoint loader restoration,
   and pair-aware forward from the restored synthetic best checkpoint.
7. One early D2C R1 harness assertion required bitwise equality across an FP32
   EMA update, failing on rounding `1.1920928955078125e-07`. R2 allows
   at most `1e-6` absolute difference for that specific updated float buffer,
   while preserving exact checks before EMA and verifying the changed EMA rho.

This is a **CPU/synthetic/structural checkpoint-transfer verification**, not a
real-data accuracy result and not a complete synthetic-epoch or end-to-end
training certification.

## Open gates carried forward without waiver
- All-zero-image nonfinite backward gradients: OPEN, specifically D4-D3.
- Signed `beta=2*tanh(rho)` assistance/suppression polarity: OPEN before D4-E.
- Persistent InfiniteDataLoader sampler-epoch and actual one-epoch synthetic
  training/EMA/validation integration: OPEN D4-D3.
- Zero-annotation batches, cross-image leakage and native decoding/NMS
  comparison under integrated D4-D3: OPEN.
- Actual B-TRAIN-derived visibility frequency counts/weights: NOT computed.
- MV-00/MV-01 identical pair-safe dataset/training recipes: NOT established.
- VGRA source freeze: FALSE; final architecture freeze: FALSE.
- GPU training and attention: NOT AUTHORIZED.
- Sealed B-TEST: NO ACCESS.

## Next governed verification
`D4-D3_VGRA_INTEGRATED_SYNTHETIC_NUMERICAL_VERIFICATION`
