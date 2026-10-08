# D4-C6 — OPEN Gates and D4-D Verification Checklist

## D4-D1: independent static and semantic verification
- Re-run C1–C5C focused CPU tests independently.
- Inspect entire code path for hidden bypass of VGRA during validation.
- Verify masks/metadata invariants, pair shuffling and no synthetic companions.
- Reconcile the true trainable parameter count and protected native source.
- Re-review `beta=2*tanh(rho)` negative-sign interpretation; no silent change.
- Verify that MV-00 and MV-01 use identical pair-safe preprocessing.

## D4-D2: checkpoint transfer and zero-residual identity
- Obtain the authoritative pretrained Early checkpoint and SHA256.
- Verify which checkpoint tensors transfer exactly and which do not.
- Check native class logits, boxes/DFL and decoded outputs against Early
  at rho=0 in TRAIN and EVAL modes, including singles/unpaired fallback.
- Check checkpoint/EMA/save-load persistence of TRAIN-only visibility buffers.
- Test parameter cap and gradient reachability.
- Do not use B-TEST.

## D4-D3: integrated and numerical verification
- Reproduce C5C R1 all-zero-input gradient failure in isolated diagnostics.
  Identify exact parameter names, source operations and NaN/Inf counts.
  Do not close this gate merely because nonconstant inputs pass.
- Test nonconstant synthetic mixed pairs/singles and zero-annotation batches.
- Verify 1+ synthetic training epoch and EMA update/reload as permitted by the
  governed synthetic CPU test contract.
- Check pair sampler under persistent InfiniteDataLoader iteration and
  deterministic epoch changes.
- Verify pair-aware decoding/NMS/metric computation versus native control.
- Verify gradients finite and no unintended cross-image contamination.

## Beyond D4-D
- Source freeze D4-E is separate and requires successful independent gates.
- The sealed B-TEST evaluation endpoint needs a separate later authorization.
- Do not GPU-train, tune hyperparameters, add attention, or access B-TEST.
