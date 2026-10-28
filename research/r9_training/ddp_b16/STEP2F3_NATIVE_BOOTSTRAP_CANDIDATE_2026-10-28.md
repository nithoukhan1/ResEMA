# Chat28 Step2F3 — Native class bootstrap candidate

Implements candidate-only native construction components:

- fixed MV00/MV01 class identity and validated source-module path;
- source-root import isolation from any globally installed Ultralytics;
- exact `BeforeNativeAMP` mixin that sets the inherited R7C AMP gate
  before the original native trainer constructor;
- physical two-T4, two-rank torchrun topology verification;
- instance-local fresh DDP-B16 G4 identity, private checkpoint root and
  native-state request binding for the already CPU-tested Step2E hook.

The module deliberately has **no `trainer.train()` invocation** and does
not create an executable CLI option. `ddp_worker.py --execute` and
`ddp_child_review.py --execute` are still fail-closed. A future executable
worker must ensure that the runtime binding is derived from actual R0
and staged private-source on-disk verification plus two-rank agreement,
not from an arbitrary in-memory object.

This code is CPU-only at this stage, with fake trainer/topology unit tests.
We have NOT exercised native Ultralytics class import, real CUDA AMP,
B-VAL inference, cross-session G4 resume or terminal best model export.
All original C1 files, frozen Split-B and sealed TEST remain unchanged.
