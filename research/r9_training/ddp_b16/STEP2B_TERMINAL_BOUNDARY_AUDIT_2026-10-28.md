# Chat28 Step 2B — rank-0-only selected-BEST terminal evaluation

**Candidate, CPU/Gloo only — NO GPU or official training authority.**

The frozen engine invokes `self.final_eval()` on both DDP ranks.
The original arm-specific selected-BEST validators are pair-aware and reject
rank 1; both-rank final evaluation is incompatible with DDP-B16.

The private exact-blob source adapter now calls
`rank0_selected_best_terminal(self)` collectively at terminal epoch for
world_size=2. Rank 0 alone executes the original MV00/MV01 `final_eval()`
with a temporary governed validator marker, verifies 3,049 B-VAL images,
checks native best/last checkpoint bytes by SHA256, and broadcasts finite
metrics and checkpoint digests. Rank 1 does not evaluate; rank-0 failures
are transmitted to both ranks. TEST remains sealed. Export is not authorized.

The separate CPU/Gloo two-process acceptance uses fake validator and synthetic
checkpoint bytes. It is NOT evidence for actual selected-BEST radiograph
inference, distributed CUDA execution, checkpoint restore, or terminal export.

Release remains BLOCKED until a real authenticated torchrun worker and
off-session two-rank restart are independently verified and accepted.
