# Chat 28 Step 2E — Native DDP-B16 FRESH/G4 resume integration

**CPU/Gloo candidate only. Not training authority.** The native private
adapter now invokes `ddp_b16_native_state_start()` after
`_setup_train()`, before callbacks/scheduler, and gates the existing
`require_ddp_b16_native_preloop()`. It uses the restored
`last_opt_step` cursor instead of forcing -1; restored parameter gradients
are not erased by the initial native zero_grad.

A resume request requires both rank capsules under the verified
independently pinned G4 manifest SHA256. The code checks selected-best
checkpoint SHA on rank 0, restores the rank-local model/BN/EMA, optimizer,
scheduler, AMP scaler, gradients, loader generator, sampler, augmentation
and process RNG state, with two-rank agreement before proceeding.
It skips a redundant first post-restore iterator reset.

The new CPU/Gloo tests use an explicit synthetic pin and a fake augmentation
restorer because the standalone CI does not install full Ultralytics. These
are fixtures only, NOT independent-session authenticity evidence. The
original frozen source files are untouched; only the exact-C1 private
source patcher changes.

**Execution remains locked:** neither `ddp_worker.py --execute` nor
`ddp_child_review.py --execute` can train. Further validation is required
for native CUDA behavior, loss/AMP, selected-best, exact replay, and durable
off-session checkpoint transport before nonofficial GPU acceptance.
