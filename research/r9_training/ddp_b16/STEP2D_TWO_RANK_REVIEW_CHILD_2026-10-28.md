# Chat28 Step 2D: explicit two-rank torchrun review-only child

The candidate `ddp_child_review.py` accepts only `--review-only`
and requires a genuine `WORLD_SIZE=2`, rank and local-rank mapping from
torchrun. Each rank separately executes the existing input R0/336-file C1
review and the hardened Stage-2C private-source admission, then initializes
a **CPU/Gloo** process group and collectively compares identities.

Any rank-local verification error, disagreement in the private stage receipt,
R0 identity or candidate DDP release digest, wrong world/rank mapping, or
authority flag causes rejection. Two-rank Gloo tests are synthetic and
independently validate the matching/failure propagation behavior.

**The `--execute` path refuses before plan read and before torch import.**
There is still no native DDP training worker, restored rank state, GPU
acceptance, independent off-session transfer, or terminal export.

The current development work is nonofficial engineering only; frozen C1,
Split-B, and sealed B-TEST remain unchanged.
