# Chat 28 Step 2F2 — Two-rank exact native-binding review

Both torchrun child ranks now independently run historical R0, full
private-stage source validation and exact five-input SHA256 checks. The
optional `--native-binding-review --project /kaggle/working/<fresh_root>`
then derives the same immutable constructor digest and compares it
collectively in CPU/Gloo before any Ultralytics model import or CUDA work.

`--review-only` remains backward-compatible. `--execute` continues to
reject before reading a plan or importing Torch. A project argument is
forbidden for the ordinary review path; required for the native-binding
review. Rank errors and mismatched constructor paths, model arm, digest or
authority flags fail closed on both ranks.

This is not full Step 2F execution: the native GPU constructor and actual
`trainer.train()` lifecycle have NOT been invoked. Two-process CI tests are
synthetic, not radiograph-data or off-session GPU acceptance. Official
MV00/MV01 remains 0/100; B-TEST sealed; C1 source immutable.
