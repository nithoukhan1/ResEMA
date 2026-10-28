# Step 2F5 — Kaggle T4×2 IMPORT-ONLY operator protocol

**Engineering date:** 2026-10-28. **Status:** READY FOR OPERATOR
EXECUTION. Neither GitHub CI nor ChatGPT has executed a Kaggle GPU
session, so **LIVE GPU ACCEPTANCE IS NOT YET CLAIMED**.

This protocol executes an authenticated *Python class import* on both
Tesla T4 ranks using a CPU/Gloo control group. It performs **zero**
training updates, checkpoint deserializations, model constructions,
validator passes or B-TEST reads.

## A. Pinned inputs (do not invent or silently replace)

Use the externally reviewed original R0 input-config JSON that maps
all five top-level categories: `assets`, `overlays`, `r21_code`,
`native_original`, and `runtime_yaml`. The file paths MUST match
the real mounted files on Kaggle and every SHA256 MUST match the
frozen in-code trusted pins.

The runtime development YAML is pinned at
`f2cc6e31f5f1be152f4dd7603a397e142963923f1d35de613eb7c61b168bf0c0`.
It may specify TRAIN and VAL only, NEVER TEST. The Early selected
pretrained checkpoint is pinned at
`620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1`.
Frozen assignment CSV SHA256 is
`87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4`.

No model weights are read as Python objects. Only the verified
pretrained files' bytes are hashed during this test.

## B. Prepare the exact Git checkout

1. Select **GPU T4 ×2** in Kaggle and verify both cards are visible.
2. Place an exact `nithoukhan1/ResEMA` Git checkout at a new
   `/kaggle/working/ResEMA_DDP_RELEASE` path. A clone is optional if
   the exact Git checkout is already provided. Network must only be
   used for retrieving the public source, never dataset or checkpoint
   substitutions.
3. **Pin the exact release commit supplied with this guide**, not
   the potentially moving branch name. Verify with:
   `git -C /kaggle/working/ResEMA_DDP_RELEASE rev-parse HEAD`
   and `git -C /kaggle/working/ResEMA_DDP_RELEASE status --short`.
   The HEAD must equal `expected_git_head` in the plan and the
   tracked worktree must be clean.
4. Confirm your approved original R0 JSON is mounted as a regular
   file inside `/kaggle/input/`. Do **not** rebuild its paths or
   approve new file hashes if the assets are missing.

## C. Exact plan schema

Create a NEW JSON file, e.g.
`/kaggle/working/ddp_b16_mv00_import_only_plan.json`, containing
the following JSON object. Replace each angle-bracket placeholder
with the **actual** approved path/SHA; the placeholders are not valid
runtime inputs.

```json
{
  "schema": "VGRA_DDP_B16_WORKER_PLAN_REVIEW_V1",
  "arm": "MV00",
  "run_id": "MV00_T4X2_IMPORT_PROBE_01",
  "mode": "FRESH_FROM_EARLY",
  "checkout_root": "/kaggle/working/ResEMA_DDP_RELEASE",
  "expected_git_head": "<PINNED_FULL_40_HEX_COMMIT>",
  "r0_input_config": "/kaggle/input/<YOUR_APPROVED_R0_DATASET>/<R0_CONFIG>.json",
  "stage_root": "/kaggle/working/MV00_T4X2_IMPORT_STAGE_01",
  "b_test_access": "NONE"
}
```

To check MV01 instead, set `arm=MV01`, use a distinct run ID,
a distinct new plan and stage path, and run the **same** program
with distinct new output paths.

## D. Execute the import-only probe (not training)

Run in a clean notebook kernel or Python child. Set GPU visibility
**before** launching the two-rank subprocess:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

From the exact pinned source checkout, run:

```bash
cd /kaggle/working/ResEMA_DDP_RELEASE

python -m research.r9_training.ddp_b16.ddp_kaggle_import_only \
  --plan /kaggle/working/ddp_b16_mv00_import_only_plan.json \
  --project /kaggle/working/MV00_T4X2_IMPORT_PROJECT_01 \
  --evidence-root /kaggle/working/MV00_T4X2_IMPORT_EVIDENCE_01 \
  --run-import-probe
```

No `--execute` flag exists for this launcher. The underlying
`ddp_worker.py --execute` and `ddp_child_review.py --execute`
remain blocked before training.

To prepare and audit stage bytes **without launching torchrun or
accessing GPUs**, replace `--run-import-probe` with
`--prepare-only`, using fresh stage and evidence directories.
Do not run the full command twice against the same stage path:
repeated runs must use NEW paths rather than overwrite evidence.

## E. Evidence and pass/fail requirements

The launcher writes exclusively under the NEW private evidence root:

- `IMPORT_ONLY_EVIDENCE.json`: pinned Git commit, source manifest,
  binding SHA256, GPU names, PyTorch/CUDA versions, state flags.
- `TWO_RANK_IMPORT_ONLY.log`: preserved raw two-rank stdout/stderr.
  Its SHA256 is included in the evidence receipt.

A valid import-only acceptance **must** satisfy all conditions:
both child processes return success; one exact rank-0 report is
present; rank identity and full binding digest match; exactly two
devices are observed and both names contain T4; backend is Gloo;
native trainer class corresponds exactly to MV00 or MV01; import
source was rehashed from frozen C1 just-in-time; and all four
training/export/TEST prohibition flags are false/none.

The only permissible pass status is
`LIVE_T4X2_IMPORT_ONLY_PASS_NOT_TRAINING`. Do not describe this
as NCCL numerical acceptance, model training, a GPU G4 recovery pass,
or official experiment authority.

If it fails: keep both evidence files, report the error, and STOP.
Do not change scientific hyperparameters, use B-TEST, bypass source
hashing, or force a PASS. A later attempt requires a new private
stage, project and evidence root.

## F. What to return for independent review

Return the raw `IMPORT_ONLY_EVIDENCE.json` and
`TWO_RANK_IMPORT_ONLY.log` from **each arm** along with the
actual pinned Git HEAD. Their hashes must agree with the log and
source checks. Paths in the log may contain sensitive mount names;
review/redact only nonessential personal mount names if necessary,
never hash values or error messages.

## Remaining gated work after import-only acceptance

An actual Kaggle T4×2 class import is the next step, not
native trainer construction, forward/backward AMP/NCCL numerical
acceptance, independent-session rank-specific G4 recovery,
selected-best B-VAL export, or official 100-epoch training.
All remain separately blocked. B-TEST remains sealed.

## Bytecode-write race prevention

A two-rank import-only probe shares one content-addressed staged source tree.
Python must NOT create pycache files inside that tree while the other rank
is auditing the 343-file inventory. The parent launcher now sets
`PYTHONDONTWRITEBYTECODE=1` for both children. The native import loader
also forces `sys.dont_write_bytecode=True` immediately before the second
source audit/import boundary. CI includes an explicit ordering regression.
A previously imported stage with bytecode artifacts is not reusable; prepare
a new private stage. This only prevents interpreter-created bytecode files;
it does not provide an OS-level read-only mount or protect against other
untrusted filesystem writes.
