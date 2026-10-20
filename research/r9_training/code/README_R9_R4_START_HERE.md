# Detection Project — R9 R4 Direct Training-Boundary Candidate (20 Oct 2026)

**Why this exists:** stop expanding CPU engineering gates. This adds actual `trainer.train()` to the verified Chat 23 R9B R3 class integration. It deliberately executes **one native nonofficial acceptance epoch per arm**, not 100 epochs. Real GPU operation is **UNTESTED**, and there is **no official-run authorization**. B-TEST remains sealed.

**Exact lineage:** the original R9B R3 archive is physically embedded unchanged; it holds R2 > R1 > R9A > R8 source ancestry. This package contains original R3/R2 Python modules unchanged (only copied) plus R4 files, without changing the 336 + 7 scientific source files. G4 original `official_training_authorized=false` remains unchanged.

## The minimum practical workflow

1. Add the R4 ZIP as a Kaggle input (and mount the previously accepted actual Split-B assets, pretrained model, recovery R0 overlays/native source, and YAML). Extract it to a new working directory. Do **not** install a different Ultralytics package over the staged source.
2. From inside the extracted R4 directory, stage authentic source once:
   `python R9_R4_STAGE_SOURCE.py --r3-archive DETECTION_PROJECT_CHAT23_R9B_NATIVE_WORKER_INTEGRATION_R3_CPU_CANDIDATE.zip --output /kaggle/working/r9r4_authentic_stage`
3. Edit `R9_R4_INPUT_MAP_TEMPLATE.json` into `/kaggle/working/r9r4_map.json`, replacing **all** placeholder paths with verified actual paths. The R0 `config_path` JSON is the accepted *historical* asset/overlay/native/code mapping; it is **not** the R9 staged-source manifest. It must remain pinned to its original R0 file-level hashes, even when R9 has an R8 source correction.
4. Create two plans, run full pin verification with **no GPU training**:
   `python R9_R4_MAKE_FRESH_PLANS.py --input-map /kaggle/working/r9r4_map.json --output-dir /kaggle/working/r9r4_plans --run-prefix R9R4_ACCEPT_20261020`
5. Launch both real native bounded GPU workers (2 T4, physically 0 for MV00, 1 for MV01, batch 8 each; **not DDP**):
   `python R9_R4_DUAL_LAUNCH.py --mv00 /kaggle/working/r9r4_plans/MV00.json --mv01 /kaggle/working/r9r4_plans/MV01.json --log-root /kaggle/working/r9r4_logs --accept-nonofficial-bounded-gpu`
6. On a complete PASS, both respective `/kaggle/working/<run_name>_R9R4/` directories should contain the nonterminal transport ZIP + `.sha256` and JSON report. **Download** these outside Kaggle, verify the entire ZIP and SHA using the unchanged `R9_OFFSESSION_TRANSPORT_R1.py` `verify` command on Windows, save the independent verification attestation, and mount for a new Kaggle session to test true resume with the one-arm R4 worker. This off-session stage is necessary to guarantee recovery from Kaggle session expiry; local ZIP creation does not prove recovery.

**STOP on any failure; inspect per-arm raw logs.** Do not relabel CPU pass as GPU acceptance; do not start official 100 epochs with this bounded-acceptance package. The native terminal epoch100 path and official authorization are separate.

## Already covered / not repeated

No R7C/R7D retraining, no R8 selected-best re-evaluation, no new architecture, no new split, no B-TEST access. The R3 CPU source manifest checks remain pinned; this package supplies the missing real GPU execution boundary only.
