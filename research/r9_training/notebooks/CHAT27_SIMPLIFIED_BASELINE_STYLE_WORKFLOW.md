# Chat27 — Simplified Kaggle Training Workflow (MV00/MV01, 100 epochs)
Date: 2026-10-22 | STATUS: PRE-AUTHORIZATION / NO OFFICIAL EPOCHS

## Why this format
The project's canonical BASELINE-FREEZE-01 completed six 100-epoch YOLO11s training experiments across eleven Kaggle session ZIPs, with fresh `research/runtime/baseline_runner.py` and `research/runtime/resume_runner.py`. See `research/tools/KAGGLE_RUN_SOP.md` and `research/00_governance/LOCAL_GITHUB_KAGGLE_WORKFLOW.md`.

Those earlier runners cannot be substituted for the current MV00/MV01 training: C1 uses isolated per-GPU trainers, visibility-conditioned evidence and full-state G4 checkpoints. **The user interface, however, should be equally simple.**

## User workflow
### First version
1. Open `CHAT27_SIMPLE_SAVE_VERSION_AUTO_RESUME_PREAUTH.ipynb` in a new Kaggle notebook; enable two physical Tesla T4 GPUs and Internet.
2. Attach the **unchanged frozen B-TRAIN/B-VAL Split-B image dataset** plus the **original pinned supporting input datasets used in the successful Chat26 native acceptance**. In the longer term these may be combined into one verified private support bundle, but the current original R0 restager and 20 pinned input paths must not be silently rewritten. GitHub clone alone does not contain pretrained weights or historical recovery artifacts.
3. Leave `MODE="FRESH"`; both official run IDs, exact release SHA, first-session target up to 12 epochs, optimizer and model recipe are fixed automatically.
4. Independently review the notebook and exact input mount identities. **User's explicit final authorization**, private matching JSON and switch to `EXECUTE_APPROVED_OFFICIAL=True` are required before official training.
5. Once authorized, use **Save Version → Save & Run All once**. No separate manual command cells are necessary. The notebook clones fixed C1 source, restores R0 exact pinned bytes, generates input map/plans, verifies dataset and checkpoints, trains two isolated workers, and exports original G4 archives.

### When the Kaggle session ends
6. The C1 post-G4 boundary handler stops **earlier than the runtime cutoff**, after finishing and verifying an epoch and leaving reserve for export. Do not wait for Kaggle to kill a mid-epoch job at 12 hours; 12 hours is an upper quota, **not** a recoverable checkpoint.
7. Confirm the saved version contains `C1_OUTPUT_INDEX.json`, `.zipblob` original archives for both arms, hash sidecars, and logs. Save the version so the outputs remain accessible.
8. Independently check the original persisted archives via pinned `research/r9_training/code/R9_OFFSESSION_TRANSPORT_R1.py verify`, produce the required **separate external transport attestation** for each arm, and retain original archives. A saved-version label alone does not establish byte integrity.
9. For a new Kaggle session, attach the **latest saved official notebook version output** plus its separate verified transport attestation files as private Kaggle inputs; preserve the same pinned source and original image/support input mounts.
10. Change only `MODE="RESUME"` in the same simple notebook. It reads `C1_OUTPUT_INDEX.json`, obtains **each arm's independent epoch, SHA and next cursor**, validates the original G4 archive and external receipt, and resumes. Use Save Version → Save & Run All once.
11. Repeat until each arm reaches **exactly native epoch100**, independently proves final B-VAL evaluation and terminal G4 authenticity, and archives external output. B-TEST stays SEALED and is **never** used for tuning.

## Why not just attach images?
- **Necessary science:** frozen Split-B images/labels, original SCConv-Early pretrained best checkpoint, per-image assignment and visibility evidence for VGRA, immutable runtime train/val YAML.
- **Existing implementation identity:** corrected R0 restager, originally approved 20 input byte identities, model/overlay evidence and historical checkpoint provenance. Most of these are engineering/reproducibility requirements, not additional model data.
- They can be **automatically checked** instead of configured manually; files must still exist physically. A single private **support dataset** is the target after verifying its internal paths and exact bytes against the original restager. No unverified file re-packaging is approved here.

## Differences from historical baseline
| Topic | BASELINE-FREEZE-01 | Current C1 MV00/MV01 |
|---|---|---|
| Fresh runner | `baseline_runner.py` | `R9_PRODUCTION_DUAL_SESSION_C1.py` and native C1 worker |
| Resume runner | `resume_runner.py` from full run / `last.pt` | Original per-arm full-state authenticated G4 |
| GPUs | Registered original baseline device pair | Two isolated independent T4 workers; not DDP |
| Model difference | YOLO11s baseline configurations | Paired SCConv-Early control vs VGRA-V1 |
| Preservation | Complete run folder and last.pt | G4 original archive, original SHA, independent off-session attestation |
| Kaggle user steps | Attach, clone, Save Version, resume | Same simplified interface; different internal verification |

## Source and authority
- Frozen science `b783eb3a470ab010785a8a24a90a666e9a3f2ed6`.
- Frozen C1 release `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
- Prototype notebook lives **only** on `research/chat27-official100-preauthorization` and is not GPU-validated.
- Reviewed Windows preflight 336+7 original source bytes PASS, 27 C1 CPU tests PASS; nonofficial native Chat26 epoch1+independent epoch2 acceptance PASS. These do not authorize an official 100-epoch run.
- The simplified notebook is additive orchestration only; it has **not** yet been compiled, run on Kaggle, checked against fresh mount layout, or authorized. No training has started.

## Next gate — do not skip
Static source/notebook review, confirmation of the actual Kaggle mounts and exact asset pins, user-approved official JSON, then the first governed Save & Run All on a **fresh Kaggle session**. Repeat nonofficial two-epoch GPU testing only if a specific previously unresolved defect requires it.
