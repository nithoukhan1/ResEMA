# Detection Project — Chat27 C1 Official MV00/MV01 100-Epoch Launch Checklist
Date: 2026-10-22. Status: PRE-AUTHORIZATION / NO TRAINING LAUNCHED.

## A. Frozen research identity
- Frozen science ancestor: `b783eb3a470ab010785a8a24a90a666e9a3f2ed6`.
- Frozen reviewed C1 production source release: `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`.
- All model/recipe/weights/dataset pins unchanged. Original source 336 and integration modules 7 verified on Windows, 27 C1 production CPU tests PASS in Chat27.
- Model arms: MV00=SCConv-Early YOLO11s control; MV01=SCConv-Early YOLO11s + VGRA-V1.
- Split B frozen patient-disjoint B-TRAIN/B-VAL only, B-TEST SEALED. The historical unreadable B-VAL PNG is unchanged.
- Chat26 nonofficial fresh epoch1 and separate-session resumed epoch2 passed independent Windows verification. **These are not official 100-epoch progress.** Both official starts must be FRESH, from the original approved pretrained model.

## B. Launch artifact and authority
- Notebook: `CHAT27_C1_OFFICIAL_FRESH_RESUME_100EPOCH_T4X2_PREAUTH.ipynb` (9 cells).
- Notebook preparation branch: `research/chat27-official100-preauthorization`. Do not confuse this notebook branch with frozen science release. The notebook itself clones/checks out the exact pinned C1 production SHA.
- Default `EXECUTE_APPROVED_OFFICIAL = False`. Running in this state should perform source/data/plan review only; **does not train**.
- An exact **separate user-approved** `CHAT27_C1_OFFICIAL_USER_AUTHORIZATION.json`, mounted privately on Kaggle, is mandatory to unlock a production run. It must match the original C1 approval schema and exactly the two official run IDs:
  - `C1OFFICIAL_MV00_20261022`
  - `C1OFFICIAL_MV01_20261022`
- Never commit a user's authorization record to GitHub. Never edit original frozen authority flags in G4.
- **User has not approved the final exact notebook/session configuration as of this file creation.** Do not set execution flag to true or manufacture authorization automatically.

## C. Before first official Save & Run All (NO GPU execution yet)
1. Download/view the exact 9-cell notebook from the preparation branch and inspect its parameters.
2. Use a NEW Kaggle notebook with accelerator T4×2, Internet enabled, and the exact original Chat26-successful six source inputs + raw frozen Split B images.
3. Provide corrected original `CHAT25_C1_R0_STALE_PATH_SAFE_RESTAGE_CORRECTED.py`, SHA `88cd4a88c373f451e54ba64736e961e735757d94b3779feea55a526868c45f4f`, by direct mount or matching Chat25 portable ZIP; do not substitute a rewritten restager.
4. Confirm `ARMS` has FRESH start 1 and target up to 12 for each model. A 12-epoch requested session is a cap, not a promise that 12 epochs will fit runtime.
5. Verify the exact GitHub checkout, 336+7 source byte identities, frozen recipe, runtime YAML, original pretrained best/early archive, assignment, visibility and R7D config; ensure the B-TEST path remains unavailable.
6. Review the final official approval record, original Kaggle input map and session plan. Obtain explicit final user authorization before toggling any GPU execute switch.
7. Only after all gates: mount the user-approved private authorization file, set `EXECUTE_APPROVED_OFFICIAL=True`, save an immutable Kaggle version with **Save & Run All** once. Do not run individual cells first; do not restart an already used version.
8. Monitor both workers' logs. The initial full B-TRAIN/B-VAL scan may be long; do not mistake it for a hang.

## D. Session budget and runs
- 100 epochs per model; 2 physically independent T4 workers, no DDP, batch8 each, 1024px, SGD, seed42, 100-epoch native total.
- Each C1 session plan is capped at 12 requested epochs per arm, worker deadline 9h, reserve 1h for export; native boundary may stop earlier. Consequently a theoretical minimum is 9 session versions per arm, and the actual number may be more.
- Independent per-arm progress is essential: MV00 and MV01 can finish at different epochs in one session. Never set both resume cursors using only one arm's result.

## E. Mandatory external preservation between sessions
1. Save Kaggle version output, download ORIGINAL `.zipblob`, fresh `.zipblob.sha256`, supervisor and per-arm logs and `C1_OUTPUT_INDEX.json`.
2. Verify all original hashes on Windows with the original R9 transport verifier. Independently attest EACH arm's exact original archive, terminal flag, completed zero-based epoch, unchanged official run ID and manifest. Do not mistake selected_best.pt hash for last.pt/optimizer progress.
3. Preserve originals and their attestation JSONs; upload the independently verified triplets to a PRIVATE Kaggle dataset for the next separate session.
4. For each arm individually: next `start = previous completed one-based epoch + 1`; next `target = min(start + 11, 100)`; set `mode=RESUME` and exact original `resume_sha256`.
5. Keep `RID`, original release SHA and science unchanged; increment `SESSION_LABEL`. A resume is valid only with the matching original official run's full-state G4, not Chat26 acceptance capsules.
6. Revalidate all frozen source/data pins in each new Kaggle session. Never clean up a checkpoint until external byte inspection and retention complete.

## F. Terminal epoch 100
- Terminal GPU execution was not proven by Chat26's two-epoch acceptance.
- Do not claim completion until **both** models separately verify zero-based epoch 99, genuine native final_eval on B-VAL, terminal G4 receipt, original ZIP transport, off-session preservation, and complete 100-epoch run history.
- Terminals must not be resumed. B-TEST is sealed and needs a distinct later authorization for any evaluation.
- Any missing/corrupted/orphaned checkpoint, mismatched cursor, failed arm, early terminal, or output integrity issue is a BLOCKER; preserve original failure evidence without declaring success.

## Chat27 progress tracker
| Milestone | State |
|---|---|
| Historical Chat1–26 handoff loaded | Partial ZIP internals not independently traversed in current environment |
| Chat26 native epoch1/epoch2 acceptance | COMPLETE nonofficial |
| Local Git release and 343-byte-source integrity | COMPLETE |
| C1 production CPU tests | COMPLETE: 27/27 |
| Reusable Kaggle official launcher, guard default | CREATED on separate preparation branch; runtime untested |
| Exact six Kaggle mount revalidation | PENDING |
| External final session-plan review | PENDING |
| Explicit final user authorization | PENDING |
| Official fresh 100-epoch run | NOT STARTED |
| B-TEST | SEALED |

## Evidence boundary
This checklist and companion notebook are preparation artifacts. The notebook has been JSON-structure/static-contract reviewed, **not** Python-compiled or Kaggle-run in Chat27. They do not certify performance, runtime success, full offsite persistence, or completed terminal epoch100.
