# Detection Project — Cumulative Master Handoff (Chats 1–27)
**Cutoff: 26 October 2026; successor chat: 28.**

## A. Research purpose and chronological lineage

The Detection Project studies nine-class pediatric wrist-radiograph abnormality detection with GRAZPEDWRI-DX and YOLO11s. Preserve distinct result families and stages:

1. **C3k2 + DySample + ResEMA V1/V2/V3:** earlier architectural research, phased local Kaggle training, corrected implementation/provenance, Split-A and Split-B runs and ablations. Specific historic early model observation for full e50 + Res-V2: 11.628M parameters, precision .6148, recall .3995, mAP50 .6400, mAP50-95 .4230. This historic result is **not a current frozen-Split-B comparison**. See nested original historical migration and repository research records for experimental-by-experimental tables and true provenance.
2. **C3k2/ResEMA forensic corrections** and **SCConv-Early vs SCConv-4stage, DySample and EMA screens**: keep separate (a) original training-time selected metric family vs (b) subsequent A12-D2 revalidation. Training-time selected mAP50-95: YOLO11s .41704; SCConv-Early .43130; SCConv-4stage .42587; DySample .41525; EMA .42796; SCConv-Early+EMA .41839. Standardized A12-D2 diagnostic revalidation mAP50-95: .41765881, .42588415, .42452888, .41427802, .42663323, .41308765 respectively. These two evaluation families must **not** be combined into one ranking or promoted as official VGRA data.
3. **A12-D3 error analysis:** at conf .25 SCConv-Early relative to YOLO11s changed TP -1, FP -44, FN +1; precision +.0055, recall -.0001, F1 +.0027; 100 FN-to-TP and 101 TP-to-FN object identity shifts. Confidence .50 and duplicate-box sensitivity caution against assuming attention or upsampling automatically improves scientific outcomes.
4. **Initial LPQ quality/gap proposals:** DAR/GDS novelty rejected after GFLV2, VarifocalNet, TOOD, YOLOv10/DHLA collision review. Research pivoted to actual AP/LAT paired-view data.
5. **VGRA candidate:** patient group pairing, supervised per-class four-state (both absent/AP-only/LAT-only/both) targets and target-specific signed companion residual to **classification logits**, not DFL/box coordinates. Novelty classification remains PLAUSIBLE CANDIDATE / NOT YET EMPIRICALLY VALIDATED.
6. **R9/C1 frozen science, nonofficial acceptance:** authenticated original SCConv-Early transfer + two Kaggle T4 GPUs, one single-GPU process per arm, global **batch 8 per model**, workers0, epoch1 and genuine different-session epoch2 recovery, Windows byte audits all preserved. **Not 100-epoch scientific results**.
7. **Chat27 alternate execution condition — DDP-B16:** user requested separate Kaggle notebooks, each using both T4 GPUs with global batch **16 = 8 images per rank** for one arm. Proposed for speed and paired-image integrity, not as a VGRA scientific architecture change. Implemented on a single additive branch with extensive CPU/Gloo verification; **not tested on real CUDA/model yet**.

The anatomy-conditioned transfer-preserving idea was previously discussed but remains paused. Do NOT silently change the current scientific direction to that idea, generic attention, or another module.

## B. Dataset freeze and pairing
- GRAZPEDWRI-DX dataset overall: **20,327 images; 6,091 patients; nine YOLO classes** (historic operative label total **47,443 boxes** per previous source handoff).
- Class ordering locked: 0 boneanomaly; 1 bonelesion; 2 foreignbody; 3 fracture; 4 metal; 5 periostealreaction; 6 pronatorsign; 7 softtissue; 8 text.
- Frozen patient-disjoint **Split-B**: B-TRAIN **14,227 images, 4,264 patients**; B-VAL **3,050 listed, 914 patients**, **3,049 operational readable** due to one historical corrupted/unreadable PNG; B-TEST **3,050 images, 913 patients SEALED**.
- Pair key patient_id + study_number + laterality; AP/PA projection with LAT companion; never synthesize partners.
- B-TRAIN **6,496 exact valid pairs plus 1,235 singles = 14,227 images**. Paired train group feasibility 6,502 AP/LAT-capable groups among 7,590 patient/study groups; 6,496 clean exact pairs. B-VAL **1,401 operational exact pairs plus 247 singles = 3,049 usable**.
- Complementarity study D4-B2B: 317 view-exclusive among 2,761 annotated pair/class units (**11.481347%**), AP/LAT Jaccard approx **.914408**. View-only annotation does NOT automatically prove one view cannot anatomically show pathology.
- B-VAL foreignbody class has no GT positive support: AP N/A not numerical zero; keep rare-class caveats.
- Never repartition, quietly repair the corrupted PNG, or access B-TEST as a stand-in for validation. Existing patient-level boundaries must remain checked.

## C. VGRA scientific design and models
- **MV00**: YOLO11s with SCConv-Early baseline, paired training/validation pipeline, **9,570,093 trainable parameters**.
- **MV01**: same backbone and pretrain plus VGRA V1, **9,672,660 parameters (+102,567)**.
- Shared native backbone/neck single pass per mini-batch. P3/P4/P5 pooled features each project 32D; per-view descriptor 96D; joined pair representation 384D; hidden layer 128 predicts 9 x 4 visibility-state logits. Training visibility labels y=a+2*l for per-class annotation presence AP/LAT.
- Four-state state weights derived from B-TRAIN-only inverse sqrt(count+1) and normalized; auxiliary visibility weighted cross entropy coefficient **0.25**.
- Directional detached gates k_AP=q11-q01 and k_LAT=q11-q10; class-conditioned rank16 spatial compatibility in [0,1]; final classification-logit additive residual beta_l*k*m, beta_l=2 tanh(rho_l), rho initialized 0. The sign of beta_l is learnable—do NOT claim it enforces a fixed positive assist or negative suppression. Native box/DFL regressions unchanged; singles have a native residual-free pathway (shared MV01 training can still modify weights).
- Authentic SCConv-Early pretrained best SHA256 **620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1**, pretrained archive SHA256 **4e3a811be5b8126e788ca4b7854e2df2b070a08e2abef396**; exact 537 Early tensors transferred. Never start MV01 randomly for main comparison.
- Paired inference vs native single-view fallback must be separately reported. Do not invent cross-view geometric correspondence or claim clinical utility without validation.

## D. Primary experiment protocol
- All DDP training must use matched, separately initiated MV00 and MV01 on frozen Split-B, seed42, imgsz1024, SGD lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005, cosine schedule, nominal nbs64, AMP true, workers0, disabled mosaic/mixup/cutmix/copy-paste, 100 epochs.
- Historical C1 uses one T4 / model and batch8. New DDP condition is **two T4s / one model, global batch16**, ideally one notebook per arm, subject to Kaggle resource/ToS availability. **Batch-16 DDP is a NEW training condition**, never a resume from batch8 C1 nonofficial states. Two simultaneous notebooks would use up to four physical T4s—not guaranteed or automatically faster overall; measure actual runtime.
- New DDP sampler yields **890 synchronized steps per epoch**: 812 pair steps, 78 single-image steps. All 14,227 images once each; 6,496 complete pairs never split across ranks; local 7–8 images/step. No fabricated padding.
- Main endpoints: validation mAP50–95 primary; mAP50, precision, recall, F1@conf.25, fracture AP, TP/FP/FN, duplicate/high confidence behavior, per-view and paired/single subgroup, visibility calibration, compute/parameters, patient-level uncertainty.
- Original promotion gate: promote on abs mAP50–95 >=+.0050 with non-regressed F1@.25, recall decline <= .0020, no material fracture regression/high-conf TP collapse; hold +.0020 to below +.0050 or credible clinically meaningful diagnostic gains; reject adverse outcomes. These are prespecified historical candidate gates, not achieved VGRA performance.
- If main result qualifies, planned controlled ablations MV02 visibility-only, MV03 generic residual, MV05 positive assist-only, MV06 no-Early VGRA etc (see frozen contract) and optional attention only after clear empirical evidence. Do not launch ablations before main matched results. B-TEST once only after complete validation/model selection and explicit permission.
- Publishing intent Q2/Q3 journal manuscript if empirical results justify it, not guaranteed acceptance or novelty.

## E. Frozen C1 release & Chat26 evidence
- Repository: https://github.com/nithoukhan1/ResEMA .
- Frozen scientist-science ancestor `b783eb3a470ab010785a8a24a90a666e9a3f2ed6`; R9/C1 accepted baseline `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`, old branch `research/vgra-r9-training`.
- Nonofficial fresh Kaggle epoch1 (Oct21,2026) dual arms and separate session authentic epoch2 (Oct22,2026) completed. Original output two T4 GPUs concurrently running different models, each batch8; 2/100 printed but **not official epoch count**.
- Independent off-session Windows G4 archive checks: epoch2 MV00 archive SHA256 `9c436620f9b10af927b8f6a964ea4ce666ba81b30174afa59bd5cfd038aa2b22`; MV01 archive SHA256 `66bc10de6abb9728752550999c70491018c0070ee56b886c156657d839ccc55e`; outer epoch2 transfer ZIP SHA256 `59b41703df7a94768751a34647fe3d2133035ca34b9024aa7d8dd21630b0a38d`.
- Prior 2-epoch engineering B-VAL rehearsal metric (MV00 .33821, MV01 .39771 mAP50–95) NOT official. Do not treat as causal or as current DDP result.
- Historic C1 best checkpoint SHA may remain unchanged between epoch1 and epoch2; this is not proof last.pt/optimizer failed to advance, nor affirmative proof of uninterrupted equivalent training. Exact original logs and checkpoint registry provide context.
- Simplified one-click C1 notebook prepared Chat27 on branch `research/chat27-official100-preauthorization` commit `e03a294612034830916a2907def16c4ba418aa4a`, path `research/r9_training/notebooks/CHAT27_SIMPLE_SAVE_VERSION_AUTO_RESUME_PREAUTH.ipynb`. This is prepared/CPU-approved, NOT officially GPU deployed. C1 science unchanged.

## F. New Chat27 DDP implementation checkpoint
- Dev branch `research/vgra-ddp-b16`, created explicitly off immutable C1 `fa6a...`. Exact acceptance checkpoint before migration docs `1b1696d3d77cf327452bcd40cb236c05c91098a2`, **54 commits ahead, 0 behind**; comparison confirmed **NO MODIFICATIONS to `frozen_source/` or `official_candidate/`**.
- Latest passing GitHub CI on that code checkpoint:
  - Research Integrity run https://github.com/nithoukhan1/ResEMA/actions/runs/40287227288 — PASS.
  - PyTorch CPU Numerics run https://github.com/nithoukhan1/ResEMA/actions/runs/40287227412 — PASS.
- Development changes are additive `research/r9_training/ddp_b16/` and two GitHub workflow files, with exact source adaptations in memory/private stage (5 existing files adapted only in isolated generated copies).
- Verified at source and CPU/Gloo levels: distributed pair sampler, loader epoch reset, exact 336 scientific + 7 integration sources staging, loss math and synthetic real 2-rank gradient test, AMP static safeguard checks, fail-closed launcher guard, per-arm worker preflight (not executable), two-rank G4 epoch capsule and SHA receipts, rank0-only B-VAL broadcast cardinality/failure test, two-rank native G4 epoch-end hook including selected best and optimizer step agreement.
- **Not yet solved:** complete native torchrun child that binds all frozen scientific/manifest/pretrained inputs on both ranks, DDP-specific startup/resume guards (old G3B/G4 pinned to C1 batch8), real CUDA AMP, authentic YOLO/VGRA forward/backward, real 3049 B-VAL, full off-session G4 transport/cross-session replay, terminal epoch100 result and final best model export.
- No CUDA/GPU new DDP experiments run. No official results. The repository migrations folder adds only handoff/docs and does NOT mean a DDP release was frozen.

## G. Research integrity rules that carry forward
- Never merge result families without source/evaluator identity, never claim nonofficial engineering runs as official training.
- Preserve authentic AP/LAT pairs and scientific loss, do not shortcut via a native single-image YOLO forward for MV01.
- Always separate validated source/CPU behavior from actual CUDA execution, real-image metrics, authentic offsite resume.
- Avoid extra branches, unnecessary variants, needless rounds of file creation; work one governed phase at a time and keep progress tracker.
- DO NOT run or tune on B-TEST. DO NOT silently change approved C1 or frozen original dataset.
- Avoid Kaggle quota/ToS circumvention; treat two independent T4×2 notebook placements as conditional resource availability.
- Always tell user ahead of next material migration checkpoint.

## H. Next action for Chat28
Read `02_DDP_B16_IMPLEMENTATION_AND_EVIDENCE.md` and `03_NEXT_STEPS_AND_GATES.md`. Continue one branch, one scoped step: implement authenticated native DDP startup/resume and the native-G4-specific hook interface in the private generated trainer stage; test real model on two GPU ranks only after all gates. Do not schedule or claim background work. User must explicitly authorize any official 100-epoch runs.
