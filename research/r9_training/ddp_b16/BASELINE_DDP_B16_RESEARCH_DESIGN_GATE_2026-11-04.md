# Plain YOLO11s baseline — DDP-B16 research design gate (2026-11-04)

**Status:** DESIGN GATE ONLY. No new baseline trainer or Kaggle job created. No B-TEST access. Preserve existing MV00/MV01 science and checkpoint lineages unchanged.

## 1. Existing baselines are already complete

Repository `research/05_experiments/BASELINE_FREEZE_01_CLOSURE.md` records 6/6 canonical YOLO11s 100-epoch baselines and 11/11 archive sessions, with all six best checkpoints independently revalidated on B-VAL (no B-TEST). Canonical Split-B original pretrained YOLO11s `BASE-B-ORG-PT-S42` best-row B-VAL mAP50–95 is **0.41704** and standardized best-checkpoint B-VAL mAP50–95 is **0.4176588141073255**. Its `best.pt` SHA-256 is `65e3c59901e0429f70a7b368dc29ebb698c427a799cfcf66354a882ec48bc4c7`. Preserve these as historical benchmark evidence, not identical DDP-B16 controls. Split-B original scratch baseline `BASE-B-ORG-SCR-S42` best-row B-VAL mAP50–95 is 0.40360.

## 2. Material mismatch that must be resolved

The active DDP-B16 pair is MV00 = SCConv-Early, MV01 = SCConv-Early + VGRA-V1. Both are initialized using the **already trained Early best** checkpoint, exact SHA-256 `620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1`. The frozen `vgra_pretrained_boundary.py` confirms the source archive member is `ResEMA_single_module_runs/BORG-PT-S42-SCCONV-EARLY-E100/weights/best.pt`.

A new *plain YOLO11s* from COCO/pretrained stock weights followed by 100 epochs of B-TRAIN is **not** initialization-equivalent to MV00/MV01, whose starting Early model already saw GRAZPEDWRI-DX training data in an earlier 100-epoch experiment. Thus benchmark and architectural causal comparisons need separate interpretations. Freeze the plain YOLO11s initialization and prior-data-exposure protocol before launch. Do not mislabel unmatched historical initialization as fully controlled architecture ablation.

## 3. Proposed new plain baseline (name BL00 provisional, not released)

- Stock **YOLO11s**, not C3k2_TPSC, SCConv-Early or VGRA; architecture source to audit: `research/r9_training/frozen_source/ultralytics/cfg/models/11/yolo11.yaml` with YOLO11s `s` scaling, `nc=9`, normal Detect head.
- Match where scientifically valid: GRAZPEDWRI-DX Split-B TRAIN/VAL, patient integrity and 9 class order, frozen 14,227 TRAIN images / 6,496 pairs + 1,235 singles, 890 synchronized steps/epoch; operational 3,049 readable B-VAL images. Reuse the **paired distributed image sampler** as a controlled data-order condition, without VGRA visibility criterion.
- Fixed new-condition recipe from `research/r9_training/ddp_b16/ddp_verified_admission.py`: 2×T4; global batch16; imgsz1024; 100 epochs; seed42; SGD lr0 .01 lrf .01 momentum .937 weight decay .0005 cosine schedule; nbs64; AMP true; workers0; no mosaic/mixup/cutmix/copy-paste; deterministic; validation-only model selection.
- Pretraining decision REQUIRED before GPU launch. Possible controls: (a) plain YOLO11s from independently pinned standard starting weights for 100 epochs as a **separate same-DDP-protocol benchmark**; (b) separately named 100-epoch continuation from earlier `BASE-B-ORG-PT-S42` for **prior-training-exposure sensitivity**, acknowledging its best epoch is different and this still is not an exact state-matched ablation. Don't assume either comparison perfectly isolates the architecture.
- Preserve independent BL00 G4 state (model, optimizer, scheduler, EMA, AMP, RNG, sampler), rank0 B-VAL, immutable run identity, SHA-256 file receipts and strict native restoration. **Current** `ddp_worker.py`, `ddp_native_binding.py`, `ddp_g4_recovery.py`, `ddp_native_bootstrap_candidate.py` deliberately whitelist only MV00/MV01 and their SCConv-Early transfer. Removing SCConv from MV00 and reusing these unchanged is forbidden; build separately reviewed fail-closed BL00 contracts without editing frozen original C1 source.
- No B-TEST access before final method/threshold/model selection is frozen. Keep original MV runs untouched.

## 4. Engineering gates in order

1. Freeze scientific baseline role and initial checkpoint, exact hash and prior training exposure; verify original Split-B patient-aware membership.
2. Freeze stock YOLO11s architecture, nine-class transfer/head initialization; audit against Early architecture and pretrained provenance.
3. Implement minimal independent BL00 DDP paired loader, native detection loss/AMP and optimizer schedule, rank0 governed validation (no visibility loss), and terminal export.
4. Add independent BL00 identity/receipts/G4 persistent epoch-boundary recovery and exact optimizer/EMA/BN/scheduler/RNG restore.
5. CPU and real 2-process unit tests; two-T4 single-epoch bound acceptance; **off-session SHA G4 audit plus zero-update native restore**.
6. After passing, implement strict Kaggle Save & Run All continuous baseline notebook; independently verify saved output and resume each session; final 100-epoch B-VAL terminal audit.
7. Report historical six YOLO11s baselines separately and fairly characterize new matched/unmatched comparisons. B-TEST SEALED.

## 5. Latest authoritative training tracker (user-provided evidence)

| Arm | Completed | Next epoch | Optimizer updates authenticated | Latest G4 manifest SHA-256 |
|---|---:|---:|---:|---|
| MV00 — SCConv-Early | 89/100 | 90 | 20432 | `2ebf42201f37bf9ab065552ed5113b672f22dad65863e4f8a06a22447daef217` |
| MV01 — SCConv-Early + VGRA-V1 | 33/100 | 34 | 7972 | `098d87afbb74717b6ee8c349d335a3b2d63ad11eb21f554a1ef67e65b16470a7` |
| BL00 — proposed plain YOLO11s | 0/100 | N/A | N/A | NOT STARTED |

Those are **last confirmed** completed epoch counts; not evidence subsequent saved-version sessions have finished. B-TEST remains sealed.

Frozen science source commit: `f440d329d4544ec08ad63c1972bcf9a8b8a7eed8` (not engineering branch HEAD).
Pinned Git bundle: `/kaggle/input/datasets/awaishit/resema-ddp-b16-acceptance/RESEMA_DDP_B16_ACCEPTANCE.bundle` SHA-256 `4dc26b7277d58ed0b1fbcc1f4fac68aaf7ad417951e735bc8be2b040764861bb`.

## 6. Essential reading on branch `research/vgra-ddp-b16`

- `research/05_experiments/BASELINE_FREEZE_01_CLOSURE.md`
- `research/05_experiments/BASELINE_FREEZE_01_CANONICAL_FINAL_SUMMARY.csv`
- `research/05_experiments/records/BASE-B-ORG-PT-S42.json`
- `research/r9_training/frozen_source/ultralytics/cfg/models/11/yolo11.yaml`
- `research/r9_training/frozen_source/ultralytics/models/yolo/detect/vgra_pretrained_boundary.py`
- `research/r9_training/frozen_source/ultralytics/models/yolo/detect/early_paired_trainer.py`
- `research/r9_training/ddp_b16/ddp_verified_admission.py`
- `research/r9_training/ddp_b16/ddp_worker.py`
- `research/r9_training/ddp_b16/ddp_native_binding.py`
- `research/r9_training/ddp_b16/ddp_g4_recovery.py`
- `research/r9_training/ddp_b16/distributed_pair_sampler.py`
- `research/r9_training/ddp_b16/kaggle_mv00_resume_epoch090_pinned_verified.ipynb`
- `research/r9_training/ddp_b16/kaggle_mv01_resume_epoch034_sha_pinned.ipynb`
