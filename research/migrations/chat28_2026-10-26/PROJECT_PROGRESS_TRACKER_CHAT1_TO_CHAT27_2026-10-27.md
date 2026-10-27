# Detection Project — Overall Progress Tracker (Chat 27 closure → Chat 28)

**Cutoff:** 27 October 2026  
**Research:** nine-class pediatric wrist-radiograph detection, GRAZPEDWRI-DX  
**Active control:** MV00 SCConv-Early YOLO11s  
**Active experimental arm:** MV01 SCConv-Early + VGRA-V1  
**Official training:** MV00 **0/100**, MV01 **0/100**  
**B-TEST:** **SEALED**; no access, diagnostics or model selection.

## A. Scientific and historical stages

| Stage | Status | Evidence/meaning |
|---|---|---|
| C3k2 + DySample + ResEMA V1/V2/V3 and Split-A/Split-B studies | Historical, preserve completely | Nested original Chat1–26 predecessor archive and tracked research records |
| Frozen patient-disjoint Split B | LOCKED | Train 14,227 / Val 3,050 listed (3,049 operational) / Test 3,050 SEALED |
| Corrected single-model screens: baseline, Early, SCConv4, DySample, EMA, combinations | Completed historical analyses | Metrics have **two different evaluator families**; never conflate |
| Paired AP/LAT visibility complementarity audit | Completed | 6,496 B-TRAIN clean pairs + 1,235 singles |
| VGRA-V1 mathematical and architectural contract | Frozen candidate | Zero-initialized bounded signed correction of classification logits; B-TRAIN-only vis labels |
| C1 independent-GPU nonofficial fresh epoch1 and authentic epoch2 off-session resume | PASS; nonofficial | October 21–22 Windows verified G4 archive hashes |
| Official 100 epochs on either arm | **NOT AUTHORIZED** | No official experiments, no final model |

## B. Active DDP-B16 execution engineering

C1 baseline commit: `fa6a7675daae385a04c7b77e69a9de29f9e1b2f3`  
DDP code-tested (before migration docs): `1b1696d3d77cf327452bcd40cb236c05c91098a2`  
Development branch: `research/vgra-ddp-b16` (single branch).  
Training condition is **new**: two physical Tesla T4 GPUs for *one* model, global batch 16, 8 images per rank, 890 synchronized train steps per epoch, seed42, 1024px, SGD and 100 epochs. The old C1 single-GPU batch8 acceptance must never be reused as DDP resume.

| Component | Actual status | Not-yet-demonstrated boundary |
|---|---|---|
| Complete pair-preserving sampler (6,496 pairs / 1,235 singles) | CPU tests PASS | Real-image loader on two CUDA ranks |
| Rank-local InfiniteDataLoader + epoch reset | CPU source tests PASS | Native training iterators under actual GPU |
| Exact-C1 5-file private source adapter / 336+7-byte source staging | CI PASS | End-to-end executable native DDP |
| Synthetic two-rank PyTorch/Gloo autograd | CPU PASS | Full YOLO/VGRA gradients and AMP on CUDA |
| Candidate AML/optimizer finite safeguards | Static CPU checks PASS | Actual AMP scaler/skip/finiteness equivalence |
| Fail-closed launcher and original inputs preflight | CI PASS, execution locked | Explicit authenticated `torchrun` child |
| Pair-aware rank0 B-VAL broadcast/3,049 cardinality | Two-rank CPU synthetic PASS | Real 3,049-image B-VAL actual inference |
| DDP full-state epoch recovery + SHA256 manifest | Two-rank CPU PASS | Independent off-session checkpoint transfer and genuine GPU resume |
| Native rank0 selected-best/last SHA + G4 epoch callback | Two-rank CPU PASS | Real native training and terminal export |
| Official MV00 / MV01 training | **0/100 each** | Explicit separate user approval after GPU acceptance |
| Final B-TEST | SEALED | Unlock only once after model selection/authorization |

## C. Exactly one next execution sequence

1. **Chat 28 begins:** read both archives together—the preserved earlier historical ZIP and the new Chat27 repository migration (or use locally merged cumulative full/portable archive). Confirm SHA256 and CI evidence. Treat scientific frozen C1 separately from unapproved DDP execution candidate.
2. **Native DDP startup/resume:** implement and authenticate `torchrun` per-rank construction with approved assignment manifest, visibility evidence, pretrained weights and private source stage. Replace incompatible C1 batch8 startup guard assumptions only in private DDP candidate, with equally strict tested lifecycle.
3. **Minimum nonofficial GPU acceptance:** one bounded MV00 and one MV01 two-rank T4x2 test, authentic pretrained transfer and AP/LAT backward, optimizer/AMP/EMA invariants, rank0 real B-VAL 3,049-image cardinality, epoch-boundary G4 and **independent fresh-session epoch2 resume**, externally pinned.
4. **Only after acceptance:** freeze one DDP release, prepare one notebook per arm, request explicit official 100-epoch authorization, start both arms FRESH, and track full G4 preservation. No official epochs exist yet.
5. **Science outcome:** compare frozen B-VAL mAP50–95 primary plus F1/recall/fracture and object-level checks. Conditional ablations later only on earned evidence; B-TEST stays sealed until final.

## D. Nonnegotiable restrictions

No other architecture, attention module, new dataset split, arbitrary parameter tuning or speculative tests. Never use B-TEST for development. Never claim CPU Gloo acceptance demonstrates CUDA training. Never claim official or clinical results from the historical two-epoch acceptance. Confirm Kaggle resource availability and follow platform terms. Maintain a single workstream tracker and migrate only at meaningful milestones.

## E. Migration provenance

The original prior full ZIP SHA256 is `942983c166536e78c02a4fe1c4e5d67b626a9809c78fddec3cb0c10c6ad1af3f`. The original prior portable ZIP SHA256 is `ffebe21e221f4b548369ca9e5a90cdfc1e160f51b70bb22796d4bf5a8d9504ce`. The GitHub-generated Chat27 snapshot ZIP SHA256 is `dc141f9ab37bc3221b5f8139a9b01e7edd7f3f51b484c75ff4839de7b410a238` (GitHub Actions run `40292351065`, 1,593 members verified). These are three distinct input archives; the original two are **not inside the GitHub snapshot**. A separate offline assembler can nest the actual verified predecessor bytes and Chat27 snapshot into cumulative FULL and PORTABLE ZIPs.

**Authoritative order:** real source/checkpoint evidence → verified pinned code and GitHub CI → current cumulative handoff → older historical summaries. All unknowns remain unknown.
