# 02 — Method Blueprint

## Working concept
**Anatomy-conditioned, transfer-preserving, patient-effective YOLO11**

The method is divided into mechanisms so each can be validated separately.

## M1 — TP-CDA
### Problem
Custom replacement blocks can disrupt transfer from a strong pretrained YOLO11s baseline.

### Solution
Keep the normal pretrained C3k2 path intact and add a lightweight residual detail adapter.

Equation:
`F_out = F_base + alpha * A_detail(F_base)`

`alpha` is initialized to zero, so at initialization `F_out == F_base`.

### Initial placement hypothesis
- after high-resolution P2/4 C3k2 feature stage
- after P3/8 C3k2 feature stage
- not initially at P5/32

### Candidate detail operators
Finalized only after D00:
- local high-pass residual
- depthwise 3x1
- depthwise 1x3
- optional dilated depthwise 3x3
- 1x1 fusion
- zero-initialized output projection

## M2 — APCF
### Problem
Radiographic appearance depends on projection and anatomical orientation.

### Dataset signal
GRAZPEDWRI-DX contains an `axis` line annotation: a two-point line along the forearm-bone axis. The dataset authors state it may help automatic alignment.

### Solution
1. Tiny auxiliary axis/orientation head from P3.
2. Combine predicted anatomy orientation with verified projection metadata.
3. Use context embedding to modulate P5->P4 and P4->P3 neck fusion.

### Identity principle
Fusion modulation initializes close to standard YOLO fusion behavior.

## M3 — PELT
### Problem
Rare-class box counts overstate independent information when many boxes/images come from the same child.

### Solution
Use unique positive-patient count as the fundamental rarity statistic:
`N_c = number of unique positive training patients for class c`

Training principles:
- patient-aware rare-case sampler
- maximum repeat factor per patient
- fixed epoch length
- optional mild capped effective-patient loss weighting only if sampler alone is insufficient
- optional geometry bins if D00 shows strong morphology imbalance

### Inference cost
Zero.

## Standard detection head
Keep the YOLO11 Detect head unchanged initially.

Only introduce a head/localization modification if D00 and M1-M3 show a persistent localization-specific bottleneck.

## Optional branch
True AP/LAT paired-view fusion.
Enter only if pair audit confirms enough usable studies and complementary errors.
