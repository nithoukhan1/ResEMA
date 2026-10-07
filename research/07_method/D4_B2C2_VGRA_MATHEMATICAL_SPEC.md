# D4-B2C2 VGRA Mathematical Specification

## Status

`CANDIDATE SPECIFICATION FROZEN FOR IMPLEMENTATION`

Working name:

`VGRA = Visibility-State-Gated Cross-View Residual Assistance`

This is NOT the final paper architecture.
This document freezes the first implementable VGRA candidate for D4-C/D4-D verification.

## Scientific input

The candidate is constrained by:

- D3: SCConv-Early suppresses several FP modes but does not robustly improve net lesion recovery;
- D4-B2A: AP/LAT pairing is structurally dominant in Split-B;
- D4-B2B: annotation/model-error complementarity is MODERATE, not universal;
- D4-B2B: fracture has higher one-view-only TP asymmetry than the all-class average.

Therefore VGRA is intentionally a bounded residual assistant, not an always-on fused detector.

## Base detector

Base candidate:

`YOLO11s + C3k2_TPSC Early`

Canonical YAML reference:

`ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml`

Existing parameter count:

`9,570,093`

The Early architecture itself is not declared final.

## Pair unit

A valid pair is:

`patient_id + study_number + laterality`

with:
- one projection 1 image (AP/PA);
- one projection 2 image (lateral).

The two images share detector weights.

No cross-view spatial/box correspondence is assumed.

## Detection features

Let:

`F_v^l`

be the three feature maps supplied to the standard Detect head for view `v` and level
`l in {P3,P4,P5}`.

For YOLO11s, the scaled Detect-input channel widths are expected to be approximately:

- P3: 128
- P4: 256
- P5: 512

The implementation must derive actual channels from the parsed model rather than
hard-code them.

## 1. Per-view semantic descriptor

For each level:

`h_v^l = SiLU(LN(W_d^l GAP(F_v^l)))`

where:
- `GAP` = global average pooling;
- descriptor rank per level `r_d = 32`.

The view descriptor is:

`h_v = concat(h_v^P3, h_v^P4, h_v^P5)`

so:

`dim(h_v) = 96`.

## 2. Pair visibility predictor

Ordered pair context:

`u = concat(h_AP, h_LAT, abs(h_AP-h_LAT), h_AP*h_LAT)`

so `dim(u)=384`.

Pair MLP:

`p = SiLU(LN(W_1 u + b_1))`

with hidden dimension:

`d_pair = 128`.

Output:

`R = reshape(W_2 p + b_2, [C,4])`

where `C=9`.

For class `c`:

`q_c = softmax(R_c)`

with ordered states:

- state 0: `00` = absent from both;
- state 1: `10` = AP only;
- state 2: `01` = lateral only;
- state 3: `11` = present in both.

## 3. Ground-truth visibility target

For class `c`:

`a_c = 1` if AP image has >=1 GT object of class c, else 0.

`l_c = 1` if LAT image has >=1 GT object of class c, else 0.

State index:

`y_c = a_c + 2*l_c`

therefore:
- a=0,l=0 -> 0 (`00`);
- a=1,l=0 -> 1 (`10`);
- a=0,l=1 -> 2 (`01`);
- a=1,l=1 -> 3 (`11`).

## 4. Visibility loss

State-frequency weighting is derived from B-TRAIN exact paired studies only.

For class c and state s:

`raw_w[c,s] = 1 / sqrt(n[c,s] + 1)`

Normalize within each class so the four state weights have mean 1:

`w[c,s] = 4 * raw_w[c,s] / sum_t raw_w[c,t]`

Weighted visibility CE:

`L_vis = mean_{pair,class}( w[c,y_c] * CE(R_c, y_c) )`

Frozen initial auxiliary coefficient:

`lambda_vis = 0.25`

This coefficient may not be tuned before the first full candidate/control comparison.

## 5. Visibility state is semantically protected

The detection loss must not be allowed to redefine the meaning of the visibility states.

Therefore the residual gate uses:

`stopgrad(q_c)`

The visibility predictor is optimized by `L_vis`.

Detection gradients do not pass through the visibility probabilities used as gates.

This prevents the detector from "gaming" the state predictor merely to improve detection loss.

## 6. Target-specific visibility coefficient

For the AP target:

`k_AP,c = stopgrad(q_11,c - q_01,c)`

For the lateral target:

`k_LAT,c = stopgrad(q_11,c - q_10,c)`

Interpretation:

For AP:
- shared (`11`) -> positive assistance permitted;
- LAT-only (`01`) -> negative residual permitted;
- AP-only (`10`) -> neutral;
- neither (`00`) -> neutral.

For LAT:
- shared (`11`) -> positive assistance permitted;
- AP-only (`10`) -> negative residual permitted;
- LAT-only (`01`) -> neutral;
- neither (`00`) -> neutral.

The `00` state is deliberately neutral in candidate V1 to reduce the risk of suppressing
rare true positives from an overconfident study-level absence prediction.

## 7. Low-rank cross-view compatibility

For each level `l`, use low rank:

`r_x = 16`

Target local embedding:

`z_v,i^l = normalize(A_l F_v,i^l)`

Companion semantic embedding:

`g_barv^l = normalize(B_l h_barv)`

Learned class embedding:

`e_c^l in R^(r_x)`

normalized before use.

Define class-conditioned companion vector:

`t_barv,c^l = normalize(g_barv^l * e_c^l)`

Compatibility at spatial position/anchor i:

`m_v,i,c^l = ReLU( dot(z_v,i^l, t_barv,c^l) )`

Because both vectors are normalized:

`0 <= m <= 1`.

No AP/LAT spatial alignment is used.

## 8. Bounded residual classification update

Let:

`zeta_v,i,c^l`

be the native YOLO classification logit at level l.

Each level has a learnable scalar parameter `rho_l`, initialized to 0.

Bounded residual strength:

`beta_l = beta_max * tanh(rho_l)`

with:

`beta_max = 2.0`

Frozen candidate equation:

`zeta'_v,i,c^l = zeta_v,i,c^l + beta_l * k_v,c * m_v,i,c^l`

Properties:
- at initialization `rho_l=0`, VGRA is classification-logit identical to the base detector;
- box/DFL outputs are unchanged;
- residual magnitude is bounded;
- cross-view evidence cannot create a spatial box on its own because compatibility is
  anchored to the target view's local feature;
- view-exclusive states do not force positive agreement.

## 9. Box-regression firewall

The following remain exactly native/view-specific:

- box branch;
- DFL branch;
- box decoding;
- TaskAlignedAssigner geometry;
- NMS/evaluator.

VGRA does not transfer, average or align AP/LAT boxes.

## 10. Detection loss

For paired samples, both images use the normal detection loss independently.

Conceptually:

`L_det_pair = 0.5 * (L_det_AP + L_det_LAT)`

Total pair objective:

`L_pair = L_det_pair + lambda_vis * L_vis`

For unpaired samples:

`L_single = L_det`

No synthetic companion is constructed.

## 11. Missing-view fallback

If a valid companion is unavailable:

- no pair state is predicted;
- residual update is exactly disabled;
- `zeta' = zeta`;
- model behavior reduces to the Early single-view detector.

This must be verified numerically before training authorization.

## 12. Data coverage

Training must preserve all B-TRAIN images.

- exact AP/LAT pairs are processed as pair units;
- remaining images are processed as independent single-view units;
- no image may be silently dropped merely because it lacks a clean pair.

Validation likewise retains the complete governed operational validation set.

For a readable image whose companion is unavailable/unreadable, use the single-view fallback.

## 13. Parameter/efficiency budget

Before training authorization, the static implementation must satisfy:

- added trainable parameters <= 300,000 over Early;
- total candidate parameters <= 9,870,093;
- parameter increase <= approximately 3.14% over Early;
- VGRA-specific per-image compute overhead target <= 5% excluding the unavoidable fact
  that a paired study contains two detector forward passes.

Failure of the parameter cap blocks training.

The compute target is reported, not silently relaxed.

## 14. Initialization contract

Required:
- all standard Early weights transfer unchanged;
- VGRA is additive;
- `rho_l=0` at initialization;
- with residual disabled, detector logits/boxes must match the Early base within numerical tolerance;
- standard single-view path must remain available.

## 15. Candidate name scope

`VGRA` is a working method name.

Do not claim:
- first multi-view detector;
- first view-aware detector;
- first view-specific label method;
- first gated fusion method.

The candidate contribution is narrower and remains subject to experimental support.
