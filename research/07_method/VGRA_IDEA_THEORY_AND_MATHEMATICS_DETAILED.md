# VGRA: Visibility-State-Gated Cross-View Residual Assistance

## A Complete Conceptual and Mathematical Explanation of the VGRA V1 Research Idea

**Method status:** Frozen research candidate V1, not a finalized or empirically proven paper method.  
**Scope of this document:** Scientific motivation, conceptual mechanism, mathematical formulation, learning dynamics, illustrative examples, and limitations. Dataset-specific, coding, repository, experiment-management, and system-deployment details are intentionally excluded.  
**Terminology:** AP and LAT denote two complementary views (the ordered first and second projections); the mathematics applies to a view pair whose correspondence is known at the *study* level, without requiring pixel, region, or bounding-box alignment.

---

## 1. The central research idea

**VGRA (Visibility-State-Gated Cross-View Residual Assistance)** is a proposed mechanism that lets one image view provide **limited, class-specific semantic assistance** to the detector processing its companion view. Its key insight is that **a finding visible in one view need not be visible in the other**. Therefore, a multi-view detector should not always treat agreement as mandatory or transfer a detected abnormality directly across projections.

VGRA explicitly learns, **for each class**, which of four visibility states best describes a two-view pair:

1. **00:** The class is absent from both annotated views.
2. **10:** The class appears only in the AP view.
3. **01:** The class appears only in the LAT view.
4. **11:** The class appears in both views.

These predicted probabilities guide a small, learned correction to the **classification logits** of each view. The correction is calculated from (i) the target view's *own local features*, (ii) a *global semantic descriptor* of the companion view, (iii) a learned *class-specific compatibility* function, and (iv) a **bounded, signed residual strength**.

The other view **does not donate its bounding boxes**. Box regression, spatial geometry, and per-view localization remain independent.

### VGRA in one sentence

> Use jointly predicted, class-wise projection-visibility states to regulate a low-rank companion-view semantic signal that modifies only target-local classification confidence, while leaving localization view-specific and restoring the ordinary single-view detector whenever a companion is unavailable.

### The principal scientific question

**Can explicitly modeling whether a class should be supported by one view, the other, both, or neither improve cross-view use of evidence without imposing false agreement or transferring geometrically incompatible boxes?**

This is a research hypothesis, **not** a demonstrated improvement or clinical guarantee.

---

## 2. Why ordinary cross-view fusion is not enough

A conventional two-view fusion mechanism can be tempted to make two incorrect assumptions:

- **Assumption A — positive agreement:** If the companion view suggests a pathology, the target view should receive a positive detection boost.
- **Assumption B — geometric correspondence:** Evidence localized in one projection can be mapped, aligned, averaged, or transferred to the other projection.

Neither is universally justified.

### 2.1 Projection-dependent visibility

The same underlying finding may appear clearly in one projection and poorly or not at all in the other because of projection geometry, overlap, contrast, orientation, and representation. A positive class-level signal in one projection therefore does **not** imply that the target projection must contain an annotation of the same class.

### 2.2 Agreement is not the same as correctness

If two views disagree, the disagreement can represent legitimate projection-exclusive evidence rather than a model error. A simple *agree/disagree* target merges three distinct possibilities: AP-only, LAT-only, and absent-both.

Those distinctions matter because the appropriate effect on an AP target is not necessarily the appropriate effect on a LAT target.

### 2.3 The intended alternative

VGRA does not perform unconditional feature fusion. It gives each view its own predictions and computes a **small contextual classification residual**, informed by predicted class visibility and bounded to prevent the companion from arbitrarily dominating the target.

---

## 3. High-level conceptual structure

For a paired observation \(X_A,X_L\), let \(A\) represent AP and \(L\) represent LAT. Both use a shared detector feature extractor \(f_\theta\), but they maintain distinct feature tensors because they are distinct images:

\[
F_A^\ell=f_\theta^\ell(X_A),\qquad F_L^\ell=f_\theta^\ell(X_L),
\]

where \(\ell\) indexes the feature-pyramid levels, conventionally \(\ell\in\{P3,P4,P5\}\).

The complete VGRA pathway is:

\[
\boxed{
(F_A^\ell,F_L^\ell)
\longrightarrow
(h_A,h_L)
\longrightarrow
\big(q_{c,00},q_{c,10},q_{c,01},q_{c,11}\big)
\longrightarrow
(k_{A,c},k_{L,c})
}
\]

in parallel with

\[
\boxed{
\big(F_v^\ell,h_{\bar v},e_c^\ell\big)
\longrightarrow
m_{v,i,c}^\ell
\longrightarrow
\Delta \zeta_{v,i,c}^\ell
\longrightarrow
\zeta_{v,i,c}^{\ell\prime}
}
\]

Here:

- \(v\in\{A,L\}\) is the *target* view;
- \(\bar v\) means the *other/companion* view;
- \(c\) indexes a pathology/object class;
- \(i\) denotes a spatial position or candidate at pyramid level \(\ell\);
- \(q_{c,s}\) is the probability of pair-level visibility state \(s\) for class \(c\);
- \(k_{v,c}\) is the state-conditioned coefficient for the target view;
- \(m_{v,i,c}^\ell\) is nonnegative local–companion semantic compatibility;
- \(\zeta_{v,i,c}^\ell\) is the unmodified classification logit;
- \(\Delta\zeta_{v,i,c}^\ell\) is VGRA's additive correction.

**Important distinction:** The visibility predictor answers a *pair-level class-presence* question; the compatibility function determines *where* that class-level context may influence the target view. Neither task is equivalent to cross-view lesion localization or pixel registration.

---

## 4. Notation and mathematical objects

| Symbol | Meaning |
|---|---|
| \(X_A,X_L\) | Two complementary projections of one observation |
| \(F_v^\ell\in\mathbb{R}^{C_\ell\times H_\ell\times W_\ell}\) | Target/companion feature map at level \(\ell\) |
| \(h_v\in\mathbb{R}^{d_h}\) | Global semantic descriptor of view \(v\) |
| \(C\) | Number of detection classes |
| \(R_{c,s}\) | Unnormalized visibility-state logit for class \(c\), state \(s\) |
| \(q_{c,s}\) | Softmax probability of visibility state \(s\) for class \(c\) |
| \(a_c,\ell_c\in\{0,1\}\) | Class-presence indicators in the AP and LAT views |
| \(y_c\in\{0,1,2,3\}\) | Supervised visibility-state target |
| \(k_{v,c}\in[-1,1]\) | Signed, view-specific visibility coefficient |
| \(r_x\) | Rank of low-dimensional cross-view semantic compatibility |
| \(m_{v,i,c}^\ell\in[0,1]\) | Target-local, class-specific compatibility |
| \(\rho_\ell\) | Learnable scalar controlling signed residual strength |
| \(\beta_\ell\) | Bounded signed residual strength for pyramid level \(\ell\) |
| \(\zeta_{v,i,c}^\ell\) | Native classification logit |
| \(\zeta_{v,i,c}^{\ell\prime}\) | Logit after the VGRA correction |
| \(\lambda_{\mathrm{vis}}\) | Weight of supervised visibility loss |
| \(\operatorname{sg}(\cdot)\) | Stop-gradient operation |

The frozen V1 mathematical choices are:

\[
r_d=32,\quad d_h=3r_d=96,\quad d_{\mathrm{pair}}=128,\quad
r_x=16,\quad \beta_{\max}=2,\quad \lambda_{\mathrm{vis}}=0.25.
\]

These values define this **candidate formulation**; they are not universal constants or claims of optimality.

---

## 5. Stage I — Compute the semantic descriptor of each view

The goal of the descriptor is to summarize **what the entire view represents**, rather than identify the exact pixel or box corresponding to a lesion in the other projection.

For pyramid level \(\ell\), first globally average pool the feature tensor:

\[
\operatorname{GAP}(F_v^\ell)
=
\frac{1}{H_\ell W_\ell}
\sum_{i=1}^{H_\ell W_\ell}F_{v,i}^\ell.
\]

This produces a vector in \(\mathbb{R}^{C_\ell}\). Transform it with a learned projection and normalization:

\[
h_v^\ell
=
\operatorname{SiLU}
\left(
\operatorname{LN}
\left(
W_d^\ell\,\operatorname{GAP}(F_v^\ell)+b_d^\ell
\right)
\right),
\]

where \(W_d^\ell\in\mathbb{R}^{r_d\times C_\ell}\), \(r_d=32\), LN is layer normalization, and SiLU is the smooth nonlinear activation.

Combine information from the three scales:

\[
h_v=
\operatorname{Concat}
\left(
h_v^{P3},h_v^{P4},h_v^{P5}
\right),
\qquad h_v\in\mathbb{R}^{96}.
\]

### Why multi-scale semantic descriptors?

Small/high-resolution and large/coarse features describe different aspects of the image. A multi-scale descriptor allows the model to estimate class visibility from a broader semantic summary rather than a single local activation.

### What this descriptor deliberately does *not* do

It does not define a transformation between AP and LAT coordinates, estimate where an AP bounding box should fall in LAT, or prove the visibility of a particular lesion instance.

---

## 6. Stage II — Predict four class-wise visibility states

VGRA constructs an **ordered pair representation**:

\[
u=
\operatorname{Concat}
\left(
h_A,\;
h_L,\;
|h_A-h_L|,\;
h_A\odot h_L
\right).
\]

Here \(\odot\) is the elementwise product.

The terms serve different purposes:

- \(h_A\): semantic information from the first view;
- \(h_L\): semantic information from the second view;
- \(|h_A-h_L|\): magnitude of cross-view semantic disagreement;
- \(h_A\odot h_L\): elementwise joint activation/agreement.

Because the pair is **ordered**, the predictor can distinguish AP-only from LAT-only.

For \(d_h=96\),

\[
u\in\mathbb{R}^{4d_h}=\mathbb{R}^{384}.
\]

A small multi-layer perceptron computes

\[
p=\operatorname{SiLU}\!\left(
\operatorname{LN}(W_1u+b_1)
\right),\qquad p\in\mathbb{R}^{128},
\]

\[
R=\operatorname{reshape}(W_2p+b_2,\,[C,4]).
\]

For each class \(c\), use a four-state softmax:

\[
q_{c,s}=
\frac{\exp(R_{c,s})}
{\displaystyle\sum_{t\in\{00,10,01,11\}}\exp(R_{c,t})},
\quad
\sum_s q_{c,s}=1.
\]

For example, the vector

\[
q_c=(0.05,\ 0.70,\ 0.10,\ 0.15)
\]

means the model assigns 5% probability to absent-both, 70% to AP-only, 10% to LAT-only, and 15% to shared visibility, in that order.

**These are predicted model probabilities, not anatomical certainties.** A large probability can still be wrong or miscalibrated.

### Why four states instead of a binary shared/not-shared label?

A binary label would collapse

\[
10,\quad01,\quad00
\]

into one category, despite their very different directional implications. The four-state distribution preserves the target-specific distinction.

---

## 7. Stage III — Construct the supervised visibility target

The VGRA visibility target is **class-level and pair-level**, not box-level.

For class \(c\), define

\[
a_c=
\begin{cases}
1,&\text{if class }c\text{ is annotated in AP},\\
0,&\text{otherwise},
\end{cases}
\qquad
\ell_c=
\begin{cases}
1,&\text{if class }c\text{ is annotated in LAT},\\
0,&\text{otherwise}.
\end{cases}
\]

Map the pair of binary indicators to a categorical state index:

\[
\boxed{y_c=a_c+2\ell_c.}
\]

| AP presence \(a_c\) | LAT presence \(\ell_c\) | State notation | State index \(y_c\) | Interpretation |
|---:|---:|---|---:|---|
| 0 | 0 | 00 | 0 | Neither annotated view contains the class |
| 1 | 0 | 10 | 1 | AP-only |
| 0 | 1 | 01 | 2 | LAT-only |
| 1 | 1 | 11 | 3 | Both |

This supervision tells the auxiliary predictor which visibility pattern is represented in the annotations. It is *not* a per-lesion correspondence map.

### Important scientific qualification

The label **00** means *not annotated as present in either image*. It must not be interpreted as proof that a disease is biologically absent. Annotation coverage and projection-specific ambiguity remain relevant limitations.

---

## 8. Stage IV — Learn visibility with weighted cross-entropy

Four visibility states can have very different frequencies. A state predictor trained with ordinary cross-entropy may overfocus on frequent states and underlearn rare but important view-exclusive states.

Let \(n_{c,s}\) denote the observed training count of class \(c\) in state \(s\). Define unnormalized weights:

\[
\widetilde w_{c,s}=\frac{1}{\sqrt{n_{c,s}+1}}.
\]

Normalize the four weights within each class so that their arithmetic mean equals one:

\[
\boxed{
w_{c,s}
=
\frac{4\,\widetilde w_{c,s}}
{\displaystyle\sum_{t\in\{00,10,01,11\}}\widetilde w_{c,t}}
}
\]

Consequently,

\[
\frac14\sum_s w_{c,s}=1.
\]

The \(+1\) prevents division by zero; the square root makes the upweighting of rare states less aggressive than inverse-frequency weighting.

For paired observations \(j=1,\ldots,P\), define the class-wise visibility loss:

\[
\begin{aligned}
\mathcal L_{\mathrm{vis}}
&=
\frac{1}{PC}\sum_{j=1}^{P}\sum_{c=1}^{C}
w_{c,y_{j,c}}
\operatorname{CE}
\big(R_{j,c,:},y_{j,c}\big)\\[3pt]
&=
-\frac{1}{PC}\sum_{j=1}^{P}\sum_{c=1}^{C}
w_{c,y_{j,c}}\log q_{j,c,y_{j,c}}.
\end{aligned}
\]

This gives every pair–class example a supervised four-state objective while increasing the relative influence of rarer states.

### What this loss does and does not guarantee

It encourages the probability vector \(q_{c,:}\) to describe annotated visibility states. It does **not** by itself ensure the probabilities are calibrated, ensure that a detection is anatomically correct, or ensure that adding a gated residual improves detection.

---

## 9. Stage V — Convert visibility probabilities into target-specific gates

This is the conceptual heart of VGRA.

The AP detector and the LAT detector should **not** use the same assistance coefficient: a finding that is AP-only should not automatically boost the LAT prediction.

VGRA uses

\[
\boxed{
k_{A,c}
=
\operatorname{sg}\big(q_{c,11}-q_{c,01}\big)
}
\]

and

\[
\boxed{
k_{L,c}
=
\operatorname{sg}\big(q_{c,11}-q_{c,10}\big).
}
\]

Because each \(q\) is in \([0,1]\), both coefficients satisfy

\[
-1\le k_{A,c}\le1,\qquad -1\le k_{L,c}\le1.
\]

### Interpreting the coefficients

For the **AP target**:

- shared visibility (11) raises \(k_{A,c}\);
- LAT-only visibility (01) lowers \(k_{A,c}\);
- AP-only visibility (10) has **zero direct signed contribution**;
- absent-both (00) has **zero direct signed contribution**.

For the **LAT target**:

- shared visibility (11) raises \(k_{L,c}\);
- AP-only visibility (10) lowers \(k_{L,c}\);
- LAT-only visibility (01) has **zero direct signed contribution**;
- absent-both (00) has **zero direct signed contribution**.

In the idealized case of a one-hot visibility distribution, the gates are:

| Visibility state | \(k_{A,c}\) | \(k_{L,c}\) | Nominal interpretation when \(\beta_\ell>0\) |
|---|---:|---:|---|
| 00 — neither | 0 | 0 | No cross-view residual in either view |
| 10 — AP-only | 0 | −1 | AP remains neutral; LAT may receive a negative correction |
| 01 — LAT-only | −1 | 0 | AP may receive a negative correction; LAT remains neutral |
| 11 — both | +1 | +1 | Both may receive positive corrections |

The word **nominal** matters because the learned residual strength \(\beta_\ell\) is **signed** and may become negative. The full prediction behavior must be interpreted using the product \(\beta_\ell k_{v,c}m_{v,i,c}^\ell\), not the gate alone.

### Why is the 00 state neutral?

V1 deliberately does **not** enforce strong suppression when both views are predicted negative. This reduces the risk that a mistakenly overconfident *absent-both* prediction directly suppresses a rare true-positive detection through the residual.

The state can still influence the softmax normalization of the remaining probabilities indirectly, but it has no explicit signed term in either gate.

---

## 10. Stage VI — Compute target-local, class-conditioned cross-view compatibility

The visibility coefficient is **global to a class and view**. If applied uniformly across all spatial positions, it could indiscriminately increase or decrease scores in irrelevant regions.

VGRA therefore introduces a second condition: the target's **local feature** must be semantically compatible with information from the companion view.

### 10.1 Target-local low-rank embedding

At location \(i\), level \(\ell\), define

\[
z_{v,i}^{\ell}
=
\operatorname{normalize}
\big(A_\ell F_{v,i}^{\ell}\big),
\]

where the learned projection maps the native feature dimension \(C_\ell\) to a smaller rank \(r_x=16\):

\[
A_\ell:\mathbb{R}^{C_\ell}\to\mathbb{R}^{16}.
\]

The resulting \(z_{v,i}^{\ell}\in\mathbb{R}^{16}\) encodes the **target view's own local evidence**.

### 10.2 Companion semantic embedding

Project the companion descriptor:

\[
g_{\bar v}^{\ell}
=
\operatorname{normalize}
\big(B_\ell h_{\bar v}\big)
\in\mathbb{R}^{16}.
\]

This provides a compact description of **what the other view supports semantically**, not where a box appears.

### 10.3 Class-conditioned companion vector

Learn a class embedding \(e_c^\ell\in\mathbb{R}^{16}\). Form

\[
t_{\bar v,c}^{\ell}
=
\operatorname{normalize}
\left(
g_{\bar v}^{\ell}
\odot
\operatorname{normalize}(e_c^\ell)
\right).
\]

The elementwise product makes the companion signal **class-conditioned**: the same companion descriptor can yield different semantic templates for different classes.

### 10.4 Compatibility with the target view

Define the local compatibility

\[
\boxed{
m_{v,i,c}^{\ell}
=
\max\left(
0,\;
\left\langle
z_{v,i}^{\ell},
t_{\bar v,c}^{\ell}
\right\rangle
\right)
}
\]

with a unit upper clamp if required for strict numerical bounds.

Since the vectors are normalized,

\[
\boxed{0\le m_{v,i,c}^{\ell}\le1.}
\]

**Interpretation:**

- \(m\approx0\): the target local features do not align positively with the class-conditioned companion semantics; little or no correction is allowed;
- \(m\approx1\): strong positive semantic compatibility; the full visibility-conditioned correction becomes possible;
- intermediate \(m\): a proportionally smaller correction.

### Why low rank?

A low-rank interaction learns a compact semantic relationship between views without building a dense cross-view spatial attention matrix or requiring a mapping between their coordinate systems.

### Why local anchoring?

A companion image cannot independently create a new target-view location. The companion's influence must pass through \(z_{v,i}^\ell\), which comes from the **target image at location \(i\)**.

This reduces unsupported cross-view score transfer, but does **not** mathematically guarantee freedom from false-positive increases.

---

## 11. Stage VII — Apply the bounded, signed logit residual

Let \(\zeta_{v,i,c}^{\ell}\) be the baseline detector's classification logit for class \(c\) at target location \(i\), pyramid level \(\ell\).

VGRA learns a scalar \(\rho_\ell\) for each pyramid level and transforms it into a bounded strength:

\[
\boxed{
\beta_\ell=\beta_{\max}\tanh(\rho_\ell),
\qquad \beta_{\max}=2.
}
\]

Therefore,

\[
-2<\beta_\ell<2.
\]

The actual correction is

\[
\boxed{
\Delta\zeta_{v,i,c}^{\ell}
=
\beta_\ell\;
k_{v,c}\;
m_{v,i,c}^{\ell}.
}
\]

The output classification logit is

\[
\boxed{
\zeta_{v,i,c}^{\ell\prime}
=
\zeta_{v,i,c}^{\ell}
+
\beta_\ell\,
k_{v,c}\,
m_{v,i,c}^{\ell}.
}
\]

Equivalently, separating the views:

\[
\boxed{
\zeta_{A,i,c}^{\ell\prime}
=
\zeta_{A,i,c}^{\ell}
+
\beta_\ell
\,\operatorname{sg}(q_{c,11}-q_{c,01})
\,m_{A,i,c}^{\ell}
}
\]

\[
\boxed{
\zeta_{L,i,c}^{\ell\prime}
=
\zeta_{L,i,c}^{\ell}
+
\beta_\ell
\,\operatorname{sg}(q_{c,11}-q_{c,10})
\,m_{L,i,c}^{\ell}.
}
\]

These are the central VGRA V1 equations.

### 11.1 A strict bound on the correction

From

\[
|\beta_\ell|<2,\qquad |k_{v,c}|\le1,\qquad 0\le m_{v,i,c}^\ell\le1,
\]

it follows that

\[
\boxed{
|\Delta\zeta_{v,i,c}^{\ell}|\le2.
}
\]

This is a bound in **logit units**, not a fixed bound of 2 percentage points on probability.

### 11.2 Why modify a logit rather than a bounding box?

The logit expresses class evidence associated with an existing target-view candidate. VGRA can revise that score using complementary context **without inventing a companion-derived box**.

### 11.3 The important signed-strength qualification

The frozen V1 formulation uses \(\tanh(\rho_\ell)\), **not** a positive-only activation.

- If \(\beta_\ell>0\), a positive \(k\) raises the logit and a negative \(k\) lowers it.
- If \(\beta_\ell=0\), VGRA makes no residual change.
- If \(\beta_\ell<0\), **those nominal directions reverse**.

Consequently, V1 **cannot honestly be described as guaranteeing positive help for shared visibility or guaranteed suppression for view-exclusive visibility**.

The scientifically accurate description is:

> **A bounded, learned, signed, visibility-conditioned classification-logit residual whose polarity is not constrained to remain clinically intuitive.**

This signed-\(\beta\) interpretation is an approved aspect of V1, rather than an error to be silently changed.

---

## 12. What happens to predicted class probabilities?

If the native class logit is \(\zeta\) and a sigmoid is applied to obtain a probability-like class score, the original and adjusted scores are

\[
p=\sigma(\zeta)=\frac{1}{1+e^{-\zeta}},
\qquad
p'=\sigma(\zeta+\Delta\zeta).
\]

The correction affects the **odds** multiplicatively:

\[
\boxed{
\frac{p'}{1-p'}
=
e^{\Delta\zeta}
\frac{p}{1-p}.
}
\]

For the bound \(|\Delta\zeta|\le2\), the odds multiplier lies between approximately

\[
e^{-2}\approx0.135
\qquad\text{and}\qquad
e^2\approx7.389.
\]

This illustrates how a bounded logit correction can still materially change confidence. It does **not** establish statistical calibration or the correctness of the resulting prediction.

---

## 13. Worked numerical examples

The following numbers are **illustrations of the equations**, not experimental results. For simplicity take a single class and spatial location, \(\beta_\ell=+1\), \(m_{v,i,c}^{\ell}=0.8\), and baseline logit \(\zeta=0\), so the starting sigmoid score is \(0.5\).

### Example A — The class is probably visible in both views

Suppose

\[
q_c=(q_{00},q_{10},q_{01},q_{11})
=(0.05,0.10,0.15,0.70).
\]

Then

\[
k_{A,c}=0.70-0.15=0.55,
\qquad
k_{L,c}=0.70-0.10=0.60.
\]

The corrections are

\[
\Delta\zeta_A=1(0.55)(0.8)=+0.44,
\]

\[
\Delta\zeta_L=1(0.60)(0.8)=+0.48.
\]

If both original logits were zero, the scores become approximately

\[
p'_A=\sigma(0.44)\approx0.608,
\qquad
p'_L=\sigma(0.48)\approx0.618.
\]

**Meaning:** When shared visibility is strongly predicted, compatible local candidates in *both* views receive positive residuals **when \(\beta_\ell>0\)**.

### Example B — The class is probably AP-only

Let

\[
q_c=(0.10,0.70,0.10,0.10).
\]

Then

\[
k_{A,c}=0.10-0.10=0,
\qquad
k_{L,c}=0.10-0.70=-0.60.
\]

Thus

\[
\Delta\zeta_A=0,
\qquad
\Delta\zeta_L=-0.48.
\]

For zero baseline logits,

\[
p'_A=0.500,
\qquad
p'_L=\sigma(-0.48)\approx0.382.
\]

**Meaning:** A likely AP-exclusive class does not directly boost AP through the VGRA gate and may lower a compatible LAT candidate, assuming positive residual strength.

### Example C — The class is probably LAT-only

Let

\[
q_c=(0.05,0.05,0.80,0.10).
\]

Then

\[
k_{A,c}=0.10-0.80=-0.70,
\qquad
k_{L,c}=0.10-0.05=+0.05.
\]

Therefore

\[
\Delta\zeta_A=-0.56,
\qquad
\Delta\zeta_L=+0.04.
\]

**Meaning:** The AP target receives a substantial negative correction under positive \(\beta_\ell\), while the LAT target receives only a small positive one because the predicted state distribution is not exactly one-hot.

### Example D — Both views are probably class-absent

Let

\[
q_c=(0.90,0.04,0.04,0.02).
\]

Then

\[
k_{A,c}=0.02-0.04=-0.02,\quad
k_{L,c}=0.02-0.04=-0.02.
\]

Corrections are small:

\[
\Delta\zeta_A=\Delta\zeta_L=-0.016.
\]

**Meaning:** The absent-both probability \(q_{00}\) is **not directly subtracted** from either target. Its dominance typically leaves the remaining differences small, but the exact coefficients depend on all four softmax probabilities.

### Example E — Signed \(\beta\) reverses the direction

Reuse Example B but set \(\beta_\ell=-1\). Now

\[
\Delta\zeta_L=(-1)(-0.60)(0.8)=+0.48.
\]

The AP-only visibility evidence now **raises**, rather than lowers, the LAT class logit.

This is not a hypothetical implementation mistake. It is a direct implication of V1's approved signed-\(\beta\) mathematics. A paper must report it transparently.

### Example F — No local compatibility, no effect

Regardless of the predicted visibility state or signed strength, when

\[
m_{v,i,c}^\ell=0,
\]

then

\[
\Delta\zeta_{v,i,c}^\ell=0.
\]

Thus the companion cannot modify that particular target candidate through the VGRA residual without positive local semantic compatibility.

---

## 14. Why the residual is initialized to zero

The parameter \(\rho_\ell\) is initialized as

\[
\rho_\ell^{(0)}=0.
\]

It follows that

\[
\beta_\ell^{(0)}=2\tanh(0)=0
\]

and therefore

\[
\boxed{
\zeta_{v,i,c}^{\ell\prime(0)}
=
\zeta_{v,i,c}^{\ell(0)}.
}
\]

So the **classification-logit residual is exactly inactive at initialization**; ordinary box predictions are also unchanged by VGRA.

This provides a conservative starting point: the detector does not begin with randomly generated cross-view classification perturbations.

### How can a zero-initialized residual ever learn?

Differentiate the residual with respect to \(\rho_\ell\):

\[
\frac{\partial\Delta\zeta}{\partial\rho_\ell}
=
\beta_{\max}
\big(1-\tanh^2\rho_\ell\big)\,
k_{v,c}\,m_{v,i,c}^{\ell}.
\]

At initialization,

\[
\left.
\frac{\partial\Delta\zeta}{\partial\rho_\ell}
\right|_{\rho_\ell=0}
=
2\,k_{v,c}\,m_{v,i,c}^{\ell}.
\]

Therefore the gradient for the residual-strength parameter can be nonzero **even though the residual itself starts at zero**.

The low-rank compatibility components have no *detection-residual* gradient through multiplication by \(\beta_\ell\) at the exact zero-strength starting point; such gradients can emerge once \(\beta_\ell\) moves away from zero. The auxiliary visibility loss can train the visibility pathway and shared descriptor features from the beginning.

**Qualification:** Exact logit identity at initialization does not imply that subsequent parameter updates keep the detector equal to the single-view baseline. That is the purpose of learning and must be evaluated experimentally.

---

## 15. The stop-gradient firewall: keeping visibility semantics meaningful

Recall that the target-specific coefficient uses

\[
k_{v,c}=\operatorname{sg}(f_v(q_c)),
\]

where \(f_A(q)=q_{11}-q_{01}\) and \(f_L(q)=q_{11}-q_{10}\).

The stop-gradient operator means that, for the **residual-gating route**,

\[
\frac{\partial k_{v,c}}{\partial q_{c,s}}=0.
\]

Thus detection-loss gradients do **not** directly push the visibility predictor's probabilities around through their use as residual gates.

### Why this matters

Without stop-gradient, detection optimization might pressure the model to predict whatever visibility state yields a convenient logit change, regardless of the annotated pair-level state. The supervised state predictor could thereby lose its intended meaning.

VGRA instead separates roles:

- **Visibility supervision** teaches \(q_{c,s}\) what the four labels mean.
- **Detection supervision** teaches whether and where the class-logit residual helps the detector, through its compatibility pathway and signed strength.

### Important nuance: gradients are not fully isolated

The firewall applies **to the gate's dependence on \(q\)**. It is not a claim that VGRA blocks all detection gradients from all shared features.

For example, the companion descriptor \(h_{\bar v}\) also enters the compatibility function \(m\), and detection loss can optimize that route when the residual strength is nonzero. Shared feature representations can receive gradient contributions from both the detection and visibility objectives.

The accurate statement is:

> Stop-gradient protects the **semantic interpretation of the visibility-probability gate** from direct detection-loss gradients through that gate; it does not make every VGRA representation mathematically independent.

---

## 16. Joint learning objective

VGRA preserves the detector's ordinary per-view detection objective and adds a separate, supervised class-visibility objective for valid paired views.

### 16.1 The standard detection objective is retained

For the two views, let

\[
\mathcal L_{\mathrm{det},A},\qquad
\mathcal L_{\mathrm{det},L}
\]

be the normal detection losses (including whatever classification and localization terms the underlying detector ordinarily uses).

Conceptually, the paired detection term is

\[
\mathcal L_{\mathrm{det,pair}}
=
\frac12\left(
\mathcal L_{\mathrm{det},A}
+
\mathcal L_{\mathrm{det},L}
\right).
\]

The paired joint objective is

\[
\boxed{
\mathcal L_{\mathrm{pair}}
=
\mathcal L_{\mathrm{det,pair}}
+
\lambda_{\mathrm{vis}}\,\mathcal L_{\mathrm{vis}},
\qquad\lambda_{\mathrm{vis}}=0.25.
}
\]

### 16.2 A view without a companion

For a standalone image, no artificial visibility target or companion representation is fabricated:

\[
\boxed{
\mathcal L_{\mathrm{single}}=
\mathcal L_{\mathrm{det,single}}.
}
\]

### 16.3 General mixed-batch interpretation

Suppose there are \(N\) images and \(P\) complete pairs among them, so \(2P\le N\). A convenient *per-image-normalized* statement of the combined objective is

\[
\boxed{
\mathcal L_{\mathrm{batch}}
=
\frac{1}{N}\sum_{i=1}^{N}\mathcal L_{\mathrm{det},i}
+
\lambda_{\mathrm{vis}}
\frac{2P}{N}\,
\mathcal L_{\mathrm{vis}}.
}
\]

Here \(\mathcal L_{\mathrm{vis}}\) is averaged over the \(P\times C\) pair–class targets.

The factor

\[
\alpha_{\mathrm{pair}}=\frac{2P}{N}
\]

makes the visibility term proportional to the **image fraction belonging to complete pairs** rather than assigning a full visibility penalty to a batch composed mostly of single-view images.

- If all images belong to pairs, \(2P=N\), so \(\alpha_{\mathrm{pair}}=1\).
- If no complete pairs exist, \(P=0\), and the auxiliary term is exactly zero.
- If half the images belong to pairs, \(\alpha_{\mathrm{pair}}=\tfrac12\).

This is a mathematical description of the loss's relative weighting; exact unnormalized loss conventions depend on the ordinary detector. The VGRA hypothesis does **not** change the definition of box regression, distributional box loss, or localization assignment.

---

## 17. The box-regression firewall

VGRA is explicitly a **classification-only** cross-view mechanism.

Let a standard detector produce

\[
\left(
\zeta_{v,i,c}^\ell,\;
b_{v,i}^\ell,\;
d_{v,i}^\ell
\right),
\]

where \(b\) denotes box-regression output and \(d\) denotes any distributional representation used for localization.

With VGRA:

\[
\boxed{
\left(
\zeta_{v,i,c}^{\ell\prime},\;
b_{v,i}^{\ell\prime},\;
d_{v,i}^{\ell\prime}
\right)
=
\left(
\zeta_{v,i,c}^\ell+\Delta\zeta_{v,i,c}^\ell,\;
b_{v,i}^\ell,\;
d_{v,i}^\ell
\right).
}
\]

Therefore

\[
b_{v,i}^{\ell\prime}=b_{v,i}^\ell,
\qquad
d_{v,i}^{\ell\prime}=d_{v,i}^\ell
\]

**for a fixed set of features and model parameters in a given forward pass**.

This means VGRA introduces **no direct cross-view box residual**. It does not:

- transfer a box from one projection to the other;
- assume pixelwise AP/LAT alignment;
- average paired-view box coordinates;
- require a lesion-instance matching function across views;
- supply a spatial localization prediction from the companion alone.

**Subtle but important:** Because training updates shared features, a trained VGRA model's boxes may differ from a separately trained baseline model's boxes. The firewall guarantees the *direct VGRA residual operation* leaves box outputs untouched; it does not guarantee that separately optimized models produce identical boxes forever.

---

## 18. Exact single-view fallback

If the target view has no valid companion, the model must recover the ordinary single-view path.

Let \(\mathbb{1}_{\mathrm{pair}}\) indicate that a valid companion exists. The residual can be written abstractly as

\[
\Delta\zeta_{v,i,c}^\ell
=
\mathbb{1}_{\mathrm{pair}}\,
\beta_\ell k_{v,c}m_{v,i,c}^\ell.
\]

If the companion is missing,

\[
\mathbb{1}_{\mathrm{pair}}=0
\quad\Longrightarrow\quad
\boxed{\zeta_{v,i,c}^{\ell\prime}=\zeta_{v,i,c}^\ell.}
\]

No four-state pair probability is required, no synthetic second image is constructed, and no auxiliary visibility objective is evaluated for that single view.

**Conceptual significance:** Cross-view context is an optional, bounded supplement to an already capable single-view detector, not a prerequisite for producing detections.

---

## 19. How VGRA differs from related families of ideas

### 19.1 Not ordinary early fusion

Early fusion combines signals before or within the principal feature representation, potentially making the two-view interaction pervasive. VGRA instead retains independent target-view prediction streams and introduces a **limited residual to class logits**.

### 19.2 Not late decision voting

Simple late fusion might average probabilities, use the maximum score, or combine detected boxes. VGRA does not merely vote between predictions: it learns **visibility states and target-local compatibility** to determine whether a class-logit adjustment is appropriate.

### 19.3 Not cross-view spatial attention or registration

VGRA uses a **global semantic descriptor** of the companion and **local semantic features** of the target. It does not compute a full AP-to-LAT spatial correspondence matrix.

### 19.4 Not mutual knowledge distillation

VGRA is not defined by a teacher–student consistency target or by forcing one view's detector to imitate the other view's predictions. It uses explicit **four-state pair-level supervision** and a controlled classification residual.

### 19.5 Not always-positive assistance

Because \(\beta_\ell\) is signed and learned, VGRA can produce positive, negative, or zero corrections, including polarity reversals. The name “assistance” refers to the *hypothesized value of contextual information*, not a guaranteed positive direction at every candidate.

---

## 20. Why the combination is scientifically distinctive

The claimed research contribution should be framed as the **specific combination of constraints**, not as inventing multi-view learning or gating itself.

The substantive VGRA V1 combination is:

1. **Four-state, per-class visibility supervision** rather than binary paired-view agreement.
2. **Target-specific AP/LAT coefficients**, so a view-exclusive state has different meanings for the two targets.
3. **A stop-gradient visibility gate**, separating state-label semantics from direct detection-driven gate manipulation.
4. **Class-conditioned low-rank compatibility**, coupling the companion's semantic descriptor to the target's *local* feature.
5. **A zero-initialized bounded signed classification residual**, so cross-view influence begins disabled and remains finite.
6. **A localization firewall**, prohibiting direct cross-view box transfer or geometric alignment.
7. **Exact missing-view fallback**, retaining standard single-image behavior when no companion exists.

These ingredients are proposed to act together. **Their individual concepts are not necessarily new**, and the combination's publishable novelty depends on contemporaneous prior art, reproducible evidence, and controlled scientific comparisons.

---

## 21. What the mathematics guarantees—and what it does not

| Property | Mathematically supported by V1? | Qualification |
|---|---|---|
| Four distinct visibility states | **Yes** | They describe annotated class presence at pair level |
| Separate AP and LAT gate values | **Yes** | Coefficients depend on different state probabilities |
| Class-specific residual correction | **Yes** | Both the gate and compatibility depend on class |
| Nonnegative local compatibility | **Yes** | \(m\in[0,1]\) |
| Bounded logit residual | **Yes** | \(|\Delta\zeta|\le2\) in V1 |
| Zero residual at initialization | **Yes** | \(\rho_\ell=0\Rightarrow\beta_\ell=0\) |
| No direct modification of box/DFL branch | **Yes** | A direct-forward-path invariant, not identical learned boxes across models |
| Exact missing-view residual fallback | **Yes, by design** | No companion means no correction |
| Shared-visible always increases confidence | **No** | Negative \(\beta_\ell\) reverses polarity |
| View-exclusive always suppresses the other view | **No** | Also depends on signed \(\beta_\ell\), probabilities, and \(m\) |
| Four-state probabilities are calibrated | **No** | Requires separate measurement |
| Cross-view semantics are necessarily anatomically correct | **No** | Pair-level representation is learned from supervision |
| VGRA improves sensitivity, precision, AP, or clinical outcomes | **Not guaranteed** | Empirical questions requiring rigorous controls |

---

## 22. Key assumptions and limitations

### 22.1 Visibility is class-level, not instance-level

A class can appear multiple times in a view. VGRA knows whether there is at least one labeled example of the class in each projection; it does **not** establish which individual lesion corresponds to which.

### 22.2 Annotations are an imperfect proxy for visibility

If an item is not annotated in a projection, the mathematical label records *annotation absence*. Projection occlusion, missed annotation, and ambiguous appearance are not automatically distinguishable.

### 22.3 Incorrect visibility probabilities can hurt

If \(q_{c,s}\) strongly favors the wrong state, the gate can direct a residual in an unhelpful direction. The stop-gradient firewall protects the supervised interpretation but does not make the classifier infallible.

### 22.4 Signed residual strength complicates interpretability

The learned sign of \(\beta_\ell\) can invert the intuitive meaning of nominal assist/suppress state coefficients. Therefore, downstream interpretation requires inspecting \(\beta_\ell\), \(k_{v,c}\), and \(m_{v,i,c}^\ell\) jointly.

### 22.5 Local semantic compatibility is not proof of lesion correspondence

A positive normalized dot product means the learned low-dimensional vectors are aligned. It is not a proof that the companion view depicts the same specific lesion at an anatomically corresponding position.

### 22.6 A bounded score change can still affect decision thresholds

Although VGRA bounds \(\Delta\zeta\), a modest logit shift can change whether a detection crosses a confidence threshold or survives downstream competition.

### 22.7 The auxiliary task can compete with detection learning

Visibility prediction and object detection share aspects of feature learning. Even with gate detachment, the auxiliary objective could help, have no effect, or hurt, depending on the features and objective interactions.

### 22.8 Scientific effectiveness remains unproven by algebra

The equations establish a coherent and testable mechanism. They do **not** establish improved recall, fewer false positives, better calibration, positive clinical consequences, or generalization. Those require independent empirical evidence.

---

## 23. Compact derivation from the full pair to the final scores

For readers who want the entire method in one uninterrupted sequence, let the paired features be \(\{F_A^\ell,F_L^\ell\}_{\ell=1}^{3}\).

**Step 1 — View descriptors**

\[
h_v^\ell
=
\operatorname{SiLU}
\big(
\operatorname{LN}(W_d^\ell\operatorname{GAP}(F_v^\ell)+b_d^\ell)
\big),
\quad
h_v=\operatorname{Concat}_\ell h_v^\ell.
\]

**Step 2 — Ordered context and four-state probabilities**

\[
u=[h_A,h_L,|h_A-h_L|,h_A\odot h_L],
\]

\[
R=\operatorname{reshape}
\left(
W_2\operatorname{SiLU}(\operatorname{LN}(W_1u+b_1))+b_2,
[C,4]
\right),
\qquad
q_{c,:}=\operatorname{softmax}(R_{c,:}).
\]

**Step 3 — Visibility labels**

\[
y_c=a_c+2\ell_c,\qquad a_c,\ell_c\in\{0,1\}.
\]

**Step 4 — Supervised visibility objective**

\[
\widetilde w_{c,s}=(n_{c,s}+1)^{-1/2},
\quad
w_{c,s}=
\frac{4\widetilde w_{c,s}}{\sum_t\widetilde w_{c,t}},
\]

\[
\mathcal L_{\mathrm{vis}}
=
-\frac{1}{PC}\sum_{j,c}
w_{c,y_{j,c}}\log q_{j,c,y_{j,c}}.
\]

**Step 5 — Target-specific visibility gates**

\[
k_{A,c}=\operatorname{sg}(q_{c,11}-q_{c,01}),
\quad
k_{L,c}=\operatorname{sg}(q_{c,11}-q_{c,10}).
\]

**Step 6 — Target-local companion compatibility**

\[
z_{v,i}^\ell=\operatorname{norm}(A_\ell F_{v,i}^\ell),
\qquad
g_{\bar v}^\ell=\operatorname{norm}(B_\ell h_{\bar v}),
\]

\[
t_{\bar v,c}^\ell
=
\operatorname{norm}
\big(g_{\bar v}^\ell\odot\operatorname{norm}(e_c^\ell)\big),
\]

\[
m_{v,i,c}^\ell
=
\operatorname{ReLU}
\langle z_{v,i}^\ell,t_{\bar v,c}^\ell\rangle
\in[0,1].
\]

**Step 7 — Zero-initialized, signed bounded residual**

\[
\beta_\ell=2\tanh(\rho_\ell),\qquad\rho_\ell^{(0)}=0,
\]

\[
\boxed{
\zeta_{v,i,c}^{\ell\prime}
=
\zeta_{v,i,c}^\ell
+
\beta_\ell\,
k_{v,c}\,
m_{v,i,c}^\ell.
}
\]

**Step 8 — Combined learning**

\[
\boxed{
\mathcal L_{\mathrm{pair}}
=
\tfrac12(\mathcal L_{\mathrm{det},A}
+\mathcal L_{\mathrm{det},L})
+
0.25\,\mathcal L_{\mathrm{vis}}.
}
\]

For missing companions,

\[
\boxed{
\Delta\zeta=0,\quad
\zeta'=\zeta,\quad
\mathcal L_{\mathrm{single}}=\mathcal L_{\mathrm{det}}.
}
\]

The native box and localization branches remain free of direct cross-view correction.

---

## 24. Suggested precise description for research writing

> **Visibility-State-Gated Cross-View Residual Assistance (VGRA)** is a candidate mechanism for exploiting complementary two-view semantic information without assuming spatial correspondence or compulsory cross-view agreement. It estimates a supervised four-state visibility distribution for each class—absent in both views, present only in the first view, present only in the second view, or present in both—and converts this distribution into target-specific, stop-gradient coefficients. A low-rank class-conditioned compatibility function compares companion-view global semantics with the target view's local features. The resulting compatibility is multiplied by the visibility coefficient and a zero-initialized, bounded, learned signed strength to form an additive classification-logit residual. Bounding and local anchoring restrict direct cross-view influence; localization predictions are not transferred across views, and a missing companion restores the ordinary single-view path. The mechanism is a testable hypothesis, not an assertion of established performance gains or guaranteed clinically appropriate assistance.

### The safest concise novelty claim

> We investigate whether **explicit four-state projection-visibility supervision**, coupled with **target-specific, low-rank, bounded signed classification residuals**, can provide more selective cross-view assistance than conventional undifferentiated fusion.

Avoid unsupported “first ever,” “guaranteed recall improvement,” “clinically safe suppression,” and “always-positive assistance” language.

---

## 25. Final conceptual summary

The VGRA idea can be understood through four questions:

1. **What does each image view contain semantically?**  
   Build a compact multi-scale descriptor for each view.

2. **In which of the two views should each class be expected, according to the learned annotation-state distribution?**  
   Predict the probabilities of 00, 10, 01, and 11 using an explicit supervised objective.

3. **At this particular target-view location, is the companion's class-specific semantic information compatible with the target's local evidence?**  
   Compute low-rank, normalized, nonnegative compatibility.

4. **How much should the native classification logit be adjusted?**  
   Apply a bounded signed residual using the learned strength, view-specific visibility coefficient, and local compatibility—while leaving localization untouched.

The resulting research hypothesis is that **visibility-aware, asymmetric, local, bounded cross-view score modification** may use complementary views more intelligently than unconditional fusion. Its strength is the precise, falsifiable mathematical design; its limitations are the absence of guaranteed gains, the reliance on visibility-state supervision, and the intentionally **signed** nature of the learned correction.

---

*Research provenance note: This explanation follows the frozen D4-B2C2 VGRA candidate V1 mathematical specification, the explicit signed-\(\beta\) scientific disposition, and the mathematical contracts of the corresponding VGRA core. It is an explanatory research document, not a new or amended candidate definition.*
