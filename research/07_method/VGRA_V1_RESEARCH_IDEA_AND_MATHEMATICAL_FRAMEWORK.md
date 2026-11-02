# VGRA V1 — Research Idea, Working Principle, and Complete Mathematical Framework

**Working name:** **Visibility-State-Gated Cross-View Residual Assistance (VGRA)**  
**Purpose:** A self-contained scientific explanation of the VGRA research idea, theoretical motivation, mathematical equations, learning objectives, and important limitations.  
**Scope:** The **VGRA V1 method itself**. This document deliberately excludes dataset descriptions, code instructions, training procedures, experiment logs, hardware, deployment, and empirical-result claims.

> **Scientific status:** VGRA V1 is a defined, testable research mechanism—not yet a demonstrated improvement. Its **bounded residual strength is signed**. Consequently, the nominal “assist shared / suppress view-exclusive” interpretation applies only when that strength is positive. The exact signed V1 formulation and its polarity reversal are expressly retained in the project's approved scientific disposition.

---

## 1. The scientific problem VGRA addresses

Suppose two projections, called **A** and **L**, depict the same underlying subject. Each has independent local evidence, but each may also contain information relevant to the other.

Two facts can hold simultaneously:

1. **Complementary information exists.** A feature that is ambiguous in A may be more evident in L; useful companion-view context should not automatically be discarded.
2. **Disagreement can be legitimate.** A class may be present or visible in A but not L, or vice versa. Forcing both projections to produce the same decision may create false positives or suppress genuine view-specific findings.

A conventional **independent-view detector** does not use companion-view context. Conversely, simple **always-on fusion** does not adequately distinguish *shared evidence* from *legitimate view-exclusive evidence*.

The VGRA research question is:

> **Can a detector learn when companion-view semantics should influence a target projection, and then change only that target's classification evidence in a bounded, locally grounded, class-specific manner, without transferring localization geometry or forcing agreement?**

VGRA answers with three complementary quantities:

- **Visibility-state gate:** What does the model predict about the presence of each class across the two projections?
- **Local-to-companion compatibility:** Does this particular target-view location semantically match the class-conditioned summary of the companion?
- **Bounded learned strength:** How large, and in what direction, should the correction be?

The method multiplies these quantities into a small additive **classification-logit residual**.

### 1.1 Why this is not ordinary feature fusion

Ordinary fusion often combines view representations directly, perhaps by concatenation, attention, summation, or a shared feature map. VGRA instead keeps the **independent target detector** intact and asks whether the companion warrants an **additional correction** to the target's class logit.

The key distinction is:

**Cross-view information can influence class evidence, but it is not allowed to become a transplanted target-view bounding box.**

---

## 2. The complete idea at a glance

The information flow is:

~~~text
                  VIEW A                         VIEW L
                     |                              |
              Multiscale features             Multiscale features
                     |                              |
              Semantic summary h_A           Semantic summary h_L
                     +--------------+---------------+
                                    |
                        Ordered pair representation
                    [h_A, h_L, |h_A-h_L|, h_A * h_L]
                                    |
                        Per-class four-state predictor
                           q_c = [00,10,01,11]
                                    |
                       Detached visibility probabilities
                         /                       \
                 Gate for A                   Gate for L
                 q11 - q01                    q11 - q10
                    |                            |
        Local A vs companion-L          Local L vs companion-A
        class-conditioned semantics     class-conditioned semantics
                    |                            |
           Compatibility m_A           Compatibility m_L
                    |                            |
            Delta_A = beta*k_A*m_A      Delta_L = beta*k_L*m_L
                    |                            |
        A classification logits       L classification logits
        receive bounded residual      receive bounded residual

           Both views retain independent box/localization predictions.
            No valid companion -> residual = 0 (ordinary view path).
~~~

Visibility prediction is **trained explicitly**, not merely invented as an unconstrained fusion switch.

---

## 3. Notation

Let the two views be \(v\in\{A,L\}\), and let \(\bar v\) denote the companion of \(v\).

The independent, weight-sharing detection network supplies three multiscale representations:

\[
\mathcal F_v=\{F_v^\ell:\ell\in\mathcal S\},\qquad
\mathcal S=\{P3,P4,P5\},
\]

where

\[
F_v^\ell\in\mathbb R^{C_\ell\times H_\ell\times W_\ell}.
\]

Let:

| Symbol | Definition |
|---|---|
| \(c\in\{1,\ldots,C\}\) | Class |
| \(\ell\) | Feature scale |
| \(i\) | Location in the target feature map |
| \(F_{v,i}^\ell\) | Target-view local feature |
| \(h_v^\ell,h_v\) | Scale-specific and concatenated global view descriptors |
| \(R_{c,s}\) | Predicted logit for class \(c\) and visibility state \(s\) |
| \(q_{c,s}\) | Probability of visibility state \(s\) |
| \(k_{v,c}\) | View- and class-specific visibility coefficient |
| \(m_{v,i,c}^\ell\) | Local target/companion compatibility |
| \(\rho_\ell\) | Trainable, initially zero residual parameter |
| \(\beta_\ell\) | Bounded signed residual strength |
| \(\zeta_{v,i,c}^\ell\) | Native, pre-activation classification logit |
| \(\Delta_{v,i,c}^\ell\) | VGRA correction |

The mathematical framework applies class-wise and is not predicated on identifying the *same individual object instance* across projections.

---

## 4. Stage I — Global semantic descriptors of both views

### 4.1 Global average pooling

For each view and feature scale, VGRA summarizes the spatial feature map:

\[
\operatorname{GAP}(F_v^\ell)
=\frac{1}{H_\ell W_\ell}\sum_{i=1}^{H_\ell W_\ell}F_{v,i}^\ell.
\]

This is a \(C_\ell\)-dimensional semantic summary.

### 4.2 Compact descriptor projection

A learned transformation produces a small vector for each scale:

\[
\boxed{
h_v^\ell
=\operatorname{SiLU}\!\left(
\operatorname{LN}\!\left(
W_d^\ell\operatorname{GAP}(F_v^\ell)+b_d^\ell
\right)\right).
}
\]

Here:

- \(W_d^\ell\in\mathbb R^{r_d\times C_\ell}\);
- \(b_d^\ell\in\mathbb R^{r_d}\);
- \(\operatorname{LN}\) denotes layer normalization;
- \(\operatorname{SiLU}(x)=x\sigma(x)\);
- the V1 descriptor rank is **\(r_d=32\)**.

Concatenating all three scales gives

\[
\boxed{
h_v=h_v^{P3}\oplus h_v^{P4}\oplus h_v^{P5}
\in\mathbb R^{96},
}
\]

where \(\oplus\) denotes concatenation.

**Interpretation.** \(h_A\) and \(h_L\) summarize the semantic content of each projection. They do **not** identify corresponding pixels, align coordinates, or transfer bounding boxes. This distinction matters because different projections need not share a simple spatial mapping.

---

## 5. Stage II — Predict per-class four-state cross-view visibility

### 5.1 Why exactly four states?

A binary “agreement” indicator cannot distinguish between three distinct forms of non-agreement. VGRA explicitly models:

| State | Index | Meaning |
|---|---:|---|
| \(00\) | 0 | Class absent from both projections |
| \(10\) | 1 | Class present only in A |
| \(01\) | 2 | Class present only in L |
| \(11\) | 3 | Class present in both |

These are **class-level projection-presence states**. The same pair can have state \(11\) for one class and \(01\) for a different class.

### 5.2 Ordered pair representation

VGRA combines both view descriptors:

\[
\boxed{
u=h_A\oplus h_L\oplus|h_A-h_L|\oplus(h_A\odot h_L).
}
\]

The four terms carry different kinds of information:

1. \(h_A\): semantic evidence in A.
2. \(h_L\): semantic evidence in L.
3. \(|h_A-h_L|\): magnitude of semantic difference.
4. \(h_A\odot h_L\): elementwise semantic co-activation.

Because each \(h_v\in\mathbb R^{96}\), the pair vector has dimension **384**.

**Order is intentional:** A-only and L-only are different labels; the predictor is not required to be invariant to swapping A and L without also swapping their state semantics.

### 5.3 The four-state predictor

A small network produces class-specific state logits:

\[
p=\operatorname{SiLU}(\operatorname{LN}(W_1u+b_1)),
\qquad p\in\mathbb R^{128},
\]

\[
R=\operatorname{reshape}(W_2p+b_2,[C,4]).
\]

Apply a softmax separately to each class:

\[
\boxed{
q_{c,s}=\frac{\exp(R_{c,s})}{\sum_{t=0}^{3}\exp(R_{c,t})}
}
\quad\text{and}\quad
\sum_{s=0}^{3}q_{c,s}=1.
\]

Therefore,

\[
q_c=(q_{c,00},q_{c,10},q_{c,01},q_{c,11}).
\]

**Interpretation:** \(q_c\) is the learned distribution over the four projection-visibility states for class \(c\). It is not a bounding-box confidence and is not guaranteed to be statistically calibrated.

---

## 6. Stage III — Give four-state visibility explicit supervision

### 6.1 Construct the target

For a class \(c\), let \(a_c\) and \(l_c\) be binary indicators of whether at least one labeled instance of that class occurs in A and L:

\[
a_c=\mathbf 1\{\text{class }c\text{ present in A}\},\qquad
l_c=\mathbf 1\{\text{class }c\text{ present in L}\}.
\]

Then the target state index is

\[
\boxed{y_c=a_c+2l_c.}
\]

It follows that \(00\to0\), \(10\to1\), \(01\to2\), \(11\to3\).

Importantly, “absent” here is a **class-label presence convention**. It does not independently establish anatomical absence or prove that an unannotated finding cannot exist.

### 6.2 State-frequency weighting

The four states may be very unequally frequent. For class \(c\), let \(n_{c,s}\) be the number of labeled pairs belonging to state \(s\). VGRA V1 defines

\[
\widetilde w_{c,s}=(n_{c,s}+1)^{-1/2}
\]

and normalizes **within each class**:

\[
\boxed{
w_{c,s}=\frac{4\widetilde w_{c,s}}
{\sum_{t=0}^{3}\widetilde w_{c,t}}.
}
\]

This ensures

\[
\frac14\sum_{s=0}^{3}w_{c,s}=1.
\]

The \(+1\) avoids zero denominators; the inverse square root upweights rarer states without the aggressiveness of full inverse frequency; and mean-one normalization stabilizes the scale.

### 6.3 Weighted visibility cross-entropy

For \(P\) paired observations and \(C\) classes,

\[
\boxed{
\mathcal L_{\mathrm{vis}}
=-\frac1{PC}\sum_{p=1}^{P}\sum_{c=1}^{C}
w_{c,y_{p,c}}\log q_{p,c,y_{p,c}}.
}
\]

VGRA V1 fixes the auxiliary coefficient at

\[
\boxed{\lambda_{\mathrm{vis}}=0.25.}
\]

For any state logit \(R_{p,c,s}\), the corresponding supervised gradient is

\[
\frac{\partial\mathcal L_{\mathrm{vis}}}{\partial R_{p,c,s}}
=\frac{w_{c,y_{p,c}}}{PC}
\left(q_{p,c,s}-\mathbf1\{s=y_{p,c}\}\right).
\]

**Why this objective matters:** VGRA does not simply hope that a generic gating network discovers projection-exclusive cases. It supplies an explicit semantic meaning and supervision for all four states.

---

## 7. Stage IV — Convert visibility probabilities into target-specific gates

### 7.1 The two gate equations

VGRA constructs **different coefficients for the two target views**:

\[
\boxed{
k_{A,c}=\operatorname{stopgrad}(q_{c,11}-q_{c,01})
}
\]

and

\[
\boxed{
k_{L,c}=\operatorname{stopgrad}(q_{c,11}-q_{c,10}).
}
\]

Each is a difference of probabilities. Consequently,

\[
-1\leq k_{A,c}\leq1,\qquad
-1\leq k_{L,c}\leq1.
\]

The gate does not hard-code a final class label. It is a **soft contextual coefficient** used in a subsequent residual calculation.

### 7.2 Interpreting the four states

In the idealized case of a perfectly certain, one-hot state prediction:

| State | \(k_{A,c}\) | \(k_{L,c}\) | *Nominal interpretation when residual strength is positive* |
|---|---:|---:|---|
| \(00\) | 0 | 0 | Both neutral |
| \(10\) | 0 | −1 | A neutral; negative correction permitted in L |
| \(01\) | −1 | 0 | Negative correction permitted in A; L neutral |
| \(11\) | +1 | +1 | Positive correction permitted in both |

This asymmetry is central. When a class is supported only in L, VGRA should not force A to agree positively simply because its companion supports the class.

The state \(00\) is **deliberately neutral** in the V1 gate. A “not visible in either” prediction is not used as an automatic negative-suppression signal.

**Caution:** The table concerns the sign of \(k\), not necessarily the final residual. The learned residual strength can be negative and reverse the net effect (Section 9).

### 7.3 Why the probabilities are detached

The operator \(\operatorname{stopgrad}\) blocks the path

\[
\mathcal L_{\mathrm{det}}\longrightarrow k\longrightarrow q.
\]

Through this gate path,

\[
\frac{\partial k_{v,c}}{\partial q_{c,s}}=0
\quad\text{during detection-loss backpropagation}.
\]

This protects the **meaning of the four-state predictor** from direct pressure to change its probabilities merely to obtain a more convenient detection residual.

The visibility predictor still receives gradients from \(\mathcal L_{\mathrm{vis}}\). Also, shared view-descriptor features can receive gradients through other, non-detached paths, especially compatibility. Therefore the correct claim is **“the probability-to-gate gradient is blocked,”** not “the two objectives share no representation learning.”

### 7.4 Soft-gate example

Consider the illustrative distribution

\[
q_c=(q_{00},q_{10},q_{01},q_{11})
=(0.05,0.10,0.65,0.20).
\]

Then

\[
k_{A,c}=0.20-0.65=-0.45,\qquad
k_{L,c}=0.20-0.10=+0.10.
\]

The *same* class in the *same* pair can thus receive a negative gate for A and a weak positive gate for L. Their final logit modifications remain conditional on local compatibility and learned residual strength.

---

## 8. Stage V — Low-rank class-conditioned cross-view compatibility

### 8.1 Why compatibility is needed

A companion projection might contain relevant study-level evidence, but that alone does not justify increasing a class score at every spatial location in the target. VGRA therefore asks a second question:

> **Does this specific local target feature align semantically with a class-conditioned global summary of the companion?**

The V1 cross-view embedding rank is **\(r_x=16\)**.

### 8.2 Target-local embedding

For target feature \(F_{v,i}^{\ell}\),

\[
\boxed{
z_{v,i}^{\ell}
=\operatorname{norm}(A_{\ell}F_{v,i}^{\ell}).
}
\]

Here \(A_{\ell}\) is a learned linear projection into \(\mathbb R^{r_x}\); \(\operatorname{norm}\) denotes numerically stabilized L2 normalization.

### 8.3 Companion-semantic embedding

Using the **global** descriptor of the opposite view,

\[
\boxed{
g_{\bar v}^{\ell}=\operatorname{norm}(B_{\ell}h_{\bar v}),
}
\]

where \(B_{\ell}\) maps the 96-dimensional companion descriptor into the same low-rank space.

Importantly, this is **not** a location in the companion view and does not assume spatial alignment.

### 8.4 Class-conditioned direction

Let \(e_c^{\ell}\in\mathbb R^{r_x}\) be a learned class embedding. Define

\[
\boxed{
t_{\bar v,c}^{\ell}
=\operatorname{norm}\left(
g_{\bar v}^{\ell}\odot\operatorname{norm}(e_c^{\ell})
\right).
}
\]

Elementwise multiplication makes the companion representation *class specific*. Different classes need not receive the same compatibility map.

### 8.5 Positive compatibility

Define

\[
\boxed{
m_{v,i,c}^{\ell}
=\max\left(0,\left\langle
z_{v,i}^{\ell},t_{\bar v,c}^{\ell}
\right\rangle\right).
}
\]

Both vectors are L2 normalized, so the inner product lies in \([-1,1]\), and

\[
\boxed{0\leq m_{v,i,c}^{\ell}\leq1.}
\]

A large \(m\) indicates strong learned semantic alignment between this **target location** and this **class-conditioned companion summary**. A zero value means no positive compatibility is supplied through this path.

**This is not a probabilistic clinical-confidence estimate.** It is learned feature compatibility.

### 8.6 Why low rank and target-local grounding?

Low rank constrains the cross-view interaction, while target-local grounding prevents the companion from adding an indiscriminate study-wide boost to every spatial prediction. The model needs both relevant companion semantics **and** a compatible target-local representation.

Neither feature matching nor this compatibility score constitutes cross-view instance matching or geometric registration.

---

## 9. Stage VI — The bounded, zero-initialized, **signed** residual

This is VGRA's defining mathematical step.

### 9.1 Learnable residual strength

For each scale \(\ell\), let \(\rho_{\ell}\) be a learned scalar initialized to zero:

\[
\boxed{\rho_{\ell}(0)=0.}
\]

The effective strength is

\[
\boxed{\beta_{\ell}=\beta_{\max}\tanh(\rho_{\ell}),\qquad
\beta_{\max}=2.0.}
\]

Therefore,

\[
|\beta_{\ell}|\leq2.
\]

### 9.2 Add only to target classification logits

Let \(\zeta_{v,i,c}^{\ell}\) denote the detector's original class logit. VGRA defines

\[
\boxed{
\Delta_{v,i,c}^{\ell}
=\beta_{\ell}\,k_{v,c}\,m_{v,i,c}^{\ell}
}
\]

and

\[
\boxed{
\zeta_{v,i,c}^{\ell\prime}
=\zeta_{v,i,c}^{\ell}
+\beta_{\ell}k_{v,c}m_{v,i,c}^{\ell}.
}
\]

The three multipliers have different roles:

- **\(\beta_{\ell}\):** learned scale **and sign** of correction;
- **\(k_{v,c}\):** class-specific visibility-state guidance for the chosen target view;
- **\(m_{v,i,c}^{\ell}\):** spatially local compatibility with companion semantics.

### 9.3 Mathematical residual bound

Because

\[
|\beta_{\ell}|\leq2,\quad |k_{v,c}|\leq1,\quad
0\leq m_{v,i,c}^{\ell}\leq1,
\]

it follows that

\[
\boxed{|\Delta_{v,i,c}^{\ell}|\leq2.}
\]

The bound limits the size of a **logit correction**. It does not guarantee a bounded change of a particular evaluation metric or that no prediction changes near a decision threshold.

### 9.4 Exact zero-residual identity at initialization

At \(\rho_{\ell}=0\),

\[
\beta_{\ell}=2\tanh(0)=0.
\]

Thus,

\[
\boxed{
\Delta_{v,i,c}^{\ell}=0,\qquad
\zeta_{v,i,c}^{\ell\prime}=\zeta_{v,i,c}^{\ell}.
}
\]

This establishes the scientific rationale for **zero-initialized assistance**: the starting model's class-logit behavior is unchanged, and the model must *learn* whether and how strongly companion evidence should enter.

### 9.5 Critical V1 fact: the coefficient is **signed**

The \(\tanh\) strength is not constrained to be positive:

\[
\rho_{\ell}<0\implies\beta_{\ell}<0.
\]

Therefore:

| \(\beta_{\ell}\) | Positive gate \(k>0\) | Negative gate \(k<0\) |
|---|---|---|
| Positive | Raises the logit where \(m>0\) | Lowers the logit where \(m>0\) |
| Negative | **Lowers** the logit where \(m>0\) | **Raises** the logit where \(m>0\) |
| Zero | No change | No change |

This means the usual prose **“shared states assist and exclusive states suppress” is *not universally true* for the frozen V1 equations.**

The project's approved interpretation is:

> **VGRA V1 is a bounded, learned, signed, visibility-conditioned classification-logit residual—not a guaranteed positive assistant.**

The ability of a negative strength to reverse nominal gate polarity was explicitly accepted for this candidate. It must not be omitted or silently “fixed” in a scientific explanation.

### 9.6 Worked residual example

Using the gate from Section 7,

\[
k_{A,c}=-0.45
\]

and suppose

\[
m_{A,i,c}^{\ell}=0.80.
\]

If \(\beta_{\ell}=+1.20\),

\[
\Delta_{A,i,c}^{\ell}
=(1.20)(-0.45)(0.80)=-0.432.
\]

For an original logit \(\zeta=0.70\),

\[
\zeta'=0.70-0.432=0.268.
\]

But if \(\beta_{\ell}=-1.20\),

\[
\Delta=+0.432,\qquad\zeta'=1.132.
\]

**Identical visibility and compatibility can produce opposite logit movements if the learned strength changes sign.**

### 9.7 Logit change versus probability change

For sigmoid-based class scoring,

\[
p=\sigma(\zeta),\qquad p'=\sigma(\zeta+\Delta).
\]

The residual changes the *log-odds* exactly by \(\Delta\):

\[
\boxed{
\log\frac{p'}{1-p'}-\log\frac{p}{1-p}=\Delta.
}
\]

Thus the odds multiply by \(e^{\Delta}\), rather than the class probability changing by a fixed amount. Since \(|\Delta|\leq2\), the odds multiplier is between \(e^{-2}\) and \(e^2\).

---

## 10. The localization firewall — Why boxes remain view specific

VGRA **does not** transfer bounding boxes, regress a geometric correspondence between the two projections, or use companion coordinates to replace target coordinates.

If the independent detector yields a class logit and a geometric prediction,

\[
(\zeta_{v,i,c}^{\ell},b_{v,i}^{\ell}),
\]

VGRA changes this to

\[
\boxed{
(\zeta_{v,i,c}^{\ell}+\Delta_{v,i,c}^{\ell},\;
b_{v,i}^{\ell}).
}
\]

The box prediction \(b_{v,i}^{\ell}\) itself remains unchanged by VGRA's residual. The target view still owns its localization and geometric decoding.

**Important distinction:** A logit change can affect downstream ranking, thresholding, or selection of *existing* candidate boxes. That is different from altering the predicted box geometry. VGRA does the former only indirectly, never the latter directly.

This separation is scientifically useful because **semantic support can be cross-view even when spatial correspondence is not**.

---

## 11. Detection and visibility learning objectives

### 11.1 Conceptual objective for a paired observation

Each projection retains ordinary independent detection supervision. Conceptually,

\[
\boxed{
\mathcal L_{\mathrm{pair}}
=\frac12(\mathcal L_{\mathrm{det},A}
+\mathcal L_{\mathrm{det},L})
+\lambda_{\mathrm{vis}}\mathcal L_{\mathrm{vis}}.
}
\]

With \(\lambda_{\mathrm{vis}}=0.25\),

\[
\mathcal L_{\mathrm{pair}}
=\tfrac12(\mathcal L_{\mathrm{det},A}
+\mathcal L_{\mathrm{det},L})
+0.25\mathcal L_{\mathrm{vis}}.
\]

The first term learns local class/box predictions. The second term teaches per-class four-state visibility.

### 11.2 Mixed paired and independent observations

Let a collection contain:

- \(P\) valid pairs (accounting for \(2P\) images);
- \(S\) independent single-view images;
- \(N=2P+S\) total images.

If \(\overline{\mathcal L}_{\mathrm{det}}\) denotes the native detection objective expressed on its per-image reporting scale, VGRA V1's normalized mixed objective is

\[
\boxed{
\overline{\mathcal L}_{\mathrm{total}}
=\overline{\mathcal L}_{\mathrm{det}}
+0.25\,\frac{2P}{N}\,\mathcal L_{\mathrm{vis}}.
}
\]

Its equivalent batch-scaled form is

\[
\boxed{
\mathcal L_{\mathrm{total,scaled}}
=N\,\overline{\mathcal L}_{\mathrm{det}}
+0.25(2P)\mathcal L_{\mathrm{vis}}.
}
\]

These expressions describe the **auxiliary weighting contract**; they do not replace or redefine the base detector's internal geometric/classification normalization.

When \(P=0\), the visibility term vanishes exactly. If every image is paired, \(2P/N=1\), giving the ordinary auxiliary weight \(0.25\).

### 11.3 What is *not* being learned

The four-state objective does **not** train cross-view instance matching. It does not predict a shared lesion box in a canonical coordinate system. It trains a **class-level semantic visibility distribution** that governs a separate, locally compatible classification correction.

---

## 12. Gradient behavior and trainability

### 12.1 Why zero initialization does not freeze the residual

From

\[
\Delta=\beta_{\max}\tanh(\rho_{\ell})km,
\]

the derivative with respect to the learned strength parameter is

\[
\boxed{
\frac{\partial\Delta}{\partial\rho_{\ell}}
=\beta_{\max}
\left(1-\tanh^2(\rho_{\ell})\right)km.
}
\]

At the initial \(\rho_{\ell}=0\),

\[
\boxed{
\left.\frac{\partial\Delta}{\partial\rho_{\ell}}\right|_{\rho_{\ell}=0}
=2km.
}
\]

So the residual is **zero in value**, but its gradient with respect to \(\rho_{\ell}\) need not be zero. The detection objective can move the signed strength away from zero.

For example, the chain-rule contribution is

\[
\frac{\partial\mathcal L_{\mathrm{det}}}{\partial\rho_{\ell}}
=\sum_{v,i,c}
\frac{\partial\mathcal L_{\mathrm{det}}}
{\partial\zeta_{v,i,c}^{\ell\prime}}
\,2\,\operatorname{sech}^{2}(\rho_{\ell})
\,k_{v,c}m_{v,i,c}^{\ell}.
\]

There is no mathematical rule requiring this gradient to move \(\rho_\ell\) in the positive direction.

### 12.2 The compatibility-path gradient at initialization

Since

\[
\frac{\partial\Delta}{\partial m}=\beta_{\ell}k,
\]

at \(\rho_{\ell}=0\) one has

\[
\frac{\partial\Delta}{\partial m}=0.
\]

Thus, at the *exact zero-strength starting point*, the residual's compatibility path receives no detection gradient **through that residual**. After the strength becomes nonzero, compatibility can receive detection gradients.

The visibility-prediction branch can learn from \(\mathcal L_{\mathrm{vis}}\) from the beginning. Shared descriptors may also receive gradients via compatibility after the residual is activated.

This explains how VGRA can **begin exactly as the independent detector**, then gradually learn additional, bounded cross-view effects.

### 12.3 Semantic separation is partial and intentional

The **gate** is detached from detection gradients:

\[
\frac{\partial k}{\partial q}=0\quad
\text{along the detection-loss gate path}.
\]

The rest of the network is not necessarily isolated. In particular, descriptor representations can be involved in both visibility supervision and learned companion compatibility.

Therefore it would be inaccurate to claim that visibility and detection learning are completely independent. VGRA's precise protection is narrower: **detection loss is not permitted to directly relabel the visibility probability distribution through the gate connection.**

---

## 13. Exact single-view fallback

When no valid companion exists, VGRA does not invent one and does not assign a synthetic four-state target.

Instead,

\[
\boxed{
\Delta_{v,i,c}^{\ell}=0
\quad\Longrightarrow\quad
\zeta_{v,i,c}^{\ell\prime}
=\zeta_{v,i,c}^{\ell}.
}
\]

Only the ordinary independent-view detection objective applies.

This is an essential design principle: VGRA is **residual assistance to a viable independent detector**, not a fusion model that loses its identity when only one view is available.

---

## 14. Three different forms of selectivity

The method makes cross-view correction selective on three axes:

### 14.1 Class selectivity

Each class has its own four-state distribution \(q_c\), gate \(k_{v,c}\), and learned class embedding \(e_c^\ell\). A companion may be useful for class \(c_1\) but not for \(c_2\).

### 14.2 Target-view selectivity

For the **same** class, the A and L gates are distinct:

\[
k_{A,c}=q_{11}-q_{01},\qquad
k_{L,c}=q_{11}-q_{10}.
\]

A finding believed exclusive to one view does not automatically induce the same correction in the opposite view.

### 14.3 Spatial selectivity

Within one target view, different locations have different compatibility values \(m_{v,i,c}^{\ell}\). Companion information therefore need not change every local class logit.

The full effect is the product of **learned scale/sign**, **view/class visibility**, and **target-local compatibility**.

---

## 15. What the method actually hypothesizes

VGRA's scientific hypothesis is **not** simply that “two views are better than one.” That general idea is already well established.

The more precise hypothesis is:

> **Explicit supervision of per-class, four-state projection visibility can make cross-view semantic influence more selective than unconditional fusion, when that influence is limited to a zero-initialized, bounded, class-conditioned, target-local classification-logit residual.**

The hypothesis has several logically separable components:

1. **Four-state supervision** adds information that a binary shared/not-shared label lacks.
2. **Target-specific gating** allows asymmetric use of one companion between projections.
3. **Local compatibility** prevents a uniform, study-wide class boost.
4. **Residual boundedness** restricts the scale of cross-view alteration.
5. **Localization independence** avoids unsupported geometric assumptions.
6. **Fallback identity** preserves ordinary single-view behavior.

A proper scientific evaluation must determine whether these properties jointly deliver useful evidence—not presume it from the equations.

---

## 16. Important theoretical limitations

| Design property | Benefit sought | Limitation |
|---|---|---|
| Four states | Distinguish shared, A-only, L-only, neither | Label presence is not ground-truth biological visibility |
| Visibility predictor | Explicit state supervision | Softmax outputs may not be calibrated |
| Detached gates | Protect state semantics from direct detection-loss pressure | Shared descriptor features still learn from multiple pathways |
| Low-rank compatibility | Controlled class-dependent cross-view influence | Learned semantic alignment is not causal or clinical certainty |
| Zero initialization | Original class logits initially unchanged | Does not guarantee stable or useful subsequent learning |
| Signed \(\beta\) | Learns scale and polarity | Can invert nominal assist/suppress meaning |
| Logit bound | No unbounded residual amplitude | Can still change decisions near thresholds |
| No box transfer | Avoid geometric alignment assumptions | Cannot directly fix a mislocalized box |
| Single-view fallback | Companion not required | No cross-view benefit without a companion |

### A particularly important limitation: signed-beta polarity reversal

Because \(\beta_{\ell}=2\tanh(\rho_{\ell})\), **the sign of the final residual cannot be read from the visibility state alone**. A claim such as “\(11\) always boosts detection” is mathematically false for the approved V1 form.

The scientifically correct phrasing is **“the four-state gate conditions a learned signed residual, with positive and negative influence depending jointly on gate, compatibility, and learned strength.”**

---

## 17. What would constitute novelty—and what would not

VGRA does **not** claim to invent:

- multi-view detection;
- paired-image feature fusion;
- adaptive gating as a broad concept;
- auxiliary view classifiers;
- view-specific labels in general;
- multi-view distillation.

Its proposed distinctiveness lies in the **specific combination** of:

1. supervised **per-class four-state projection visibility** \(00,10,01,11\);
2. **directional A/L gate asymmetry** derived from those probabilities;
3. detection-gradient **stop-gradient through the probabilities used for gating**;
4. **class-conditioned companion semantics** compared with target-local features;
5. a **bounded, zero-initialized, signed classification-logit correction**;
6. **independent localization geometry**; and
7. **exact missing-companion fallback**.

A cautious methods description would be:

> “We investigate a projection-visibility-supervised, target-specific residual assistance mechanism that modulates independent-view classification evidence using class-conditioned companion semantics, without enforcing cross-view localization or mandatory projection agreement.”

This is a **candidate scientific contribution**, not proof that no similar prior method exists or that the mechanism improves detection.

---

## 18. All essential V1 mathematics in one place

**View descriptor**

\[
h_v^\ell=\operatorname{SiLU}(
\operatorname{LN}(W_d^\ell\operatorname{GAP}(F_v^\ell)+b_d^\ell)),
\qquad h_v=\bigoplus_{\ell}h_v^\ell.
\]

**Ordered pair context**

\[
u=h_A\oplus h_L\oplus|h_A-h_L|\oplus(h_A\odot h_L).
\]

**Four-state visibility**

\[
R=\operatorname{reshape}
\bigl(W_2\operatorname{SiLU}(\operatorname{LN}(W_1u+b_1))+b_2,[C,4]\bigr),
\qquad q_c=\operatorname{softmax}(R_c).
\]

**State target**

\[
y_c=a_c+2l_c\in\{0,1,2,3\}.
\]

**State-frequency weight**

\[
w_{c,s}=
\frac{4(n_{c,s}+1)^{-1/2}}
{\sum_{t=0}^{3}(n_{c,t}+1)^{-1/2}}.
\]

**Visibility supervision**

\[
\mathcal L_{\mathrm{vis}}
=-\frac1{PC}\sum_{p,c}
w_{c,y_{p,c}}\log q_{p,c,y_{p,c}}.
\]

**Directional gates**

\[
\boxed{
k_{A,c}=\operatorname{stopgrad}(q_{c,11}-q_{c,01}),\qquad
k_{L,c}=\operatorname{stopgrad}(q_{c,11}-q_{c,10}).
}
\]

**Low-rank compatibility**

\[
z_{v,i}^\ell=\operatorname{norm}(A_\ell F_{v,i}^\ell),
\quad
g_{\bar v}^\ell=\operatorname{norm}(B_\ell h_{\bar v}),
\]

\[
t_{\bar v,c}^\ell=
\operatorname{norm}\left(g_{\bar v}^\ell\odot\operatorname{norm}(e_c^\ell)\right),
\qquad
m_{v,i,c}^\ell=
\max(0,\langle z_{v,i}^\ell,t_{\bar v,c}^\ell\rangle).
\]

**Signed strength and residual**

\[
\boxed{
\beta_\ell=2\tanh(\rho_\ell),\quad\rho_\ell(0)=0,
\qquad
\zeta_{v,i,c}^{\ell\prime}
=\zeta_{v,i,c}^{\ell}
+\beta_\ell k_{v,c}m_{v,i,c}^\ell.
}
\]

**Bound**

\[
\boxed{
|\zeta_{v,i,c}^{\ell\prime}-\zeta_{v,i,c}^{\ell}|\leq2.
}
\]

**Conceptual pair objective**

\[
\mathcal L_{\mathrm{pair}}
=\tfrac12(\mathcal L_{\mathrm{det},A}+\mathcal L_{\mathrm{det},L})
+0.25\,\mathcal L_{\mathrm{vis}}.
\]

**Normalized mixed objective** for \(P\) pairs and \(N\) images

\[
\boxed{
\overline{\mathcal L}_{\mathrm{total}}
=\overline{\mathcal L}_{\mathrm{det}}
+0.25\,\frac{2P}{N}\mathcal L_{\mathrm{vis}}.
}
\]

**Fallback**

\[
\boxed{
\text{No companion}\implies\Delta=0\implies\zeta'=\zeta.
}
\]

### V1 mathematical constants

| Constant | Value |
|---|---:|
| Visibility states | 4 |
| Feature levels | 3 |
| Per-level view-descriptor rank \(r_d\) | 32 |
| Concatenated view descriptor | 96 |
| Ordered pair vector | 384 |
| Visibility predictor hidden dimension | 128 |
| Low-rank cross-view dimension \(r_x\) | 16 |
| Residual bound parameter \(\beta_{\max}\) | 2.0 |
| Initial \(\rho_\ell\) | 0 |
| Auxiliary visibility-loss coefficient | 0.25 |
| Gradient through probability-to-gate path | Stopped |
| Companion-driven box geometry | None |

---

## 19. One-paragraph research-paper description

**Visibility-State-Gated Cross-View Residual Assistance (VGRA)** is a class-conditioned multi-view learning mechanism that selectively incorporates companion-view semantics into an otherwise independent target-view detector. Semantic descriptors from the two views are combined to predict a supervised four-state distribution for each class, distinguishing absence in both views, view-exclusive presence, and shared presence. These probabilities form detached, target-specific visibility coefficients. At each prediction location, low-rank compatibility measures the alignment between target-local evidence and class-conditioned companion semantics. A zero-initialized, bounded **signed** residual, determined jointly by this compatibility, the target-specific visibility coefficient, and a learnable scale-dependent strength, is added exclusively to the target's classification logits. Independent box geometry is preserved, while a weighted auxiliary cross-entropy objective trains the visibility predictor alongside ordinary detection supervision. Without a valid companion, the residual is exactly zero and the model reverts to independent-view behavior. The central hypothesis is that explicit projection-visibility reasoning can make cross-view semantic influence more selective than unconditional fusion; its practical benefit remains an empirical question.

---

## Source basis and scientific interpretation

This document reconstructs the project's **frozen VGRA V1 specification** and its **later approved signed-beta clarification**. It is not a new proposal for changes to V1 and makes no claim of proven performance improvement.

Authoritative scientific records:

1. [VGRA mathematical specification](https://github.com/nithoukhan1/ResEMA/blob/f440d329d4544ec08ad63c1972bcf9a8b8a7eed8/research/07_method/D4_B2C2_VGRA_MATHEMATICAL_SPEC.md)
2. [VGRA novelty boundary](https://github.com/nithoukhan1/ResEMA/blob/f440d329d4544ec08ad63c1972bcf9a8b8a7eed8/research/07_method/D4_B2C2_VGRA_NOVELTY_BOUNDARY.md)
3. [Approved signed-beta scientific interpretation](https://github.com/nithoukhan1/ResEMA/blob/f440d329d4544ec08ad63c1972bcf9a8b8a7eed8/research/07_method/D4_D3_SCIENTIFIC_DISPOSITIONS_APPROVED.md)
4. [Visibility objective normalization contract](https://github.com/nithoukhan1/ResEMA/blob/f440d329d4544ec08ad63c1972bcf9a8b8a7eed8/research/07_method/D4_C5B_RUNTIME_NATIVE_LOSS_CONTRACT.md)

**End of VGRA V1 theoretical and mathematical framework.**
