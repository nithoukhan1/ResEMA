# D4-D3 — Approved scientific dispositions (A1 and B1)

**Decision authority:** Explicit user/project-design-authority instruction issued 2026-10-08 (Asia/Singapore) during D4-D3 formal-acceptance preparation.
**Status:** A1 APPROVED; B1 CONDITIONALLY APPROVED. At preparation, this document is a **candidate documentation-only Git record**; its formal acceptance is conditional on subsequent independent remote-head verification.
**Pinned source:** `research/vgra-impl-01 @ aecaaadbc60c075c8a0965e5d80fd27650093196`.
**Canonical replay evidence archive SHA256:** `23558c8bf26e76e861b27c7d90aea279f7d5b4d2e30f2ac4896e8e088cad497f`.

## A1 — SIGNED-BETA APPROVED

- **Decision:** Retain the exact frozen D4-B2C2 candidate-V1 equation `beta_l = 2*tanh(rho_l)` with `rho_l` initialized to zero. Preserve `k_AP,c = stopgrad(q_11,c - q_01,c)` and `k_LAT,c = stopgrad(q_11,c - q_10,c)`.
- **Observed semantics:** With `m >= 0`, positive beta preserves the nominal shared-positive/exclusive-negative contribution; negative beta reverses that nominal direction. D3C1 verified this against the actual implementation. This polarity reversal is **explicitly acknowledged and approved**.
- **Permitted research description:** VGRA is a **bounded learned signed, visibility-conditioned classification-logit residual**, not an always-positive assistive mechanism or a guarantee of clinically appropriate suppression.
- **Source decision:** NO VGRA code, architecture, pre-registered candidate equation, initialization or frozen ablation criteria changes are authorized by A1.
- **Evidence and limitation:** D3C1 proves mathematical and gradient consistency, not diagnostic effectiveness, calibration, patient-level improvement, or AP/mAP gains. The interpretation applies to model logits, not anatomical causal effects.

## B1 — PRETRAINED-PATH NUMERICAL RISK CONDITIONALLY APPROVED

- **Confirmed hazard:** Fresh initialization on exactly zero image inputs produced nonfinite FP32 backward gradients in Early and/or VGRA diagnostic branches. D3A1 reported up to 1,104,002 NaN gradient elements in the tested case, with subsequent D3A2–D3A4 tracing the effect through backward operations including a native BatchNorm FP32 overflow boundary. Preserve these as **known open risk evidence**, not solved bugs.
- **Bounded safety evidence:** D3A5 ran eight checks with verified pretrained Early initialization and did not reproduce nonfinite gradients. D3B1 completed two in-memory synthetic CPU epochs, six SGD updates, six EMA updates, and finite all-zero/empty-annotation post-update probes.
- **Approved disposition:** Accept the **limited CPU synthetic pretrained-path evidence**, conditional on implementing the safeguards below *before any separately authorized real-data or GPU training*. Do not claim a general numerical stability guarantee or a numerical root-cause repair.

### Mandatory conditions for any future D4-F training authorization

1. Load the **specific verified pretrained Early checkpoint** `best.pt`, SHA256 `620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1`, from archive SHA256 `4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396`. Fail closed on mismatch.
2. Verify transfer of **all 537 canonical Early state tensors** into VGRA with unchanged values, matching the accepted D4-D2 protocol. Scratch-only initialization is **not** the approved initial MV-01 execution path.
3. Require finiteness checks before and during any future training: total and component losses, all computed gradients, parameters/buffers, and optimizer/EMA state after updates as technically applicable. Any NaN or infinity means **abort**, not skip, mask, retry silently, or silently change the method. The exact runtime instrumentation must be specified and independently tested in D4-E before D4-F.
4. Record the first failing operation or batch metadata and program state safely for diagnosis, honoring the data privacy and sealed-test restrictions. Reproduction of the edge or a guard failure blocks the run until independently reviewed.
5. No workaround, FP64 substitution, gradient clipping, altered BN behavior or model-source repair is implicitly approved; each would require a separately governed design and re-verification.
6. D4-E must bind the complete identical MV-00/MV-01 recipe and provenance, and **training itself requires separate authorization**. A1/B1 alone do not authorize any real dataset access or experiments.

## Scope and unambiguous no-go statuses

| Governance item | Status after these scientific decisions |
|---|---|
| A1 signed beta interpretation | `APPROVED_RETAIN_SIGNED_WITH_EXPLICIT_POLARITY_REVERSAL` |
| B1 numerical disposition | `CONDITIONALLY_APPROVED_PRETRAINED_PATH_ONLY_GUARDS_MANDATORY` |
| D4-D3 Git acceptance | `CONDITIONALLY_EFFECTIVE_ONLY_AFTER_REMOTE_HEAD_VERIFICATION` |
| Exact MV-00/MV-01 run recipe | `UNFROZEN_D4_E` |
| Source or final paper architecture freeze | `FALSE` |
| GPU/real-data training authorized | `FALSE` |
| B-TEST access | `NONE` |

**Evidence integrity note:** the replay manifest predates the user decisions and historically says A1/B1 were pending. Its JSON values and observations remain unchanged; the Git copy uses LF line endings required by `.gitattributes`, while the original evidence ZIP remains untouched. This document supersedes only the **decision status**, not the original observations.
