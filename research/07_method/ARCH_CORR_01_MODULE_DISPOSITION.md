# ARCH-CORR-01 — Module Disposition

| Module | Historical/new | Technical disposition | Verification state | Experiment disposition |
|---|---|---|---|---|
| Native YOLO11 C3k2 | baseline | reference topology | frozen baseline | reference |
| Historical C3k2_SC | historical | topology/semantics/transfer confounded | forensic audit complete | historical only |
| SCConv core | historical operator | retain | verified through corrected wrappers | use only through TPSC |
| C3k2_TPSC Early | new | transfer-preserving candidate | remote + local PASS | eligible for controlled authorization |
| C3k2_TPSCG4 | new | transfer-preserving candidate | remote + local PASS | eligible for controlled authorization |
| DySample | historical operator | retain reference-consistent implementation | remote + local PASS | eligible for controlled authorization |
| Historical ResEMA-V2 | historical | noncanonical EMA-inspired mechanism | forensic audit complete | retired |
| CanonicalEMA | new | canonical EMA candidate | remote + local PASS | use through TPEMA candidate |
| C3k2_TPEMA | new | transfer-preserving EMA wrapper | remote + local PASS | eligible for controlled authorization |

## Advancement rule

Technical verification is closed.

A corrected candidate still requires a separately governed experiment registration and
source authorization before GPU training.

Combination experiments remain forbidden until the corresponding single-module
validation experiments are complete and reviewed.
