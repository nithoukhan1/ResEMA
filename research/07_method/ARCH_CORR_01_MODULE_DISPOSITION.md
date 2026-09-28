# ARCH-CORR-01 — Module Disposition

| Module | Historical/new | Technical disposition | Training disposition |
|---|---|---|---|
| Native YOLO11 C3k2 | baseline | reference topology | frozen reference |
| Historical C3k2_SC | historical | topology/semantics/transfer confounded | historical only |
| SCConv core | historical operator | retain | use only through corrected wrapper |
| C3k2_TPSC Early | new | transfer-preserving candidate | not yet authorized |
| C3k2_TPSCG4 | new | transfer-preserving candidate | not yet authorized |
| DySample | historical operator | retain reference-consistent implementation | not yet authorized in new matrix |
| Historical ResEMA-V2 | historical | noncanonical EMA-inspired mechanism | retired from new design |
| CanonicalEMA | new | canonical EMA candidate | pending full verification closure |
| C3k2_TPEMA | new | transfer-preserving EMA wrapper | pending full verification closure |

## Advancement rule

A candidate may enter training only after:
1. focused tests pass;
2. official checkpoint transfer audit passes;
3. local exact-head reproduction passes;
4. parameter/state/output identity evidence is frozen;
5. branch closure CI passes.

Combination experiments are not authorized until single-module candidates have been
independently evaluated.
