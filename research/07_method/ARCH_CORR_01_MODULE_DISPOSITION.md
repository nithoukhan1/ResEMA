# ARCH-CORR-01 — Module Disposition

| Module | Historical/new | Technical disposition | Verification state | Training disposition |
|---|---|---|---|---|
| Native YOLO11 C3k2 | baseline | reference topology | frozen baseline | frozen reference |
| Historical C3k2_SC | historical | topology/semantics/transfer confounded | forensic audit complete | historical only |
| SCConv core | historical operator | retain | wrapped by verified TPSC candidates | use only through corrected wrapper |
| C3k2_TPSC Early | new | transfer-preserving candidate | remote + local PASS | pending closure attestation |
| C3k2_TPSCG4 | new | transfer-preserving candidate | remote + local PASS | pending closure attestation |
| DySample | historical operator | retain reference-consistent implementation | focused remote + local PASS | pending closure attestation |
| Historical ResEMA-V2 | historical | noncanonical EMA-inspired mechanism | forensic audit complete | retired from new design |
| CanonicalEMA | new | canonical EMA candidate | remote + local PASS | pending closure attestation |
| C3k2_TPEMA | new | transfer-preserving EMA wrapper | remote + local PASS | pending closure attestation |

## Advancement rule

A candidate may enter training only after:
1. focused tests pass;
2. official checkpoint transfer audit passes where applicable;
3. local exact-head reproduction passes;
4. parameter/state/output identity evidence is frozen;
5. repository closure candidate CI passes;
6. final closure attestation is recorded.

Combination experiments are not authorized until single-module candidates have been
independently evaluated.
