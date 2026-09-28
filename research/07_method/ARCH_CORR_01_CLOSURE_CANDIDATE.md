# ARCH-CORR-01 Closure Candidate

## Status

`CLOSURE CANDIDATE — REMOTE CI REQUIRED — TRAINING NOT AUTHORIZED`

Verified implementation head:

`a30df2165cb24a0796086a32b07445ea04b5c7bc`

## Same-head verification already complete

Remote:
- Research Integrity run #33 / id `36430937501`: PASS;
- ARCH-CORR runtime audit run #5 / id `36430937722`: PASS.

Local:
- 16 focused tests PASS;
- locked `yolo11s.pt` SHA256:
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`;
- TPSC audit PASS;
- canonical EMA/TPEMA audit PASS;
- DySample focused verification PASS;
- worktree clean after evidence generation.

Local artifact hashes:
- TPSC:
  `f8aed12d60a3e60937bf5c5917bf61ce657f50eb45a6a239273c645e602f918c`;
- EMA:
  `d2002175791c9035af0894de3c5e32823a786db8b7e763628c2b4d274858258d`;
- master:
  `ac2d2076d7f936999349809b83141f63dd45a88ad4a8b3ae6ff45ef35bf54420`.

## Verified module contracts

TPSC Early:
- 9,570,093 parameters;
- +138,818 over baseline;
- 499/499 native state items preserved;
- 493 official checkpoint items transferable;
- 38 new adapter state items;
- zero unexpected native nontransfer;
- exact zero-gate output identity.

TPSC G4:
- 10,021,679 parameters;
- +590,404;
- 499/499 native state items preserved;
- 493 official checkpoint items transferable;
- 76 new adapter state items;
- zero unexpected native nontransfer;
- exact zero-gate output identity.

TPEMA Head:
- 9,435,423 parameters;
- +4,148;
- 499/499 native state items preserved;
- 493 official checkpoint items transferable;
- 28 new EMA state items;
- zero unexpected native nontransfer;
- exact zero-gate output identity.

DySample:
- retained implementation;
- 9,455,915 expected YOLO11s-9C parameters;
- +24,640 over baseline;
- focused LP/PL/gradient/configuration checks passed.

## Firewall

- training_started = false;
- dataset_access = NONE;
- test_access = NONE;
- Split-B test remains sealed.

## Final closure rule

This candidate does not authorize training.

It must first receive both Research Integrity PASS and dedicated ARCH-CORR runtime
audit PASS. Those run IDs will then be written into a final closure attestation.
