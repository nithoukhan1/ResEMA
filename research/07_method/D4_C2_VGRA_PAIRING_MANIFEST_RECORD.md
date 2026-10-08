# D4-C2 VGRA Pairing Manifest / Visibility Schema

## Status
`COMPLETE / DETERMINISTIC / METADATA-ONLY`

Branch:
`research/vgra-impl-01`

Parent:
`5492a60d58ac96101af57f7ed8811b8e606faa00`

## Input binding
- TRAIN CSV SHA256: `6b99f5d08e62eb2c4c3cf5c3481a1e2ca37bfb983f79d462b6ab3fed254fdf0e`
- VAL CSV SHA256: `f33c14ae479cffe0ee769549a1e3d53d5c5e198734c6bebb86258afad7bd074e`

No Split-B test file was opened.
No raw image was opened.
No YOLO label file was opened.

## Pair definition
A structural pair is:
`patient_id + study_number + laterality`

and must contain exactly:
- one projection 1 (AP/PA);
- one projection 2 (lateral);
- no additional member in that side-specific study group.

Non-exact AP/LAT-capable groups are deliberately not force-paired.

## TRAIN
- images: 14227
- patients: 4264
- side-specific study groups: 7590
- AP/LAT-capable groups: 6502
- exact AP/LAT pairs: 6496
- structurally paired images: 12992
- structural singles: 1235
- ambiguous non-exact AP/LAT groups: 6

## VAL
- images: 3050
- patients: 914
- side-specific study groups: 1622
- exact AP/LAT pairs: 1402
- structurally paired images: 2804
- structural singles: 246
- known unreadable members: 1
- readable companions forced to single fallback: 1
- operational paired images: 2802

## Visibility target policy
D4-C2 freezes the target schema only.

For each detection class:
- 00 = absent in AP and LAT;
- 10 = AP only;
- 01 = LAT only;
- 11 = present in both.

Presence is defined from YOLO object labels, not metadata columns.

Actual B-TRAIN state targets and TRAIN-only state-frequency weights are intentionally
deferred to D4-C5 when criterion integration is implemented.

VAL visibility targets may be used only for validation/diagnostics.
B-TEST visibility labels remain forbidden before the sealed-test phase.

## Outputs
- `research/04_data/manifests/D4_C2_VGRA_PAIR_MANIFEST.csv`
- `research/04_data/manifests/D4_C2_VGRA_IMAGE_ASSIGNMENTS.csv`
- `research/04_data/manifests/D4_C2_VGRA_VISIBILITY_SCHEMA.json`
- `research/04_data/manifests/D4_C2_VGRA_PAIRING_SUMMARY.json`
- reproducible generator: `research/tools/d4_vgra_pair_manifest.py`

## Next
`D4-C3_VGRA_HEAD_MODEL_INTEGRATION`
