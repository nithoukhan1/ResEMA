# D00 Input Provenance Freeze

Date: 2026-09-11
Status: FROZEN FOR P1 / D00
Repository base before freeze: `9f51a8af00e65a288774ce41809d1014d79567b3`

## Purpose

Bind the authoritative GRAZPEDWRI-DX inputs and the physical
Split-B Kaggle-upload package before D00 scientific diagnostics.

No model training, architecture modification, test prediction,
or test performance evaluation occurred during this transaction.

## Original source identity

Original `dataset.csv` SHA256:

`55a53cbe9673e5f29ef0f2f6074ae791f1c27a3b48efadcd0b771225ff6379e8`

Canonical source counts:

- 20,327 images
- 6,091 patients
- 47,443 operative YOLO boxes
- 15 empty YOLO detection images

## Frozen Split-B

Authoritative split:

`Frozen-Split-B-v1.0`

Counts:

- train: 4,264 patients / 14,227 images
- validation: 914 patients / 3,050 images
- test: 913 patients / 3,050 images

Rare-class facts:

- bonelesion positive patients: 16 / 4 / 4
- foreignbody positive patients: 4 / 0 / 0
- held-out foreignbody AP: N/A

Candidate-v2 was verified byte-identical to the embedded
accepted-candidate audit for all 26 files.

The frozen package file registry passed 37/37 size and SHA256 checks.

## Physical Kaggle-upload package

User-reported Kaggle dataset reference:

`nettokhan/grazpedwri-dx-split-b`

User-reported Kaggle dataset page:

`https://www.kaggle.com/datasets/nettokhan/grazpedwri-dx-split-b`

The local physical upload source contains:

- clean train images/labels: 14,227 / 14,227
- validation images/labels: 3,050 / 3,050
- test images/labels: 3,050 / 3,050
- historical augmented images/labels: 28,454 / 28,454

For clean train, validation, and test, physical image stems exactly
match Frozen Split-B membership.

Physical membership stem SHA256:

- train: `4ab032d669ee40daeaa6d37680def3ba4aad4dc64b9d0f16b7b658be9e48812b`
- validation: `f36040d4a798cbda113909907ba21e3dac1bb91b268124ec51c4fb2899e4a816`
- test: `edbcaa1393f874cd27903e3b5ffec419f3be3f821ef32fab1d2b99a8bd4e6128`

Physical package binding SHA256:

`5db20741f23496208429db87a4625e2e9fb45ae70071b266696602b30374e16e`

Upload-provenance binding SHA256:

`8dac94724f58f3329ce3014ad9db5190b0a3be7e307ae57fef0bde630713ccce`

The actual Kaggle notebook mount path has not yet been verified.
It must be discovered inside Kaggle before D00 execution.

## Authoritative clean runtime definition

`split_B_original.yaml`

SHA256:

`5afcd08e6025ce16ff57716b88f2bee80bd884a44c542f92e52ccd2c83511999`

It declares:

- `train: images/train`
- `val: images/val`
- `test: images/test`

This is the authoritative package-side clean Split-B definition.

## Historical augmentation branch

The package also contains `train_aug_historical`.

It has:

- 28,454 images
- 28,454 labels
- all 14,227 clean training stems are present
- 14,227 additional distinct stems are present

This branch is historical B0 continuity evidence and is
non-authoritative for D00.

A provenance drift was detected for `split_B_historical_aug.yaml`.

Historical upload-provenance snapshot:

- size: 248 bytes
- SHA256:
  `df8cfe5d70b9127dce2713aa02a0166ff7df13c61b4d67eee3f7c0934b687986`

Current local copy:

- size: 303 bytes
- SHA256:
  `d5e5336b401f88be6bfb8026e26e573738101462fc21d6b971a3d183819b7ff0`

The exact time/reason for this historical configuration change was not
reconstructed. No causal explanation is inferred.

The current file also contains a hard-coded historical Kaggle path and
must not be used as the runtime definition for new D00/mechanism work.

## Upload provenance

The package contains six upload-provenance files.

Five of those six files are mirrored into Git:

- `01_expected_folder_counts.csv`
- `02_image_label_stem_checks.csv`
- `04_critical_small_file_hashes.csv`
- `KAGGLE_PACKAGE_MANIFEST.json`
- `README_KAGGLE_PACKAGE.txt`

The remaining file, the large structure manifest, remains external:

`kaggle_upload_provenance/03_upload_structure_manifest.csv`

Size: 10,049,844 bytes

SHA256:

`2d6e9afdc881dacadb0c51cab08485a4f2576854db7891db71bc46405f311f23`

Its hash is frozen in the machine-readable D00 manifest rather than
adding a 10 MB inventory file to Git.

## Repository mirror policy

Publication-governed text files are strict UTF-8 and LF-only.

For source files that required newline/BOM normalization, the D00
manifest records both the original source SHA256 and repository-mirror
SHA256. No semantic field/value edits are permitted in source mirrors.

## Validation operational note

Frozen validation membership remains 3,050 images.

The known problematic member
`1502_0635264266_05_WRI-R2_M015`
is physically present and remains part of the frozen membership.

Existing project policy uses 3,049 readable validation images /
7,110 objects operationally without changing frozen membership.

## Test seal

Individual test membership files are not copied into the development
repository.

Before P10, test predictions, metrics, errors, geometry, patient-level
performance, or other test outcomes must not influence scientific
decisions.

Pre-P10 access is limited to frozen integrity/hash/membership checks
required to establish leakage-free provenance.

## Mixed-root safety rule

The parent local GRAZPEDWRI-DX workspace is mixed-purpose and contains
original images, physical Split-B copies, historical augmented copies,
and EDA outputs.

No D00 or model-training code may recursively glob that parent root for
images.

Explicit governed Split-B paths and membership must be used.

## Next scientific checkpoint

D00-A - transfer coverage audit.

Split-B test remains sealed.
