# SMS-01 Local Artifact Archive Closure

## Status

`COMPLETE`

SMS-01 raw execution artifacts were preserved locally using the same external-artifact principle used for BASELINE-FREEZE-01.

- Seven source ZIP archives were preserved byte-for-byte.
- Four canonical final sessions passed frozen checkpoint/results/args authority checks.
- Three interrupted parent sessions passed exact stopping-epoch and parent last.pt authority checks.
- Every extracted file has a SHA256 entry in its local per-session manifest.
- Canonical final outputs contain the complete available Ultralytics training/validation plots.
- No duplicate ZIP-content groups remain after correction of the local Canonical-EMA parent download.
- Split-B test access remained NONE.
- Heavy ZIP/checkpoint/image artifacts remain outside Git under artifacts_external/sms01.

## Scientific binding

- Scientific branch: research/single-module-screen-01
- Execution commit: 9d65b7adae3d488f1cb70856476fef0488e9749a
- Training source commit: 660475b3aa3614f58f21c98f760561c72e3f1436
- Data binding: DATA01:B-ORG:v1
- Initialization: pretrained
- Seed: 42
- Target epochs: 100
- Selection split: validation
- Split-B test access: NONE

## Preserved sessions

| Archive | Role | Rows | Last epoch | Images/plots | Classification |
|---|---|---:|---:|---:|---|
| BORG-PT-S42-SCCONV-EARLY-E100-1.zip | resume_parent | 99 | 99 | 7 | LINEAGE_SESSION_PASS |
| BORG-PT-S42-SCCONV-EARLY-E100-2.zip | canonical_final_after_resume | 100 | 100 | 20 | CANONICAL_FINAL_PASS |
| BORG-PT-S42-SCCONV-4STAGE-E100-1.zip | resume_parent | 98 | 98 | 7 | LINEAGE_SESSION_PASS |
| BORG-PT-S42-SCCONV-4STAGE-E100-2.zip | canonical_final_after_resume | 100 | 100 | 20 | CANONICAL_FINAL_PASS |
| BORG-PT-S42-DYSAMPLE-E100-1.zip | canonical_final_single_session | 100 | 100 | 20 | CANONICAL_FINAL_PASS |
| BORG-PT-S42-CANONICAL-EMA-E100-1.zip | resume_parent | 89 | 89 | 4 | LINEAGE_SESSION_PASS |
| BORG-PT-S42-CANONICAL-EMA-E100-2.zip | canonical_final_after_resume | 100 | 100 | 20 | CANONICAL_FINAL_PASS |

## Local archive authority hashes

- SMS_ARCH01B_R1_REPORT.json: 2572c38f188a93a6f600fab024acc212bb2195a42075f07c304dc87488b14631
- SMS01_SESSION_ARCHIVES.csv: 23cd456fa9d320f1698ed00f669997920c3a40bf9c915a08a903b5f065b46529
- SMS01_CANONICAL_FINALS.csv: f1846523fc968799dc8f54973272b1f9a61cf7a4fdfc0378b6eee51d6df8d442
- SMS01_LINEAGE_SESSIONS.csv: bd911a87e1765bea7d6e2f66e99485dbb9318ed9436a6ca85e4cd013a11c9c3a
- SMS01_DUPLICATE_ZIPS.csv: cf797656b03b98564daf5ddcb6b444c828966dfdf8a75673ee9f06bf9ba45f87

The local SMS-01 artifact archive is intentionally excluded from Git. Git stores only compact provenance and closure records.

## Git text normalization

The external local archive remains the byte-level authority for the original downloaded ZIPs and locally generated registry files.

Tracked compact records are stored as UTF-8 without BOM with LF line endings, consistent with the repository research text-normalization policy.

Tracked normalized registry SHA256 values:

- SMS01_LOCAL_SESSION_ARCHIVES.csv: f423300ec86703eb4a43e61acd0798ff73cd4f1564d5e0464124dc664c1a9f26
- SMS01_LOCAL_CANONICAL_FINALS.csv: 448e6a1a297e2fb783050bc48096b9ebe75af5f7e7bf859853ab913ef0115ecf
- SMS01_LOCAL_LINEAGE_SESSIONS.csv: a500dbe57fe4005209c9d46bc1de545c105d8584c8c678e97e605af65723965c
- SMS01_LOCAL_DUPLICATE_ZIPS.txt: 6b972e08ba071bc77afd4319021225fbb47b9d50a0c65febbbad3756d5950ce3

The local-source SHA256 values and tracked normalized SHA256 values are separate provenance layers and must not be conflated.
