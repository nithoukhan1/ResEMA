# BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100

## Purpose

COMBINATION-SCREEN-01 controlled combination experiment:

**YOLO11s + SCConv-Early + Canonical EMA**

The experiment tests whether the two independently positive modules provide complementary validation improvement when combined under the frozen Split-B Original pretrained condition.

## Frozen scientific condition

- Data binding: `DATA01:B-ORG:v1`
- Initialization: pretrained
- Seed: 42
- Epoch budget: 100
- Image size: 1024
- Global batch: 16
- Optimizer: SGD
- Primary selection metric: validation mAP50-95
- Test access: `NONE`

## Scientific provenance

- Scientific branch: `research/combination-screen-01`
- Scientific source commit: `ab76873d98d040cbe8f06a77e2810a7c647b995b`
- Execution/authorization commit: `22117d7334b53c7782a872a0114d0831fe41cb57`
- Candidate parameters: 9,574,241
- Candidate state items: 565
- Technical verification SHA256: `0295a6e90bbe6856517c5e723fbe58901f87d405743a8b6ecf334f335191fe29`

## Execution history

### Session 001

Fresh Kaggle Save-Version execution.

- Epochs completed: 1-87
- Status: interrupted safely / resumed
- Saved output: `vickyhit/borg-pt-s42-scconv-early-canonical-ema-e100-1`
- Parent results.csv SHA256: `675327ec39220ebbb19c94ce9ed20719efe25d590e1baaa2199500740dd4c989`
- Parent last.pt SHA256: `e99c43994b06d6a3e8f578f0a9bda9efdb05f5585575aeb96500a972795cf523`

### Session 002

Self-contained governed Kaggle Save-Version resume.

- Resume session index: 1
- Resume from completed epochs: 87
- Resume start epoch: 88
- Final epoch: 100
- Status: complete
- Final saved output: `vickyhit/borg-pt-s42-scconv-early-canonical-ema-e100-2`
- Test access: `NONE`

## Final validation result

Global best validation epoch from the authoritative 100-row `results.csv`:

- Best epoch: **47**
- Precision: **0.6140700000**
- Recall: **0.6329600000**
- F1: **0.6233719272**
- mAP50: **0.6435700000**
- mAP50-95: **0.4183900000**

Final epoch 100:

- Precision: 0.6151900000
- Recall: 0.6133100000
- F1: 0.6142485615
- mAP50: 0.6187200000
- mAP50-95: 0.4012500000

## Complementarity decision

Frozen references:

- Baseline YOLO11s: 0.41704
- SCConv-Early: 0.43130
- SCConv-4Stage: 0.42587
- Canonical EMA: 0.42796
- DySample: 0.41525

Combination deltas:

- vs baseline: +0.00135
- vs SCConv-Early: -0.01291
- vs Canonical EMA: -0.00957
- vs SCConv-4Stage: -0.00748
- vs DySample: +0.00314

**STRICT_COMPLEMENTARITY = FALSE**

**SCIENTIFIC_DECISION = DO_NOT_PROMOTE_COMBINATION_PRIMARY_METRIC**

The combination marginally exceeds the baseline but underperforms both individually positive constituent modules. Therefore it does not demonstrate positive complementarity under the frozen primary development condition.

## Final artifact hashes

- `results.csv`: `63258d1a04c4629d08725d9ef189aaa71f915c0c4e280d5a14c5fe55c207c1a4`
- `best.pt`: `531c9927592781e7e19b6af1a35ce910113758d7ad5842677a5b7065487c897d`
- `last.pt`: `a560a2a87fa577cb83b1dd63c461270153304c3e245df81d66e8f80b0037f3ee`

## Archive status

Scientific implementation provenance is exact and Git-bound.

- Scientific code status: `EXACT_GIT_CAPTURE`
- Kaggle wrapper status: `VERIFIED_EXECUTION_TRANSCRIPT`
- Original Kaggle notebook bytes preserved in output ZIPs: `FALSE`
- Raw Save-Version outputs: locally preserved and SHA256-bound
- Test access: `NONE`

The Kaggle output ZIPs do not contain the original notebook source. Session 001 contains only an Ultralytics-generated DDP worker script, which is not the launch wrapper, and Session 002 contains no Python or notebook source.

Therefore this archive does not make a byte-for-byte notebook-source claim. Instead it preserves the verified executed orchestration, exact Git scientific implementation, exact checkpoint/result hashes, exact Save-Version references, and raw-output ZIP hashes.

See:

- `evidence/CODE_PROVENANCE.json`
- `evidence/LOCAL_RAW_OUTPUT_BACKUPS.json`
- `kaggle/001_SESSION001_VERIFIED_EXECUTION_TRANSCRIPT.md`
- `kaggle/002_SESSION002_VERIFIED_EXECUTION_TRANSCRIPT.md`
