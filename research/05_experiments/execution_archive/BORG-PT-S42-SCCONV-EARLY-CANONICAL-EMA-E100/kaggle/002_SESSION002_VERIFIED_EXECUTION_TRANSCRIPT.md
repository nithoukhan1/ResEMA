# COMB-01 Session 002 — verified execution transcript

Experiment:

`BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100`

## Frozen resume state

- Scientific source commit:
  `ab76873d98d040cbe8f06a77e2810a7c647b995b`
- Execution commit:
  `22117d7334b53c7782a872a0114d0831fe41cb57`
- Data binding:
  `DATA01:B-ORG:v1`
- Resume session index: 1
- Completed epochs before resume: 87
- Next epoch: 88
- Total epochs: 100
- Remaining epochs: 13
- Test access: `NONE`

## Parent binding

- `results.csv` SHA256:
  `675327ec39220ebbb19c94ce9ed20719efe25d590e1baaa2199500740dd4c989`
- `last.pt` SHA256:
  `e99c43994b06d6a3e8f578f0a9bda9efdb05f5585575aeb96500a972795cf523`
- `best.pt` SHA256:
  `757f27923cc0f3df1035d87eb85626d799bc909a06a43266f3c6b003ea6f806c`
- `args.yaml` SHA256:
  `332136ec15053d50d13f8a39ec479e8ae5b2bbf58ea7b74ca982c4c2192bec41`

## Resume preflight

Executed command:

`/usr/bin/python3 -u -m research.runtime.combination_screen_resume_runner --experiment-id BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100 --mode preflight-only --input-root /kaggle/input --work-root /kaggle/working`

Verified output included:

- `COMPLETED_EPOCHS=87`
- `NEXT_EPOCH_ONE_BASED=88`
- `TOTAL_EPOCHS=100`
- `EXPECTED_PARAMETERS=9574241`
- `EXPECTED_STATE_ITEMS=565`
- `PREVIEW_SHA256=3d5ba25559f08ff892fb0ac863399cc9e09871dfb03cb7d3c645f719852c82e4`
- `DESTINATION_RUN_MATERIALIZED=FALSE`
- `TEST_ACCESS=NONE`
- `COMB01_RESUME_PREFLIGHT=PASS`

## Governed resume execution

Executed command:

`/usr/bin/python3 -u -m research.runtime.combination_screen_resume_runner --experiment-id BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100 --mode execute --input-root /kaggle/input --work-root /kaggle/working`

Training log explicitly reported:

`Resuming training .../weights/last.pt from epoch 88 to 100 total epochs`

The run then completed epochs 88 through 100.

## Final state

- Final completed epochs: 100
- Final `results.csv` SHA256:
  `63258d1a04c4629d08725d9ef189aaa71f915c0c4e280d5a14c5fe55c207c1a4`
- Final `best.pt` SHA256:
  `531c9927592781e7e19b6af1a35ce910113758d7ad5842677a5b7065487c897d`
- Final `last.pt` SHA256:
  `a560a2a87fa577cb83b1dd63c461270153304c3e245df81d66e8f80b0037f3ee`
- Save-Version output:
  `vickyhit/borg-pt-s42-scconv-early-canonical-ema-e100-2`
- Training complete: `TRUE`
- Test access: `NONE`

## Capture classification

`VERIFIED_EXECUTION_TRANSCRIPT`

This file records the verified executed commands and resulting governed state.
It does not claim a byte-for-byte export of the original Kaggle notebook.
The scientific implementation itself is preserved exactly by the bound Git commit.
