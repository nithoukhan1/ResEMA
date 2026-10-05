# COMB-01 Session 001 — verified execution transcript

Experiment:

`BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100`

## Provenance

- Scientific branch: `research/combination-screen-01`
- Scientific source commit: `ab76873d98d040cbe8f06a77e2810a7c647b995b`
- Execution/authorization commit: `22117d7334b53c7782a872a0114d0831fe41cb57`
- Data binding: `DATA01:B-ORG:v1`
- Test access: `NONE`

## Verified wrapper behavior

The fresh Kaggle Save-Version wrapper performed the following governed orchestration:

1. Verified the Kaggle runtime contract.
2. Verified required Split-B Original and offline pretrained inputs.
3. Cloned `https://github.com/nithoukhan1/ResEMA.git`.
4. Checked out branch `research/combination-screen-01` at execution commit `22117d7334b53c7782a872a0114d0831fe41cb57`.
5. Verified its scientific source parent `ab76873d98d040cbe8f06a77e2810a7c647b995b`.
6. Installed the checked-out repository with editable Ultralytics 8.4.7 using `pip install -e . --no-deps`.
7. Ran the governed COMB-01 fresh preflight.
8. Launched the governed COMB-01 fresh training runner for:
   `BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100`.
9. Preserved the Save-Version output as:
   `vickyhit/borg-pt-s42-scconv-early-canonical-ema-e100-1`.
10. Wrote:
    `COMB01_SESSION001_SAVE_VERSION_SUMMARY.json`.

## Resulting preserved state

- Completed epochs: 87
- Parent `results.csv` SHA256:
  `675327ec39220ebbb19c94ce9ed20719efe25d590e1baaa2199500740dd4c989`
- Parent `last.pt` SHA256:
  `e99c43994b06d6a3e8f578f0a9bda9efdb05f5585575aeb96500a972795cf523`
- Parent `best.pt` SHA256:
  `757f27923cc0f3df1035d87eb85626d799bc909a06a43266f3c6b003ea6f806c`
- Parent `args.yaml` SHA256:
  `332136ec15053d50d13f8a39ec479e8ae5b2bbf58ea7b74ca982c4c2192bec41`

## Capture classification

`VERIFIED_EXECUTION_TRANSCRIPT`

This file does not claim to be a byte-for-byte export of the original Kaggle notebook.
The scientific implementation itself is preserved exactly by the Git commit above.
