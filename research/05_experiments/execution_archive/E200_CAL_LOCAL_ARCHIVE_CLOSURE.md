# E200 Epoch-Calibration Local Artifact Archive Closure

## Status

COMPLETE

Experiment: BASE-B-ORG-SCR-S42-E200-CAL

The fresh 200-epoch Split-B Original scratch calibration is preserved as a separate experiment from both BASELINE-FREEZE-01 and SMS-01.

## Proven execution lineage

- Parent archive: BASE-B-ORG-SCR-S42-E200-1.zip
- Parent completed epochs: 104
- Parent last.pt SHA256: ed9d789cedabe1adf9fb913065f2e660829a8e96f4cef69988b47d63cb00ba10
- Governed resume begins at epoch 105
- Final archive: BASE-B-ORG-SCR-S42-E200-2.zip
- Final completed epochs: 200
- Resume lineage: PROVEN
- Resume preflight SHA256: 6e58d38b4f6f054370a723684af66d545acfcd19c876b817a99fecbbfad724a0
- Resume runtime SHA256: d169ac68fc7c24b8f1cc08b5f7bf6e80254034ca98ab9b3871c038af96d2df05
- Split-B test access: NONE

## Historical contract handling

The frozen research/05_experiments/EPOCH_CALIBRATION_EXPERIMENTS.csv entry remains unchanged with its historical status NOT_STARTED.

That record is treated as the immutable execution-authorization contract. This closure separately records the observed completed execution and therefore does not rewrite historical governance evidence.

## Scientific binding

- Training source commit: 08e1db969bd2de68338896e0474f1fba65035220
- Execution commit: e59edd55ae7be1fb3b185e9224e192125fbb63fe
- Data binding: DATA01:B-ORG:v1
- Initialization: scratch
- Seed: 42
- Epoch budget: 200
- Primary metric: val_mAP50-95

## Local archive authority

- E200_ARCH01B_R1_REPORT.json: fa7a79151bc1adca21a9413dab0f2c76a65130689f1940e49760c0ef10928021
- E200_CAL_SESSION_ARCHIVES.csv: d6a6cc9d83a92935838b7038702ca23f03212f1e7688256207565ec91f45d875
- E200_CAL_CANONICAL_FINAL.csv: 5ccbdc81f1c5a3d0ae62e9deeda99208ab15a6de21d9bce8972c85a7e52d34be
- E200_CAL_LINEAGE_SESSION.csv: 1338dd8514e77233c12660d6ff7fa0d3b1694c02e95b70c69dffe3e015a90651

## Git-normalized registry hashes

- E200_CAL_LOCAL_SESSION_ARCHIVES.csv: c47b81bc98dc3aaeaff6473a7daccccbd26f9b0dee5148cbefafa3f7207c5e71
- E200_CAL_LOCAL_CANONICAL_FINAL.csv: f61316702fa906620aa7840099690650bacf19e645b5e3db605b876b09d333c5
- E200_CAL_LOCAL_LINEAGE_SESSION.csv: f78726b5c249e0745474945ba7465d9d5df7430c8767d9074f9d039dc24fa3da

The heavyweight ZIPs, checkpoints, images, and extracted run files remain outside Git under artifacts_external/epoch_calibration_01.
