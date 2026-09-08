# 11 — Manuscript Evidence Map

## Introduction claims -> required evidence
- subtle pediatric findings -> dataset + recent detection literature
- long tail -> class counts + patient-level EDA + recent long-tail literature
- projection/anatomy variation -> dataset metadata/axis + D00 projection analysis

## Methods claims -> repository evidence
- split generation -> frozen scripts/manifests
- pretrained initialization -> transfer audit
- TP-CDA -> source + identity test
- APCF -> source + axis/projection parser tests
- PELT -> sampler + exposure audit
- training -> committed configs + runtime manifests

## Planned Results tables
1. historical protocol/recipe ablation
2. old architecture negative ablation
3. M01/M02/M03 mechanism ablation
4. combined method ablation
5. two-seed confirmation
6. 3-fold robustness
7. sealed test
8. external/efficiency

## Figures
- method diagram
- patient-disjoint split
- patient/class imbalance
- axis/projection examples
- ablation chart
- fold-wise paired plot
- qualitative success/failure examples

## Limitations
- single-center main dataset
- extreme rare classes
- foreignbody held-out N/A
- projection-metadata dependency
- possible external label/domain mismatch
