# EDA Summary

- EDA-00: Split A/B structure and provenance audited.
- EDA-01: patient-disjointness and class/patient imbalance audited.
- EDA-02 to EDA-10: geometry, patient concentration, co-occurrence, metadata, contrast, spatial distribution, frequency behavior, augmentation integrity and data-side synthesis completed.

Main conclusions:
- severe long-tail imbalance exists at box/image/patient levels;
- rare classes include very small objects;
- several clinical classes have weak ROI-context contrast;
- spatial structure is strong for several abnormalities;
- A/B frequency behavior is highly concordant;
- historical offline augmentation preserves labels exactly and sampled pairs are predominantly photometric.

Fresh model-side diagnostics remain pending.
