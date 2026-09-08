# Manuscript and Submission Plan V8

## Reporting standards
Use both:
- CLAIM 2024 for medical-imaging AI
- STARD-AI 2025 for diagnostic-accuracy reporting where applicable

Create checklist compliance during drafting, not after the manuscript is finished.

## Recommended manuscript structure

### Title
Method + clinical task + rigorous evaluation; avoid unsupported "SOTA" in title.

### Abstract
Structured if journal allows:
- Background/Objectives
- Methods
- Results
- Conclusion

Include:
- dataset patients/images
- patient-level split
- primary endpoint
- paired baseline improvement
- robustness/external test
- efficiency

### 1 Introduction
1. pediatric wrist diagnostic problem
2. limits of current YOLO improvements
3. long-tail and multi-view/anatomical structure
4. gap in preserving transfer + using axis/projection/patient information
5. contributions

### 2 Related Work
- GRAZPEDWRI-DX detector evolution
- detail-preserving architectures
- long-tail medical detection
- projection/multi-view/anatomical conditioning
- parameter-efficient transfer adaptation

### 3 Materials and Methods
3.1 Dataset/reference standard
3.2 Split-B and leakage control
3.3 Baseline and training calibration
3.4 D00 diagnostic rationale
3.5 TP-CDA
3.6 APCF
3.7 PELT
3.8 Training protocol
3.9 Evaluation metrics/statistics
3.10 External/generalization protocol
3.11 Reproducibility/software

### 4 Results
4.1 dataset/diagnostic findings
4.2 historical baseline/negative ablation
4.3 mechanism ablation
4.4 combined model/seed confirmation
4.5 CV robustness
4.6 sealed test
4.7 external generalization
4.8 efficiency
4.9 explainability/error analysis

### 5 Discussion
- what worked and why
- comparison with literature
- rare-class instability
- transfer-preservation insight
- projection/anatomy findings
- clinical implications
- limitations
- future paired-view/multicenter work

### 6 Conclusion

## Journal strategy
Do not choose solely by current model performance.
Re-evaluate journal at method lock.

If final evidence is strong (clear novelty, >=~1 pp paired gain, CV + external evidence):
- consider Biomedical Signal Processing and Control and other strong Q2 imaging/biomedical-engineering venues.

If gain is modest but methodology is rigorous:
- Biomedical Engineering Letters or another suitable SCIE Q2/Q3 engineering journal may be more realistic.

Before submission verify in live JCR:
- SCIE status
- category/quartile
- APC/subscription route
- manuscript limits
- first-decision metrics
- AI-use disclosure policy

## Submission package
- manuscript
- cover letter
- graphical abstract/highlights if required
- supplementary methods/results
- CLAIM checklist
- STARD-AI checklist if applicable
- data availability statement
- code availability statement
- model/split manifests
- conflict/funding/ethics statements
- AI-assisted writing disclosure if required by journal
