# Artifact Policy

Do not commit large model checkpoints to Git.

For every Kaggle run, record:
- immutable Kaggle Model path/version
- file size
- SHA256 for best.pt
- SHA256 for last.pt
- SHA256 for results.csv
- SHA256 for args.yaml
- Git commit SHA
- model YAML/source hash
- dataset manifest hash
- runtime manifest

Small CSV/YAML/JSON/Markdown summaries and selected figures are committed to Git.
