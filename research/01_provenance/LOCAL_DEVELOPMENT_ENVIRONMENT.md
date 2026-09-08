# Local Development Environment

Date verified: 2026-09-08

## Role

This environment is used for:
- local repository editing;
- static/integrity testing;
- model construction checks;
- small CPU smoke tests.

It is NOT the canonical scientific training environment.

Scientific training runs remain Kaggle-based and must record their
own immutable runtime manifest.

## Verified local environment

Python:
3.11.9

PyTorch:
2.5.1+cu121

Ultralytics:
8.4.7

Ultralytics import source:
E:\PhD\Admitted\Research\Project 1\ResEMA-Github Repo\ResEMA\ultralytics\__init__.py

This confirms that local tests imported the repository source tree,
rather than an unrelated installed Ultralytics package.

## Verification results

Repository verifier:
PASS

Research integrity:
8 passed

Model smoke tests:
4 passed

Historical model configurations successfully constructed:
- ultralytics/cfg/models/11/yolo11.yaml
- ultralytics/cfg/models/11/yolo11s-sc-e50-v2.yaml
- ultralytics/cfg/models/11/yolo11s-dysample-v2.yaml

A baseline CPU forward pass also completed successfully.

## Important reproducibility distinction

These local package versions are not being declared as the definitive
Kaggle training environment.

Every future Kaggle experiment must record:
- Git commit SHA
- Python
- PyTorch
- torchvision
- Ultralytics source/version
- CUDA
- GPU
- dataset paths/manifests
- training configuration
- relevant file hashes
