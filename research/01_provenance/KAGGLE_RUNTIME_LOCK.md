# Kaggle Runtime Lock

Successful controlled experiments used:
- Ultralytics 8.4.7 from the pinned repository
- Python 3.12.13
- PyTorch 2.10.0+cu128
- Tesla T4 x2 for training
- imgsz 1024
- batch 16

Important:
The pinned fork's package requirements and Kaggle runtime may not be identical. Full training should clone the exact Git commit and use an environment-preserving install strategy rather than silently replacing the CUDA/PyTorch stack.

Every future run must generate a runtime manifest.
