from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")

from ultralytics import YOLO  # noqa: E402

@pytest.mark.parametrize(
    "yaml_path",
    [
        "ultralytics/cfg/models/11/yolo11.yaml",
        "ultralytics/cfg/models/11/yolo11s-sc-e50-v2.yaml",
        "ultralytics/cfg/models/11/yolo11s-dysample-v2.yaml",
    ],
)
def test_historical_model_yaml_builds(yaml_path):
    model = YOLO(str(ROOT / yaml_path))
    assert model.model is not None

def test_baseline_cpu_forward():
    import torch
    model = YOLO(str(ROOT / "ultralytics/cfg/models/11/yolo11.yaml"))
    model.model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model.model(x)
    assert out is not None
