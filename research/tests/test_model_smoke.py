from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import torch  # noqa: E402

from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


ARCHITECTURE_YAMLS = [
    "ultralytics/cfg/models/11/yolo11.yaml",
    "ultralytics/cfg/models/11/yolo11s-sc-e50-v2.yaml",
    "ultralytics/cfg/models/11/yolo11s-dysample-v2.yaml",
    "ultralytics/cfg/models/11/yolo11s-resema-only-v2.yaml",
    "ultralytics/cfg/models/11/yolo11s-sc-dysample-e50-v2.yaml",
    "ultralytics/cfg/models/11/yolo11s-sc-resema-e50-v2.yaml",
    "ultralytics/cfg/models/11/yolo11s-dysample-resema-v2.yaml",
    "ultralytics/cfg/models/11/yolo11s-resema-v2-e50.yaml",
]


def build_yolo11s_target(
    yaml_path: str,
):
    """Build the exact YOLO11s-scale, 9-class architecture used by D00-A."""

    path = ROOT / yaml_path

    assert path.is_file(), path

    cfg = YAML.load(path)

    cfg["scale"] = "s"
    cfg["yaml_file"] = str(path)
    cfg["nc"] = 9

    return DetectionModel(
        cfg=cfg,
        ch=3,
        nc=9,
        verbose=False,
    )


@pytest.mark.parametrize(
    "yaml_path",
    ARCHITECTURE_YAMLS,
)
def test_d00_architecture_builds_at_exact_s_scale(
    yaml_path,
):
    model = build_yolo11s_target(
        yaml_path
    )

    assert model is not None
    assert model.yaml["scale"] == "s"
    assert model.yaml["nc"] == 9

    assert model.model[-1].__class__.__name__ == "Detect"


def test_baseline_cpu_forward():
    model = build_yolo11s_target(
        "ultralytics/cfg/models/11/yolo11.yaml"
    )

    model.eval()

    x = torch.zeros(
        1,
        3,
        128,
        128,
    )

    with torch.no_grad():
        out = model(x)

    assert out is not None
