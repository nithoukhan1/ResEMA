from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.custom import DySample
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[2]


def test_dysample_lp_shape_and_gradient():
    torch.manual_seed(42)
    module = DySample(32, scale=2, style="lp", groups=4).train()
    x = torch.randn(2, 32, 7, 9, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 32, 14, 18)
    y.square().mean().backward()
    assert module.offset.weight.grad is not None
    assert torch.isfinite(module.offset.weight.grad).all()


def test_dysample_pl_shape_and_gradient():
    torch.manual_seed(42)
    module = DySample(32, scale=2, style="pl", groups=4).train()
    x = torch.randn(2, 32, 7, 9, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 32, 14, 18)
    y.square().mean().backward()
    assert module.offset.weight.grad is not None
    assert torch.isfinite(module.offset.weight.grad).all()


def test_dysample_init_pos_and_fail_closed_contracts():
    module = DySample(32, scale=2, style="lp", groups=4)
    assert module.init_pos.shape == (1, 32, 1, 1)
    with pytest.raises(ValueError):
        DySample(30, scale=2, style="lp", groups=4)
    with pytest.raises(ValueError):
        DySample(30, scale=2, style="pl", groups=4)
    with pytest.raises(ValueError):
        DySample(32, scale=2, style="bad", groups=4)


def test_yolo11s_dysample_parameter_delta_matches_registry():
    baseline = DetectionModel(
        str(ROOT / "ultralytics/cfg/models/11/yolo11s.yaml"),
        ch=3,
        nc=9,
        verbose=False,
    )
    dysample = DetectionModel(
        str(ROOT / "ultralytics/cfg/models/11/yolo11s-dysample-v2.yaml"),
        ch=3,
        nc=9,
        verbose=False,
    )
    baseline_params = sum(p.numel() for p in baseline.parameters())
    dysample_params = sum(p.numel() for p in dysample.parameters())
    assert baseline_params == 9_431_275
    assert dysample_params == 9_455_915
    assert dysample_params - baseline_params == 24_640
