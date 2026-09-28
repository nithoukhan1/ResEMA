from __future__ import annotations

import inspect
from pathlib import Path

import torch
import yaml

from ultralytics.nn.modules.block import C3k2, C3k2_TPEMA
from ultralytics.nn.modules.custom import CanonicalEMA


ROOT = Path(__file__).resolve().parents[2]


def _signature_names(cls):
    return [
        p.name
        for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self"
    ][:8]


def test_canonical_ema_matches_released_reference_equation():
    torch.manual_seed(42)
    module = CanonicalEMA(64, factor=8).eval()
    x = torch.randn(2, 64, 13, 17)

    with torch.no_grad():
        actual = module(x)

        b, c, h, w = x.shape
        group_x = x.reshape(b * module.groups, -1, h, w)
        x_h = module.pool_h(group_x)
        x_w = module.pool_w(group_x).permute(0, 1, 3, 2)
        hw = module.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, (h, w), dim=2)
        x1 = module.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = module.conv3x3(group_x)
        x11 = module.softmax(module.agp(x1).reshape(b * module.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * module.groups, c // module.groups, -1)
        x21 = module.softmax(module.agp(x2).reshape(b * module.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * module.groups, c // module.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * module.groups, 1, h, w
        )
        expected = (group_x * weights.sigmoid()).reshape(b, c, h, w)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_tpema_preserves_native_constructor_semantics():
    expected = ["c1", "c2", "n", "c3k", "e", "attn", "g", "shortcut"]
    assert _signature_names(C3k2) == expected
    assert _signature_names(C3k2_TPEMA) == expected


def test_tpema_zero_gate_is_exact_native_identity():
    torch.manual_seed(42)
    native = C3k2(128, 128, n=1, c3k=True, e=0.5, shortcut=True).eval()
    target = C3k2_TPEMA(128, 128, n=1, c3k=True, e=0.5, shortcut=True).eval()
    result = target.load_state_dict(native.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(key.startswith("ema_adapter.") or ".ema_adapter." in key for key in result.missing_keys)
    x = torch.randn(1, 128, 16, 16)
    with torch.no_grad():
        a = native(x)
        b = target(x)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    assert float(target.ema_adapter.alpha.detach()) == 0.0


def test_tpema_gate_receives_gradient_at_identity_initialization():
    torch.manual_seed(42)
    module = C3k2_TPEMA(128, 128, n=1, c3k=False, e=0.5, shortcut=True).train()
    x = torch.randn(2, 128, 12, 12, requires_grad=True)
    module(x).square().mean().backward()
    grad = module.ema_adapter.alpha.grad
    assert grad is not None
    assert torch.isfinite(grad)
    assert grad.abs().item() > 0.0


def test_tpema_yaml_keeps_native_layer_indices_and_backbone():
    data = yaml.safe_load(
        (ROOT / "ultralytics/cfg/models/11/yolo11s-tpema-head-v1.yaml").read_text(encoding="utf-8")
    )
    assert all(layer[2] == "C3k2" for layer in (data["backbone"][2], data["backbone"][4], data["backbone"][6], data["backbone"][8]))
    assert [data["head"][i][2] for i in (2, 5, 8, 11)] == [
        "C3k2_TPEMA", "C3k2_TPEMA", "C3k2_TPEMA", "C3k2_TPEMA"
    ]
    assert data["head"][-1][2] == "Detect"
    assert data["head"][-1][0] == [16, 19, 22]


def test_parser_registers_tpema_in_native_c3k2_family():
    text = (ROOT / "ultralytics/nn/tasks.py").read_text(encoding="utf-8")
    assert "C3k2_TPEMA" in text
    assert "C3k2_TPEMA}" in text or "C3k2_TPEMA}:" in text
