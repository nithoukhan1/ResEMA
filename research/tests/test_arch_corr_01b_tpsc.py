from __future__ import annotations

import inspect
from pathlib import Path

import torch
import yaml

from ultralytics.nn.modules.block import C3k2, C3k2_TPSC, C3k2_TPSCG4


ROOT = Path(__file__).resolve().parents[2]


def _native_signature_names(cls):
    return [
        p.name
        for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self"
    ][:8]


def _copy_native_state(native, target):
    result = target.load_state_dict(native.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(".sc_adapters." in key for key in result.missing_keys)
    for key, value in native.state_dict().items():
        assert key in target.state_dict()
        assert target.state_dict()[key].shape == value.shape
        assert torch.equal(target.state_dict()[key], value)


def test_tpsc_preserves_native_constructor_semantics():
    expected = ["c1", "c2", "n", "c3k", "e", "attn", "g", "shortcut"]
    assert _native_signature_names(C3k2) == expected
    assert _native_signature_names(C3k2_TPSC) == expected
    assert _native_signature_names(C3k2_TPSCG4) == expected


def test_tpsc_zero_gate_is_native_identity_for_bottleneck_mode():
    torch.manual_seed(42)
    native = C3k2(64, 64, n=2, c3k=False, e=0.5, shortcut=True).eval()
    target = C3k2_TPSC(64, 64, n=2, c3k=False, e=0.5, shortcut=True).eval()
    _copy_native_state(native, target)
    x = torch.randn(2, 64, 16, 16)
    with torch.no_grad():
        a = native(x)
        b = target(x)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    assert all(float(adapter.alpha.detach()) == 0.0 for adapter in target.sc_adapters)


def test_tpsc_g4_zero_gate_is_native_identity_for_c3k_mode():
    torch.manual_seed(42)
    native = C3k2(128, 128, n=1, c3k=True, e=0.5, shortcut=True).eval()
    target = C3k2_TPSCG4(128, 128, n=1, c3k=True, e=0.5, shortcut=True).eval()
    _copy_native_state(native, target)
    x = torch.randn(1, 128, 16, 16)
    with torch.no_grad():
        a = native(x)
        b = target(x)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_tpsc_gate_can_receive_gradient_at_identity_initialization():
    torch.manual_seed(42)
    module = C3k2_TPSC(64, 64, n=1, c3k=False, e=0.5, shortcut=True).train()
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    module(x).square().mean().backward()
    alpha_grad = module.sc_adapters[0].alpha.grad
    assert alpha_grad is not None
    assert torch.isfinite(alpha_grad)
    assert alpha_grad.abs().item() > 0.0


def test_candidate_yaml_placements_are_controlled():
    early = yaml.safe_load(
        (ROOT / "ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml").read_text(encoding="utf-8")
    )
    g4 = yaml.safe_load(
        (ROOT / "ultralytics/cfg/models/11/yolo11s-tpsc-g4-v1.yaml").read_text(encoding="utf-8")
    )

    assert [early["backbone"][i][2] for i in (2, 4, 6, 8)] == [
        "C3k2_TPSC", "C3k2_TPSC", "C3k2", "C3k2"
    ]
    assert [g4["backbone"][i][2] for i in (2, 4, 6, 8)] == [
        "C3k2_TPSCG4", "C3k2_TPSCG4", "C3k2_TPSCG4", "C3k2_TPSCG4"
    ]
    assert all(layer[2] not in {"C3k2_TPSC", "C3k2_TPSCG4"} for layer in early["head"])
    assert all(layer[2] not in {"C3k2_TPSC", "C3k2_TPSCG4"} for layer in g4["head"])


def test_parser_registers_tpsc_as_native_c3k2_family():
    text = (ROOT / "ultralytics/nn/tasks.py").read_text(encoding="utf-8")
    assert "C3k2_TPSC" in text
    assert "C3k2_TPSCG4" in text
    assert "if m in {C3k2, C3k2_TPSC, C3k2_TPSCG4}" in text