from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.nn.modules import Detect, VGRADetect
from ultralytics.nn.tasks import DetectionModel


CHANNELS = (128, 256, 512)
NC = 9
EARLY_YAML = Path("ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml")
VGRA_YAML = Path("ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml")


def _features(batch=2):
    torch.manual_seed(123)
    return [
        torch.randn(batch, 128, 20, 20),
        torch.randn(batch, 256, 10, 10),
        torch.randn(batch, 512, 5, 5),
    ]


def _native_raw(head: VGRADetect, features):
    return head.forward_head(features, **head.one2many)


def test_vgra_detect_is_detect_subclass_and_single_view_fallback_is_native():
    head = VGRADetect(nc=NC, ch=CHANNELS)
    assert isinstance(head, Detect)
    head.train()
    x = _features()
    inherited = head(x)
    native = _native_raw(head, x)
    torch.testing.assert_close(inherited["boxes"], native["boxes"], rtol=0, atol=0)
    torch.testing.assert_close(inherited["scores"], native["scores"], rtol=0, atol=0)


def test_zero_rho_paired_path_is_exact_native_identity_for_boxes_and_scores():
    head = VGRADetect(nc=NC, ch=CHANNELS)
    head.train()
    ap = _features()
    lat = [x.clone() * 0.7 for x in _features()]
    ap_native = _native_raw(head, ap)
    lat_native = _native_raw(head, lat)

    paired = head.forward_pair_heads(ap, lat)

    torch.testing.assert_close(paired["ap"]["boxes"], ap_native["boxes"], rtol=0, atol=0)
    torch.testing.assert_close(paired["lat"]["boxes"], lat_native["boxes"], rtol=0, atol=0)
    torch.testing.assert_close(paired["ap"]["scores"], ap_native["scores"], rtol=0, atol=0)
    torch.testing.assert_close(paired["lat"]["scores"], lat_native["scores"], rtol=0, atol=0)


def test_active_vgra_changes_scores_but_never_boxes():
    torch.manual_seed(7)
    head = VGRADetect(nc=NC, ch=CHANNELS)
    head.train()

    with torch.no_grad():
        # Make pair visibility strongly favor shared presence for every class.
        head.vgra.visibility.fc1.weight.zero_()
        head.vgra.visibility.fc1.bias.zero_()
        head.vgra.visibility.fc2.weight.zero_()
        bias = head.vgra.visibility.fc2.bias.view(NC, 4)
        bias.zero_()
        bias[:, 3] = 10.0
        for level in head.vgra.residual.levels:
            level.rho.fill_(0.5)

    ap = _features()
    lat = [x.clone() * 0.5 + 0.1 for x in _features()]
    ap_native = _native_raw(head, ap)
    lat_native = _native_raw(head, lat)

    paired = head.forward_pair_heads(ap, lat)

    torch.testing.assert_close(paired["ap"]["boxes"], ap_native["boxes"], rtol=0, atol=0)
    torch.testing.assert_close(paired["lat"]["boxes"], lat_native["boxes"], rtol=0, atol=0)
    assert not torch.equal(paired["ap"]["scores"], ap_native["scores"])
    assert not torch.equal(paired["lat"]["scores"], lat_native["scores"])


def test_vgra_pair_path_preserves_native_output_shapes():
    head = VGRADetect(nc=NC, ch=CHANNELS)
    head.train()
    paired = head.forward_pair_heads(_features(batch=3), _features(batch=3))
    total_anchors = 20 * 20 + 10 * 10 + 5 * 5
    assert paired["ap"]["boxes"].shape == (3, 64, total_anchors)
    assert paired["lat"]["boxes"].shape == (3, 64, total_anchors)
    assert paired["ap"]["scores"].shape == (3, NC, total_anchors)
    assert paired["lat"]["scores"].shape == (3, NC, total_anchors)


def test_vgra_yaml_parses_to_vgra_detect_and_expected_parameter_count():
    model = DetectionModel(str(VGRA_YAML), verbose=False)
    assert isinstance(model.model[-1], VGRADetect)
    params = sum(p.numel() for p in model.parameters())
    assert params == 9_672_660
    assert params <= 9_870_093


def test_early_yaml_remains_stock_detect_and_parameter_count_unchanged():
    model = DetectionModel(str(EARLY_YAML), verbose=False)
    assert type(model.model[-1]) is Detect
    params = sum(p.numel() for p in model.parameters())
    assert params == 9_570_093


def test_vgra_yaml_single_view_forward_smoke():
    model = DetectionModel(str(VGRA_YAML), verbose=False)
    model.eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, tuple)
    decoded, raw = y
    assert decoded.shape[0] == 1
    assert raw["boxes"].shape[0] == 1
    assert raw["scores"].shape[1] == NC


def test_vgra_head_rejects_end2end_paired_path():
    head = VGRADetect(nc=NC, end2end=True, ch=CHANNELS)
    try:
        head.forward_pair_heads(_features(), _features())
    except RuntimeError as exc:
        assert "does not support end2end" in str(exc)
    else:
        raise AssertionError("expected paired VGRA path to reject end2end mode")
