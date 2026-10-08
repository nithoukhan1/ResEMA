"""D4-C5B synthetic mixed-batch raw dispatch + native loss tests. No real data access."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from ultralytics.models.yolo.detect.vgra_targets import (
    VISIBILITY_LOSS_WEIGHT_V1,
    visibility_targets_from_batch,
    visibility_weighted_ce,
)
from ultralytics.utils.loss import v8DetectionLoss

VGRA_YAML = "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml"


def _model():
    torch.manual_seed(17)
    model = VGRADetectionModel(cfg=VGRA_YAML, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, fl_gamma=0.0)
    model.train()
    return model


def _batch(paired=True):
    torch.manual_seed(34)
    n = 3
    imgs = torch.randn(n, 3, 64, 64)
    # Every image has one object, preserving exactly one original image index.
    boxes = torch.tensor(
        [[0.5, 0.5, 0.25, 0.20],
         [0.6, 0.6, 0.25, 0.22],
         [0.4, 0.4, 0.20, 0.20]], dtype=torch.float32
    )
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 1.0, 2.0]),
        "cls": torch.tensor([[3.0], [3.0], [7.0]]),
        "bboxes": boxes,
        "vgra_pair_index": torch.tensor([1, 0, -1] if paired else [-1, -1, -1]),
        "vgra_pair_valid": torch.tensor([True, True, False] if paired else [False, False, False]),
        "vgra_view_code": torch.tensor([1, 2, 1]),
        "vgra_pair_id": ("P1", "P1", "") if paired else ("", "", ""),
    }
    return batch


def _bind_weights(model):
    model.bind_train_visibility_weights(torch.ones(9, 4), source="B-TRAIN")


def test_native_single_image_predict_forward_is_inherited():
    model = _model()
    batch = _batch()
    raw = model.predict(batch["img"])
    assert raw["scores"].shape[0] == 3
    assert raw["boxes"].shape[0] == 3


def test_rho_zero_mixed_raw_is_exact_native_identity_and_box_firewall():
    model = _model()
    batch = _batch()
    with torch.no_grad():
        native = model._predict_once(batch["img"])
        routed = model.forward_vgra_batch(batch)
    assert torch.equal(routed["boxes"], native["boxes"])
    assert torch.equal(routed["scores"], native["scores"])
    assert routed["vgra_visibility_logits"].shape == (1, 9, 4)
    assert routed["vgra_ap_indices"].tolist() == [0]
    assert routed["vgra_lat_indices"].tolist() == [1]


def test_active_vgra_modifies_only_pairs_and_preserves_single_and_boxes():
    model = _model()
    with torch.no_grad():
        head = model.model[-1]
        for level in head.vgra.residual.levels:
            level.rho.fill_(0.6)
        head.vgra.visibility.fc2.weight.zero_()
        head.vgra.visibility.fc2.bias.view(9, 4)[:, :].zero_()
        head.vgra.visibility.fc2.bias.view(9, 4)[:, 3] = 10.0
        batch = _batch()
        native = model._predict_once(batch["img"])
        routed = model.forward_vgra_batch(batch)

    assert torch.equal(routed["boxes"], native["boxes"])
    assert torch.equal(routed["scores"][2], native["scores"][2])  # single exactly native
    assert not torch.equal(routed["scores"][:2], native["scores"][:2])


def test_no_pair_batch_is_exact_native_and_empty_visibility():
    model = _model()
    batch = _batch(paired=False)
    with torch.no_grad():
        native = model._predict_once(batch["img"])
        routed = model.forward_vgra_batch(batch)
    assert routed["vgra_visibility_logits"].shape == (0, 9, 4)
    assert torch.equal(native["boxes"], routed["boxes"])
    assert torch.equal(native["scores"], routed["scores"])


def test_visibility_weights_fail_closed_without_train_binding():
    model = _model()
    batch = _batch()
    preds = model.forward_vgra_batch(batch)
    with pytest.raises(RuntimeError, match="not bound"):
        model.init_criterion()(preds, batch)
    with pytest.raises(ValueError, match="B-TRAIN"):
        model.bind_train_visibility_weights(torch.ones(9, 4), source="B-VAL")


def test_mixed_native_detection_loss_and_visibility_scaling():
    model = _model()
    _bind_weights(model)
    batch = _batch()
    preds = model.forward_vgra_batch(batch)
    native_total, native_items = v8DetectionLoss(model)(preds, batch)
    total, items = model.init_criterion()(preds, batch)
    target, _, _ = visibility_targets_from_batch(batch, 9)
    vis = visibility_weighted_ce(preds["vgra_visibility_logits"], target,
                                 model.vgra_visibility_weights)
    expected_vis = VISIBILITY_LOSS_WEIGHT_V1 * 2 * vis
    assert total.shape == (4,)
    torch.testing.assert_close(total[:3], native_total)
    torch.testing.assert_close(total[3], expected_vis)
    torch.testing.assert_close(total.sum(), native_total.sum() + expected_vis)
    torch.testing.assert_close(items[:3], native_items)
    torch.testing.assert_close(items[3], (VISIBILITY_LOSS_WEIGHT_V1 * 2 * vis / 3).detach())


def test_single_batch_native_detection_loss_no_visibility_weights_needed():
    model = _model()
    batch = _batch(paired=False)
    preds = model.forward_vgra_batch(batch)
    total, items = model.init_criterion()(preds, batch)
    native_total, native_items = v8DetectionLoss(model)(preds, batch)
    torch.testing.assert_close(total[:3], native_total)
    assert total.shape == (4,) and total[3].item() == 0.0
    torch.testing.assert_close(items[:3], native_items)
    assert items[3].item() == 0.0


def test_gradient_reaches_native_heads_visibility_head_and_residual_gate():
    model = _model()
    _bind_weights(model)
    batch = _batch()
    total, items = model(batch)
    assert total.shape == (4,) and items.shape == (4,)
    assert torch.isfinite(total).all()
    total.sum().backward()
    head = model.model[-1]
    checked = (
        head.cv2[0][-1].weight,
        head.cv3[0][-1].weight,
        head.vgra.visibility.fc2.weight,
        head.vgra.residual.levels[0].rho,
    )
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in checked)


def test_bad_companion_map_fails_before_model_forward():
    model = _model()
    batch = _batch()
    batch["vgra_pair_index"] = torch.tensor([2, 0, -1])
    with pytest.raises(ValueError, match="reciprocal"):
        model.forward_vgra_batch(batch)


def test_weight_buffers_checkpoint_roundtrip_and_no_trainable_parameter_growth():
    model = _model()
    _bind_weights(model)
    state = model.state_dict()
    assert "vgra_visibility_weights" in state
    assert "vgra_train_weights_ready" in state
    model2 = _model()
    model2.load_state_dict(state, strict=True)
    assert bool(model2.vgra_train_weights_ready.item())
    torch.testing.assert_close(model2.vgra_visibility_weights, torch.ones(9, 4))
    assert sum(p.numel() for p in model.parameters()) == 9_672_660
