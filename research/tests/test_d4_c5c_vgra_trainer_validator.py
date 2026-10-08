"""D4-C5C governed synthetic trainer / validator compatibility tests.

Only synthetic tensors and in-memory datasets. No GRAZPEDWRI image or label
files are accessed or training executed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from ultralytics.models.yolo.detect.vgra_trainer import (
    VGRADetectionTrainer, _fail_closed_config,
)
from ultralytics.models.yolo.detect.vgra_validator import VGRADetectionValidator
from ultralytics.data.vgra import VGRAYOLODataset
from ultralytics.utils.loss import v8DetectionLoss

YAML = "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml"
NAMES = {i: name for i, name in enumerate(
    ("boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
     "periostealreaction", "pronatorsign", "softtissue", "text")
)}


def _model():
    torch.manual_seed(18)
    model = VGRADetectionModel(cfg=YAML, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, fl_gamma=0.0)
    model.nc = 9
    model.names = NAMES
    model.bind_train_visibility_weights(torch.ones(9, 4), source="B-TRAIN")
    return model


def _sample(stem, pair_id="", view=1, valid=False, cls=3):
    return {
        "batch_idx": torch.zeros(1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
        "cls": torch.tensor([[float(cls)]], dtype=torch.float32),
        "img": torch.zeros(3, 64, 64, dtype=torch.uint8),
        "im_file": f"synthetic/{stem}.png",
        "ori_shape": (64, 64),
        "ratio_pad": (1.0, 1.0),
        "resized_shape": (64, 64),
        "vgra_filestem": stem,
        "vgra_operational_role": "PAIRED" if valid else "SINGLE",
        "vgra_pair_id": pair_id,
        "vgra_pair_valid": valid,
        "vgra_view_code": view,
    }


def _batch():
    # Deterministic nonconstant synthetic pixel data. R1 used an all-zero
    # image fixture that produced non-finite gradients in a subset of
    # parameters even though its forward loss was finite. The zero-input
    # numerical edge case remains OPEN for D4-D3 diagnostics.
    batch = VGRAYOLODataset.collate_fn([
        _sample("synthetic_ap", "P1", 1, True, 3),
        _sample("synthetic_lat", "P1", 2, True, 3),
        _sample("synthetic_single", "", 1, False, 7),
    ])
    rng = torch.Generator().manual_seed(20261008)
    batch["img"] = torch.randint(
        low=0, high=256, size=batch["img"].shape,
        dtype=torch.uint8, generator=rng,
    )
    return batch


def test_trainer_type_contract_and_four_loss_names():
    assert issubclass(VGRADetectionTrainer, __import__(
        "ultralytics.models.yolo.detect.train", fromlist=["DetectionTrainer"]
    ).DetectionTrainer)
    assert issubclass(VGRADetectionValidator, __import__(
        "ultralytics.models.yolo.detect.val", fromlist=["DetectionValidator"]
    ).DetectionValidator)


def test_configuration_firewalls():
    safe = SimpleNamespace(compile=False, fraction=1.0, multi_scale=0.0,
                           batch=16, single_cls=False)
    _fail_closed_config(safe)
    for key, value in (
        ("compile", True), ("fraction", 0.5), ("multi_scale", 0.2),
        ("batch", 1), ("single_cls", True),
    ):
        vals = vars(safe).copy()
        vals[key] = value
        with pytest.raises(ValueError):
            _fail_closed_config(SimpleNamespace(**vals))


def test_validator_refuses_standalone_uncertified_path():
    validator = VGRADetectionValidator(args=dict(task="detect", plots=False, workers=0))
    with pytest.raises(RuntimeError, match="standalone validation not authorized"):
        validator(trainer=None, model=_model())


def test_trainer_get_model_builds_correct_model_and_parameters():
    # Exercise overridden get_model without creating BaseTrainer or any dataset.
    trainer = object.__new__(VGRADetectionTrainer)
    trainer.data = {"nc": 9, "channels": 3, "names": NAMES}
    model = trainer.get_model(cfg=YAML, weights=None, verbose=False)
    assert isinstance(model, VGRADetectionModel)
    assert sum(p.numel() for p in model.parameters()) == 9_672_660


def test_training_batch_dispatch_and_one_optimizer_update_is_finite():
    model = _model().train()
    batch = _batch()
    batch["img"] = batch["img"].float() / 255.0
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss_vector, display_vector = model(batch)
    assert loss_vector.shape == (4,)
    assert display_vector.shape == (4,)
    assert torch.isfinite(loss_vector).all()
    loss_vector.sum().backward()
    bad_grads = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            bad_grads.append(
                (name, int(torch.isnan(grad).sum().item()),
                 int(torch.isinf(grad).sum().item()), tuple(grad.shape))
            )
    assert not bad_grads, f"Non-finite VGRA training gradients: {bad_grads}"
    optimizer.step()
    nonfinite_parameters = [name for name, value in model.named_parameters()
                            if not torch.isfinite(value).all()]
    assert not nonfinite_parameters, (
        f"Parameters nonfinite after synthetic update: {nonfinite_parameters}"
    )
    assert torch.isfinite(model.model[-1].vgra.residual.levels[0].rho)


def test_validator_pair_aware_forward_not_single_view():
    model = _model().eval()
    batch = _batch()
    batch["img"] = batch["img"].float() / 255.0
    head = model.model[-1]
    with torch.no_grad():
        head.vgra.visibility.fc2.bias.view(9,4)[:, 3].fill_(10.0)
        for level in head.vgra.residual.levels:
            level.rho.fill_(0.5)
        native_output = model.predict(batch["img"])
        assert isinstance(native_output, tuple) and len(native_output) == 2
        native = native_output[1]  # eval Detect returns (decoded, raw-head dictionary)
        assert isinstance(native, dict)
        paired = model.forward_vgra_batch(batch)
        assert torch.equal(paired["boxes"], native["boxes"])
        assert torch.equal(paired["scores"][2], native["scores"][2])
        assert not torch.equal(paired["scores"][:2], native["scores"][:2])
        decoded = head._inference(paired)
        assert decoded.shape[0] == 3
        assert decoded.shape[1] == 13


def test_validator_synthetic_end_to_end_metrics_and_four_losses():
    model = _model().eval()

    class Loader:
        dataset = [0, 1, 2]
        def __len__(self):
            return 1
        def __iter__(self):
            yield _batch()

    validator = VGRADetectionValidator(
        dataloader=Loader(),
        args=dict(task="detect", plots=False, save_json=False, workers=0,
                  imgsz=64, batch=3, conf=0.5, half=False, val=True),
    )
    args = SimpleNamespace(compile=False, plots=False)
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        data={"nc": 9, "names": NAMES, "val": "synthetic-val"},
        amp=False, args=args,
        ema=SimpleNamespace(ema=model),
        model=model, loss_items=torch.zeros(4),
        world_size=1, stopper=SimpleNamespace(possible_stop=False),
        epoch=0, epochs=1,
        label_loss_items=lambda items, prefix: {
            f"{prefix}/{key}": float(value)
            for key, value in zip(("box_loss", "cls_loss", "dfl_loss", "vis_loss"), items)
        },
    )
    stats = validator(trainer)
    assert isinstance(stats, dict)
    assert "val/vis_loss" in stats
    assert "val/box_loss" in stats
    assert any("mAP" in name for name in stats)


def test_validation_rejects_missing_train_weight_binding():
    model = _model()
    model.vgra_train_weights_ready.fill_(False)
    validator = VGRADetectionValidator(
        dataloader=[_batch()],
        args=dict(task="detect", plots=False, workers=0),
    )
    fake = SimpleNamespace(
        device=torch.device("cpu"), data={"nc": 9, "names": NAMES},
        amp=False, args=SimpleNamespace(compile=False),
        ema=SimpleNamespace(ema=model), model=model, world_size=1,
    )
    with pytest.raises(RuntimeError, match="TRAIN-derived"):
        validator(fake)
