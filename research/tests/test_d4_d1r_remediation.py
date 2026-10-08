"""D4-D1R governed independent regression tests for discovered integration blockers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.data.base import BaseDataset
from ultralytics.data.vgra import VGRAYOLODataset, bind_vgra_assignments
from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from ultralytics.models.yolo.detect.vgra_trainer import (
    VGRADetectionTrainer,
    _require_explicit_development_val,
)

YAML = "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml"


def _rows():
    return [
        {"split": "val", "filestem": "usable_ap", "pair_id": "P1",
         "view_code": "AP", "structural_role": "PAIRED",
         "operational_role": "SINGLE_FALLBACK_COMPANION_UNREADABLE"},
        {"split": "val", "filestem": "bad_lat", "pair_id": "P1",
         "view_code": "LAT", "structural_role": "PAIRED",
         "operational_role": "EXCLUDED_UNREADABLE"},
    ]


def _dataset():
    ds = object.__new__(VGRAYOLODataset)
    ds.vgra_split = "val"
    ds.vgra_assignment_rows = _rows()
    ds.vgra_expected_stems = {"usable_ap"}
    return ds


def test_frozen_val_exclusion_applied_before_image_label_scan(monkeypatch):
    # Simulate BaseDataset image discovery WITHOUT touching filesystem.
    monkeypatch.setattr(BaseDataset, "get_img_files",
                        lambda self, path: ["synthetic/bad_lat.png", "synthetic/usable_ap.png"])
    ds = _dataset()
    discovered = ds.get_img_files("synthetic-only")
    assert discovered == ["synthetic/usable_ap.png"]
    meta, units = bind_vgra_assignments(discovered, ds.vgra_assignment_rows, "val")
    assert units == [(0,)]
    assert meta[0]["pair_valid"] is False


def test_excluded_val_image_cannot_enter_pair_units():
    with pytest.raises(ValueError, match="frozen unreadable"):
        bind_vgra_assignments(["synthetic/bad_lat.png"], _rows(), "val")


def test_missing_operational_member_fails_closed(monkeypatch):
    monkeypatch.setattr(BaseDataset, "get_img_files",
                        lambda self, path: ["synthetic/bad_lat.png"])
    with pytest.raises(ValueError, match="missing operational"):
        _dataset().get_img_files("synthetic-only")


def test_unrecognized_image_fails_closed(monkeypatch):
    monkeypatch.setattr(BaseDataset, "get_img_files",
                        lambda self, path: ["synthetic/usable_ap.png", "synthetic/unknown.png"])
    with pytest.raises(ValueError, match="outside the frozen"):
        _dataset().get_img_files("synthetic-only")


def test_test_split_cannot_replace_missing_val():
    _require_explicit_development_val({"train": "synthetic/train", "val": "synthetic/val", "test": "sealed/test"})
    with pytest.raises(ValueError, match="TEST fallback"):
        _require_explicit_development_val({"train": "synthetic/train", "test": "sealed/test"})
    with pytest.raises(ValueError, match="must differ"):
        _require_explicit_development_val({"train": "synthetic/train", "val": "sealed/test", "test": "sealed/test"})


def test_final_eval_uses_selected_best_and_paired_trainer_context(monkeypatch, tmp_path):
    model = VGRADetectionModel(cfg=YAML, verbose=False)
    model.bind_train_visibility_weights(torch.ones(9, 4), source="B-TRAIN")
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"synthetic checkpoint stand-in")

    from ultralytics.nn import tasks
    observed = []
    monkeypatch.setattr(tasks, "load_checkpoint", lambda path, device=None: (model, {"epoch": 0}))

    class FakeValidator:
        args = SimpleNamespace(plots=True, compile=True)
        def __call__(self, *, trainer, model):
            observed.append((trainer, model))
            return {"fitness": 0.7, "metrics/mAP50(B)": 0.3}

    trainer = object.__new__(VGRADetectionTrainer)
    trainer.best = ckpt
    trainer.device = torch.device("cpu")
    trainer.data = {"train": "synthetic/train", "val": "synthetic/val", "test": "sealed/test"}
    trainer.args = SimpleNamespace(plots=False)
    trainer.validator = FakeValidator()
    trainer.run_callbacks = lambda name: observed.append(name)
    trainer.final_eval()
    assert observed[0] == (trainer, model)
    assert observed[1] == "on_fit_epoch_end"
    assert trainer.metrics == {"metrics/mAP50(B)": 0.3}
    assert trainer.validator.args.compile is False


def test_final_eval_rejects_missing_best_checkpoint(tmp_path):
    trainer = object.__new__(VGRADetectionTrainer)
    trainer.best = tmp_path / "missing.pt"
    trainer.data = {"train": "synthetic/train", "val": "synthetic/val"}
    with pytest.raises(FileNotFoundError, match="selected-best"):
        trainer.final_eval()
