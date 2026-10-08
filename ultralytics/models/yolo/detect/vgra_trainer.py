# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA V1 governed pair-aware trainer (D4-C5C).

No modification to stock DetectionTrainer. This class must be constructed with
explicit TRAIN/VAL frozen assignment manifest paths. It does not itself authorize
training; execution requires a separate D4-E/D4-F experiment contract.
"""

from __future__ import annotations

import hashlib
from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.vgra import build_vgra_dataloader, build_vgra_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from ultralytics.models.yolo.detect.vgra_targets import train_state_weight_binding
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .vgra_validator import VGRADetectionValidator


ASSIGNMENT_MANIFEST_SHA256 = "87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4"
VGRA_YAML_PATH = "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml"


def _sha256(path: Path) -> str:
    hashobj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hashobj.update(chunk)
    return hashobj.hexdigest()


def _check_manifest(path: str | Path, name: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen {name} assignment manifest missing: {path}")
    actual = _sha256(path)
    if actual != ASSIGNMENT_MANIFEST_SHA256:
        raise ValueError(f"{name} frozen D4-C2 assignment SHA drift: {actual}")
    return path


def _fail_closed_config(cfg: Any) -> None:
    """Protect paired-batch integrity and unsupported experimental options."""
    if bool(getattr(cfg, "compile", False)):
        raise ValueError("VGRA V1 compile=True unsupported; refuse compiled native-only forward")
    if getattr(cfg, "fraction", 1.0) != 1.0:
        raise ValueError("VGRA V1 requires fraction=1.0 for complete pair accounting")
    if float(getattr(cfg, "multi_scale", 0.0)) != 0.0:
        raise ValueError("VGRA V1 requires multi_scale=0 pending separate verification")
    if int(getattr(cfg, "batch", 16)) < 2:
        raise ValueError("VGRA V1 batch size must be >= 2")
    if bool(getattr(cfg, "single_cls", False)):
        raise ValueError("VGRA V1 nine-class visibility supervision forbids single_cls=True")


def _require_explicit_development_val(data: dict[str, Any]) -> None:
    """Never fall back from missing development VAL to the sealed TEST split."""
    if not data.get("train") or not data.get("val"):
        raise ValueError("VGRA training requires explicit B-TRAIN and B-VAL paths; TEST fallback is prohibited")
    if data.get("test") and data["val"] == data["test"]:
        raise ValueError("VGRA B-VAL path must differ from the sealed B-TEST path")


class VGRADetectionTrainer(DetectionTrainer):
    """Specialized trainer preserving standard YOLO training loop with pair-safe inputs."""

    def __init__(
        self,
        *,
        train_assignment_manifest: str | Path,
        val_assignment_manifest: str | Path,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ):
        self.vgra_train_manifest = _check_manifest(train_assignment_manifest, "B-TRAIN")
        self.vgra_val_manifest = _check_manifest(val_assignment_manifest, "B-VAL")
        if self.vgra_train_manifest != self.vgra_val_manifest:
            raise ValueError("VGRA V1 requires the same frozen C2 combined assignment manifest for both splits")
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        _fail_closed_config(self.args)
        _require_explicit_development_val(self.data)
        if self.world_size > 1:
            raise NotImplementedError("VGRA V1 distributed training is not authorized")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = VGRADetectionModel(
            cfg=cfg or VGRA_YAML_PATH,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights is not None:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode not in ("train", "val"):
            raise ValueError("VGRA only supports TRAIN or VAL in this stage")
        if self.data["nc"] != 9:
            raise ValueError("VGRA candidate V1 requires exactly nine detection classes")
        _fail_closed_config(self.args)
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        manifest = self.vgra_train_manifest if mode == "train" else self.vgra_val_manifest
        return build_vgra_yolo_dataset(
            self.args,
            img_path,
            batch if batch is not None else self.args.batch,
            self.data,
            assignment_manifest=manifest,
            mode=mode,
            stride=stride,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if rank != -1 or self.world_size > 1:
            raise NotImplementedError("VGRA V1 requires non-distributed training/validation")
        if mode not in ("train", "val"):
            raise ValueError("VGRA dataloader mode must be train or val")
        _fail_closed_config(self.args)
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)
        if mode == "train":
            # B-TRAIN only. The separate source freeze will register exact count/weight hashes.
            counts, weights = train_state_weight_binding(dataset, nc=9, split="train")
            model = unwrap_model(self.model)
            if not isinstance(model, VGRADetectionModel):
                raise TypeError("VGRA trainer requires VGRADetectionModel")
            model.bind_train_visibility_weights(weights.to(next(model.parameters()).device), source="B-TRAIN")
            self.vgra_train_state_counts = counts.detach().cpu().clone()
            self.vgra_train_state_weights = weights.detach().cpu().clone()

        return build_vgra_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=(mode == "train"),
            seed=self.args.seed,
            rank=rank,
            drop_last=False,
            pin_memory=True,
        )

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "vis_loss")
        return VGRADetectionValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def final_eval(self):
        """Validate the selected BEST checkpoint with the governed pair-aware loader.

        BaseTrainer.final_eval calls validator(model=path) without a trainer,
        which cannot carry AP/LAT metadata. Preserve best-checkpoint selection,
        but explicitly provide both the trainer's paired B-VAL context and
        the loaded best model. Never fall back to native image-only validation.
        """
        from ultralytics.nn.tasks import load_checkpoint

        _require_explicit_development_val(self.data)
        best = Path(self.best)
        if not best.is_file():
            raise FileNotFoundError(f"VGRA selected-best checkpoint missing: {best}")
        selected, _ = load_checkpoint(best, device=self.device)
        if not isinstance(selected, VGRADetectionModel):
            raise TypeError("VGRA best checkpoint did not restore a VGRADetectionModel")
        if not bool(selected.vgra_train_weights_ready.item()):
            raise RuntimeError("VGRA best checkpoint lost B-TRAIN visibility weight binding")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        metrics = self.validator(trainer=self, model=selected)
        if not isinstance(metrics, dict):
            raise TypeError("VGRA final paired validation did not return a metrics dictionary")
        self.metrics = dict(metrics)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")


__all__ = ("VGRADetectionTrainer", "_require_explicit_development_val")
