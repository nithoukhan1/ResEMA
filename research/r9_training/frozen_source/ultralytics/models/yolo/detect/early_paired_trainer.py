# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D4-E2 R2 candidate MV-00 Early-only paired control; review-only, no training authority."""
from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.data.vgra import build_vgra_dataloader
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import RANK

from .vgra_fail_closed import (
    GuardedEarlyDetectionModel,
    require_finite_model,
    verify_early_tensor_transfer,
)
from .vgra_trainer import (
    VGRADetectionTrainer, _fail_closed_config, _require_explicit_development_val,
    _rebind_best_detection_loss_args,
)
from .early_paired_validator import EarlyPairedValidator


class EarlyPairedTrainer(VGRADetectionTrainer):
    """Reuses the *same* vetted C2 assignment, dataset and sampler without visibility loss."""

    MODEL_YAML_PATH = "ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml"

    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg is not None and cfg != self.MODEL_YAML_PATH:
            raise ValueError("D4 MV-00 model architecture drift")
        if weights is None or weights is not self._r5_c2_approved_weights:
            raise RuntimeError("D4 MV-00 requires the exact approved in-memory Early weights")
        model = GuardedEarlyDetectionModel(
            cfg=self._r5_c2_yaml_snapshot.model_config(), nc=self.data["nc"], ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.load(weights)
        verify_early_tensor_transfer(weights, model)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        if rank != -1 or self.world_size > 1 or mode not in {"train", "val"}:
            raise NotImplementedError("D4 MV-00 requires non-distributed TRAIN/VAL only")
        _fail_closed_config(self.args)
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)
        # No B-TRAIN visibility supervision, no VGRA branch; same sampler in both arms.
        return build_vgra_dataloader(
            dataset, batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=mode == "train", seed=self.args.seed, rank=rank,
            drop_last=False, pin_memory=True,
        )

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return EarlyPairedValidator(
            dataloader=self.test_loader, save_dir=self.save_dir,
            args=copy(self.args), _callbacks=self.callbacks,
        )

    def final_eval(self):
        """Re-evaluate actually selected Early best.pt over governed pair-aware B-VAL."""
        _require_explicit_development_val(self.data)
        best = Path(self.best)
        if not best.is_file():
            raise FileNotFoundError("D4 MV-00 selected-best checkpoint absent")
        selected, checkpoint = load_checkpoint(best, device=self.device)
        if not isinstance(selected, GuardedEarlyDetectionModel):
            raise TypeError("D4 MV-00 best checkpoint has incorrect model type")
        require_finite_model(selected, "early_selected_best")
        _rebind_best_detection_loss_args(selected, checkpoint, self.args, arm="MV00")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        metrics = self.validator(trainer=self, model=selected)
        if not isinstance(metrics, dict):
            raise RuntimeError("D4 MV-00 paired final B-VAL returned no metrics")
        self.metrics = dict(metrics)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")


__all__ = ("EarlyPairedTrainer",)
