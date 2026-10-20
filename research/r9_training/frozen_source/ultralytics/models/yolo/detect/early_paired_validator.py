# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D4-E2 R2 candidate: native Early detection validator over governed paired VAL loader.

Development validation only. Standalone/test access is not exposed.
"""
from __future__ import annotations

import torch

from ultralytics.models.yolo.detect.val import DetectionValidator
from .d4e_fixed_conf_metric import FixedThresholdBValMixin
from ultralytics.utils import RANK, TQDM
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import smart_inference_mode, unwrap_model

from .vgra_fail_closed import GuardedEarlyDetectionModel, require_finite_model


class EarlyPairedValidator(FixedThresholdBValMixin, DetectionValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        if trainer is None:
            raise RuntimeError("D4 Early paired validator requires governed B-VAL trainer")
        if trainer.world_size not in {0, 1} or RANK not in {-1, 0}:
            raise NotImplementedError("D4 Early paired DDP validation unauthorized")
        if bool(trainer.args.compile) or self.dataloader is None:
            raise RuntimeError("D4 Early paired validator requires uncompiled pair-aware loader")
        self.training = True
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != "cpu" and bool(trainer.amp)
        if model is None:
            model = trainer.ema.ema if trainer.ema is not None and trainer.ema.ema is not None else trainer.model
        model = unwrap_model(model)
        if not isinstance(model, GuardedEarlyDetectionModel):
            raise TypeError("D4 Early paired validator requires guarded Early model")
        require_finite_model(model, "before_early_validation")
        model = model.half() if self.args.half else model.float()
        model.eval()
        self.loss = torch.zeros_like(trainer.loss_items, device=self.device)
        self.args.plots &= trainer.stopper.possible_stop or trainer.epoch == trainer.epochs - 1
        self.run_callbacks("on_val_start")
        self.init_metrics(model)
        self._d4_init_fixed_bval()
        self.jdict = []
        counters = tuple(Profile(device=self.device) for _ in range(4))
        for batch_i, batch in enumerate(TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))):
            self.batch_i = batch_i
            self.run_callbacks("on_val_batch_start")
            with counters[0]:
                batch = self.preprocess(batch)
            with counters[1]:
                pred = model(batch["img"])
                if not isinstance(pred, tuple) or len(pred) != 2:
                    raise RuntimeError("D4 Early eval requires native (decoded,raw) tuple")
                decoded, raw = pred
            with counters[2]:
                loss_vector, display_loss = model.loss(batch, raw)
                if loss_vector.shape != (3,) or display_loss.shape != (3,):
                    raise RuntimeError("D4 Early validation requires three native losses")
                self.loss += display_loss
            with counters[3]:
                detections = self.postprocess(decoded)
            self.update_metrics(detections, batch)
            self._d4_update_fixed_bval(detections, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, detections, batch_i)
            self.run_callbacks("on_val_batch_end")
        self.gather_stats()
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(),
                              (profile.t / len(self.dataloader.dataset) * 1e3 for profile in counters)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        model.float()
        if len(self.dataloader) == 0:
            raise RuntimeError("D4 Early paired VAL loader empty")
        losses = trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")
        # Both arms must evaluate every operational B-VAL image exactly once.
        # The known unreadable frozen member is excluded by the governed loader.
        fixed = self._d4_finalize_fixed_bval()
        if (
            self._d4_fixed_seen != 3049
            or self.seen != 3049
            or len(self.dataloader.dataset) != 3049
        ):
            raise RuntimeError("D4_OPERATIONAL_BVAL_CARDINALITY_DRIFT")
        return {key: round(float(value), 5) for key, value in {**stats, **losses, **fixed}.items()}


__all__ = ("EarlyPairedValidator",)
