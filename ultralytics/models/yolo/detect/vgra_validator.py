# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA V1 paired-view validation under the native detection metric conventions.

Stock BaseValidator calls model(batch["img"]), which discards the companion
metadata and would silently run VGRA as single-view. This subclass instead
runs `model.forward_vgra_batch(batch)` and feeds the decoded predictions to
the unchanged native DetectionValidator NMS and metric functions.
"""

from __future__ import annotations

from copy import copy

import torch

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from ultralytics.utils import RANK, TQDM
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import smart_inference_mode, unwrap_model


class VGRADetectionValidator(DetectionValidator):
    """TRAIN/VAL, pair-aware raw inference plus native YOLO NMS/metrics.

    The standalone validator API is not opened in C5C: calling without a
    trainer fails closed. An independent final-evaluation API requires its
    own D4-D/D4-E contract before the B-TEST stage.
    """

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        if trainer is None:
            raise RuntimeError(
                "VGRA standalone validation not authorized in C5C; "
                "supply the governed trainer for B-VAL development evaluation"
            )
        # Stock BaseTrainer reports world_size=0 in non-distributed CPU mode.
        if trainer.world_size not in {0, 1} or RANK not in {-1, 0}:
            raise NotImplementedError("VGRA V1 DDP validation not supported")
        if bool(trainer.args.compile):
            raise RuntimeError("VGRA V1 compiled validation not supported")
        if self.dataloader is None:
            raise ValueError("VGRA validator requires a pair-aware dataloader")

        self.training = True
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != "cpu" and bool(trainer.amp)
        if model is None:
            model = trainer.ema.ema if trainer.ema is not None and trainer.ema.ema is not None else trainer.model
        model = unwrap_model(model)
        if not isinstance(model, VGRADetectionModel):
            raise TypeError("VGRA validator requires VGRADetectionModel or its EMA copy")
        if not bool(model.vgra_train_weights_ready.item()):
            raise RuntimeError("VGRA validator requires TRAIN-derived visibility weights bound on evaluation model")

        model = model.half() if self.args.half else model.float()
        model.eval()
        self.loss = torch.zeros_like(trainer.loss_items, device=self.device)
        self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)

        self.run_callbacks("on_val_start")
        self.init_metrics(model)
        self.jdict = []
        counters = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        for batch_i, batch in enumerate(TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))):
            self.batch_i = batch_i
            self.run_callbacks("on_val_batch_start")
            with counters[0]:
                batch = self.preprocess(batch)

            with counters[1]:
                raw = model.forward_vgra_batch(batch)
                # Apply the original Detect._inference geometry and sigmoid function.
                decoded = model.model[-1]._inference(raw)

            with counters[2]:
                # Loss is computed from the SAME raw paired predictions, never
                # from a second native single-image forward.
                loss_vector, display_loss = model.loss(batch, raw)
                if loss_vector.shape != (4,) or display_loss.shape != (4,):
                    raise RuntimeError("VGRA validator expected four-component native+visibility loss")
                self.loss += display_loss

            with counters[3]:
                detections = self.postprocess(decoded)

            self.update_metrics(detections, batch)
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
            raise RuntimeError("VGRA validation loader unexpectedly empty")
        losses = trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")
        return {key: round(float(value), 5) for key, value in {**stats, **losses}.items()}


__all__ = ("VGRADetectionValidator",)
