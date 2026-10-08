# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA V1 mixed paired/single-image raw runtime and loss integration (D4-C5B).

The entire batch is processed through the shared YOLO backbone and native heads ONCE.
Only the subset with valid reciprocal AP/LAT pairing receives VGRA class residuals.
The native box/DFL tensors and all single-view class logits remain unchanged.

The trainer and validator integrations are deliberately deferred to D4-C5C.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.vgra_head import VGRADetect
from ultralytics.utils.loss import v8DetectionLoss
from .vgra_targets import (
    VISIBILITY_LOSS_WEIGHT_V1,
    canonical_pair_indices,
    visibility_targets_from_batch,
    visibility_weighted_ce,
)


class VGRADetectionModel(DetectionModel):
    """A detection model with an explicit VGRA training/validation raw-prediction path.

    Default .predict(Tensor) stays inherited single-view YOLO/Detect behavior.
    The paired path is available only via .forward_vgra_batch(batch), which
    requires the governed C4 metadata.
    """

    def __init__(self, cfg="yolo11s-tpsc-early-vgra-v1.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if not isinstance(self.model[-1], VGRADetect):
            raise TypeError("VGRADetectionModel requires VGRADetect as its final model layer")
        head = self.model[-1]
        if head.end2end:
            raise RuntimeError("VGRA V1 only supports native one-to-many detection")

        # Buffers persist with the model/checkpoint and do not count as trainable parameters.
        self.register_buffer("vgra_visibility_weights", torch.zeros(head.nc, 4, dtype=torch.float32))
        self.register_buffer("vgra_train_weights_ready", torch.tensor(False, dtype=torch.bool))

    def bind_train_visibility_weights(self, weights: Tensor, *, source: str) -> None:
        """Explicitly bind frozen TRAIN-only class-state weights; reject VAL/TEST sources."""
        if source != "B-TRAIN":
            raise ValueError("VGRA visibility weights must derive from B-TRAIN only")
        if not isinstance(weights, Tensor):
            raise TypeError("visibility weights must be a tensor")
        head: VGRADetect = self.model[-1]
        if weights.shape != (head.nc, 4):
            raise ValueError(f"visibility weights must have shape ({head.nc},4)")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("visibility weights must be finite and strictly positive")
        if not torch.allclose(
            weights.float().mean(dim=-1),
            torch.ones(head.nc, device=weights.device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("visibility weights must be normalized to per-class mean 1")
        self.vgra_visibility_weights.copy_(weights.detach().to(
            device=self.vgra_visibility_weights.device,
            dtype=self.vgra_visibility_weights.dtype,
        ))
        self.vgra_train_weights_ready.fill_(True)

    def _shared_head_features(self, images: Tensor) -> list[Tensor]:
        """Reproduce the native BaseModel `_predict_once` routing through all pre-head layers.

        Crucially, there is ONE backbone/neck forward pass over the entire input
        batch, so BatchNorm statistics match native training behavior.
        """
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("VGRA images must be BCHW tensor")
        x: Tensor | list[Tensor] = images
        saved: list[Any] = []
        for layer in self.model[:-1]:
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if source == -1 else saved[source] for source in layer.f]
                )
            x = layer(x)
            saved.append(x if layer.i in self.save else None)

        head = self.model[-1]
        sources = head.f if isinstance(head.f, (list, tuple)) else [head.f]
        feats = [x if source == -1 else saved[source] for source in sources]
        if len(feats) != head.nl or any(not isinstance(t, Tensor) for t in feats):
            raise RuntimeError("failed to obtain native P3/P4/P5 Detect-input features")
        if any(t.shape[0] != images.shape[0] for t in feats):
            raise RuntimeError("Detect feature batch size drift")
        return feats

    def forward_vgra_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Raw mixed paired/single detections in original image index order."""
        if not isinstance(batch, dict) or "img" not in batch:
            raise ValueError("VGRA batch must be a dict containing img")
        images = batch["img"]
        ap, lat = canonical_pair_indices(batch)
        if images.ndim != 4 or len(batch["vgra_pair_index"]) != images.shape[0]:
            raise ValueError("VGRA companion map must match the image batch")

        feats = self._shared_head_features(images)
        head: VGRADetect = self.model[-1]
        if head.cv2 is None or head.cv3 is None:
            raise RuntimeError("VGRA paired raw runtime requires unfused native Detect heads")

        # Native box and class heads run once over the entire batch.
        native = head.forward_head(feats, **head.one2many)
        if not native:
            raise RuntimeError("native head returned no predictions")
        scores = native["scores"]

        ap = ap.to(device=images.device)
        lat = lat.to(device=images.device)
        if ap.numel():
            ap_feat = [level.index_select(0, ap) for level in feats]
            lat_feat = [level.index_select(0, lat) for level in feats]
            context = head.vgra.forward_pair(ap_feat, lat_feat)

            ap_correction = torch.cat(
                [r.flatten(start_dim=2) for r in context["ap_residuals"]], dim=-1
            )
            lat_correction = torch.cat(
                [r.flatten(start_dim=2) for r in context["lat_residuals"]], dim=-1
            )
            if ap_correction.shape != (len(ap), head.nc, scores.shape[-1]):
                raise RuntimeError("AP residual/classification shape mismatch")
            if lat_correction.shape != (len(lat), head.nc, scores.shape[-1]):
                raise RuntimeError("LAT residual/classification shape mismatch")

            # Out-of-place index_add maintains autograd and original image order.
            delta = torch.zeros_like(scores)
            delta = delta.index_add(0, ap, ap_correction)
            delta = delta.index_add(0, lat, lat_correction)
            adjusted_scores = scores + delta
            visibility_logits = context["visibility_logits"]
        else:
            # Exact native passthrough with no needless zero-add operation.
            adjusted_scores = scores
            visibility_logits = scores.new_empty((0, head.nc, 4))

        return {
            "boxes": native["boxes"],
            "scores": adjusted_scores,
            "feats": native["feats"],
            "vgra_visibility_logits": visibility_logits,
            "vgra_ap_indices": ap,
            "vgra_lat_indices": lat,
        }

    def loss(self, batch: dict[str, Any], preds: dict[str, Any] | None = None):
        """Run VGRA raw model and loss for the supplied governed training batch."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward_vgra_batch(batch)
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Create additive VGRA criterion around unchanged native detection loss."""
        return VGRADetectionCriterion(self)


class VGRADetectionCriterion(v8DetectionLoss):
    """Native YOLO detection loss plus pair-proportional frozen visibility CE."""

    def __init__(self, model: VGRADetectionModel):
        super().__init__(model)
        self.vgra_model = model

    def __call__(self, preds: dict[str, Any], batch: dict[str, Any]):
        if not isinstance(preds, dict):
            raise TypeError("VGRA loss requires a raw prediction dictionary")
        if "vgra_visibility_logits" not in preds:
            raise KeyError("VGRA raw predictions missing visibility logits")

        # One native detection loss computation for every image, paired and single.
        native_total, native_items = super().__call__(preds, batch)
        if native_items.shape != (3,):
            raise RuntimeError("unexpected native detection loss item shape")

        target, ap, lat = visibility_targets_from_batch(batch, self.nc)
        logits = preds["vgra_visibility_logits"]
        if logits.shape != (target.shape[0], self.nc, 4):
            raise ValueError("VGRA visibility logits/GT cardinality mismatch")
        if not torch.equal(preds["vgra_ap_indices"].to(ap.device), ap):
            raise ValueError("VGRA model/loss AP index mismatch")
        if not torch.equal(preds["vgra_lat_indices"].to(lat.device), lat):
            raise ValueError("VGRA model/loss LAT index mismatch")

        bs = int(batch["img"].shape[0])
        pair_images = int(2 * target.shape[0])

        if pair_images:
            if not bool(self.vgra_model.vgra_train_weights_ready.item()):
                raise RuntimeError("TRAIN-only VGRA visibility weights not bound")
            vis_mean = visibility_weighted_ce(
                logits, target.to(logits.device), self.vgra_model.vgra_visibility_weights
            )
            # Native v8DetectionLoss returns detection sum scaled by batch size.
            # Multiply the mean(pair,class) visibility CE by 2*number_of_pairs
            # so unpaired examples do not inflate its contribution.
            vis_term = VISIBILITY_LOSS_WEIGHT_V1 * pair_images * vis_mean
        else:
            vis_term = native_total.new_zeros(())

        # This repository's native criterion returns THREE weighted components,
        # each already multiplied by batch size. Its BaseTrainer explicitly
        # applies loss.sum() before backward. Preserve that exact convention:
        # return [box, cls, dfl, VGRA] rather than reducing to a scalar here.
        if native_total.shape != (3,):
            raise RuntimeError("unexpected native detection loss vector shape")
        loss_vector = torch.cat((native_total, vis_term.reshape(1)))
        display_vis = (vis_term / bs).detach()
        loss_items = torch.cat((native_items, display_vis.reshape(1)))
        if not torch.isfinite(loss_vector).all() or not torch.isfinite(loss_items).all():
            raise FloatingPointError("non-finite VGRA mixed-batch objective")
        return loss_vector, loss_items


__all__ = ("VGRADetectionModel", "VGRADetectionCriterion")
