# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA-specific detection head.

D4-C3 scope:
- subclass stock Detect;
- keep stock single-view forward/fallback unchanged;
- expose an explicit paired raw-head path;
- apply VGRA only to classification logits;
- never modify box/DFL tensors with cross-view information.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .head import Detect
from .vgra import VGRA


class VGRADetect(Detect):
    """Detect subclass with explicit paired-view VGRA classification residual support.

    The inherited ``forward`` method remains the ordinary stock Detect path. This is the
    exact single-view fallback.

    Pair-aware execution is explicit through ``forward_pair_heads`` and returns raw
    prediction dictionaries for AP and LAT. Trainer/validator integration is deliberately
    deferred to D4-C5.
    """

    def __init__(
        self,
        nc: int = 80,
        descriptor_rank: int = 32,
        pair_hidden_dim: int = 128,
        cross_view_rank: int = 16,
        beta_max: float = 2.0,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.vgra = VGRA(
            in_channels=ch,
            nc=nc,
            descriptor_rank=descriptor_rank,
            pair_hidden_dim=pair_hidden_dim,
            cross_view_rank=cross_view_rank,
            beta_max=beta_max,
        )

    def forward_head_with_residual(
        self,
        x: list[torch.Tensor],
        residuals: Sequence[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        """Return native boxes and classification logits plus VGRA residuals."""
        if box_head is None or cls_head is None:
            return {}
        if len(x) != self.nl or len(residuals) != self.nl:
            raise ValueError(
                f"expected {self.nl} feature/residual levels, got x={len(x)} residuals={len(residuals)}"
            )

        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )

        score_levels = []
        for i in range(self.nl):
            native = cls_head[i](x[i])
            residual = residuals[i]
            if residual.shape != native.shape:
                raise ValueError(
                    f"VGRA residual shape mismatch at level {i}: "
                    f"native={tuple(native.shape)} residual={tuple(residual.shape)}"
                )
            score_levels.append((native + residual).view(bs, self.nc, -1))

        scores = torch.cat(score_levels, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward_pair_heads(
        self,
        ap_features: list[torch.Tensor],
        lat_features: list[torch.Tensor],
    ) -> dict[str, object]:
        """Return paired AP/LAT raw predictions with VGRA applied only to class logits."""
        if self.end2end:
            raise RuntimeError("VGRA V1 paired path does not support end2end/one-to-one Detect mode")
        if len(ap_features) != self.nl or len(lat_features) != self.nl:
            raise ValueError(
                f"expected {self.nl} AP/LAT levels, got AP={len(ap_features)} LAT={len(lat_features)}"
            )
        if ap_features[0].shape[0] != lat_features[0].shape[0]:
            raise ValueError("AP and LAT paired feature batches must have equal batch size")

        context = self.vgra.forward_pair(ap_features, lat_features)
        ap_preds = self.forward_head_with_residual(
            ap_features,
            context["ap_residuals"],
            box_head=self.cv2,
            cls_head=self.cv3,
        )
        lat_preds = self.forward_head_with_residual(
            lat_features,
            context["lat_residuals"],
            box_head=self.cv2,
            cls_head=self.cv3,
        )
        return {"ap": ap_preds, "lat": lat_preds, "vgra": context}


__all__ = ("VGRADetect",)
