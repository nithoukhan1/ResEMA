# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# D4-C1 VGRA core mathematical primitives.
#
# This file intentionally contains no dataset, trainer, validator, or Detect-head
# integration. It implements only the frozen D4-B2C2 candidate-V1 mathematics.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


VGRA_NUM_VISIBILITY_STATES = 4
VGRA_STATE_NONE = 0
VGRA_STATE_AP_ONLY = 1
VGRA_STATE_LAT_ONLY = 2
VGRA_STATE_BOTH = 3


def visibility_state_from_presence(ap_present: Tensor, lat_present: Tensor) -> Tensor:
    """Map per-class AP/LAT presence indicators to frozen VGRA state indices.

    Args:
        ap_present: Boolean or 0/1 tensor of shape (..., nc).
        lat_present: Boolean or 0/1 tensor with the same shape.

    Returns:
        Long tensor with state indices:
        0=neither, 1=AP only, 2=LAT only, 3=both.
    """
    if ap_present.shape != lat_present.shape:
        raise ValueError(
            f"AP/LAT presence shapes must match, got {tuple(ap_present.shape)} and {tuple(lat_present.shape)}"
        )
    ap = ap_present.to(dtype=torch.long)
    lat = lat_present.to(dtype=torch.long)
    if not torch.all((ap == 0) | (ap == 1)):
        raise ValueError("ap_present must contain only boolean/0/1 values")
    if not torch.all((lat == 0) | (lat == 1)):
        raise ValueError("lat_present must contain only boolean/0/1 values")
    return ap + 2 * lat


def visibility_state_weights(counts: Tensor) -> Tensor:
    """Compute frozen per-class four-state weights and normalize them to mean one."""
    if counts.ndim != 2 or counts.shape[-1] != VGRA_NUM_VISIBILITY_STATES:
        raise ValueError(f"counts must have shape (nc, 4), got {tuple(counts.shape)}")
    dtype = counts.dtype if counts.is_floating_point() else torch.float32
    x = counts.to(dtype=dtype)
    if torch.any(x < 0):
        raise ValueError("visibility state counts must be non-negative")
    raw = torch.rsqrt(x + 1.0)
    return VGRA_NUM_VISIBILITY_STATES * raw / raw.sum(dim=-1, keepdim=True)


def visibility_coefficients(probabilities: Tensor, *, detach: bool = True) -> tuple[Tensor, Tensor]:
    """Return frozen target-specific AP and LAT visibility coefficients."""
    if probabilities.ndim < 2 or probabilities.shape[-1] != VGRA_NUM_VISIBILITY_STATES:
        raise ValueError(f"probabilities must end in four visibility states, got {tuple(probabilities.shape)}")
    q = probabilities.detach() if detach else probabilities
    k_ap = q[..., VGRA_STATE_BOTH] - q[..., VGRA_STATE_LAT_ONLY]
    k_lat = q[..., VGRA_STATE_BOTH] - q[..., VGRA_STATE_AP_ONLY]
    return k_ap, k_lat


class VGRAViewDescriptor(nn.Module):
    """Create a shared per-view semantic descriptor from Detect-input feature levels."""

    def __init__(self, in_channels: Sequence[int], descriptor_rank: int = 32):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature level")
        if descriptor_rank <= 0:
            raise ValueError("descriptor_rank must be positive")
        self.in_channels = tuple(int(c) for c in in_channels)
        if any(c <= 0 for c in self.in_channels):
            raise ValueError("all in_channels must be positive")
        self.descriptor_rank = int(descriptor_rank)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(c, self.descriptor_rank),
                nn.LayerNorm(self.descriptor_rank),
                nn.SiLU(),
            )
            for c in self.in_channels
        )
        self.output_dim = len(self.in_channels) * self.descriptor_rank

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.projections):
            raise ValueError(f"expected {len(self.projections)} feature levels, got {len(features)}")
        outputs = []
        batch = None
        for level, (x, expected_c, proj) in enumerate(zip(features, self.in_channels, self.projections)):
            if x.ndim != 4:
                raise ValueError(f"feature level {level} must be BCHW, got shape {tuple(x.shape)}")
            if x.shape[1] != expected_c:
                raise ValueError(
                    f"feature level {level} channel mismatch: expected {expected_c}, got {x.shape[1]}"
                )
            if batch is None:
                batch = x.shape[0]
            elif x.shape[0] != batch:
                raise ValueError("all feature levels must share the same batch size")
            pooled = x.mean(dim=(-2, -1))
            outputs.append(proj(pooled))
        return torch.cat(outputs, dim=-1)


class VGRAPairVisibilityPredictor(nn.Module):
    """Predict class-wise four-state AP/LAT visibility from ordered paired descriptors."""

    def __init__(self, view_dim: int, nc: int, hidden_dim: int = 128):
        super().__init__()
        if view_dim <= 0 or nc <= 0 or hidden_dim <= 0:
            raise ValueError("view_dim, nc and hidden_dim must be positive")
        self.view_dim = int(view_dim)
        self.nc = int(nc)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(4 * self.view_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.nc * VGRA_NUM_VISIBILITY_STATES)

    def pair_context(self, h_ap: Tensor, h_lat: Tensor) -> Tensor:
        if h_ap.shape != h_lat.shape:
            raise ValueError(f"AP/LAT descriptor shapes must match, got {tuple(h_ap.shape)} and {tuple(h_lat.shape)}")
        if h_ap.ndim != 2 or h_ap.shape[-1] != self.view_dim:
            raise ValueError(f"descriptors must have shape (B, {self.view_dim}), got {tuple(h_ap.shape)}")
        return torch.cat((h_ap, h_lat, (h_ap - h_lat).abs(), h_ap * h_lat), dim=-1)

    def forward(self, h_ap: Tensor, h_lat: Tensor) -> Tensor:
        u = self.pair_context(h_ap, h_lat)
        p = self.act(self.norm(self.fc1(u)))
        return self.fc2(p).reshape(-1, self.nc, VGRA_NUM_VISIBILITY_STATES)

    @staticmethod
    def probabilities(logits: Tensor) -> Tensor:
        if logits.ndim != 3 or logits.shape[-1] != VGRA_NUM_VISIBILITY_STATES:
            raise ValueError(f"visibility logits must have shape (B, nc, 4), got {tuple(logits.shape)}")
        return logits.softmax(dim=-1)


class VGRALevelResidual(nn.Module):
    """Low-rank class-conditioned cross-view residual for one Detect feature level."""

    def __init__(
        self,
        in_channels: int,
        companion_dim: int,
        nc: int,
        rank: int = 16,
        beta_max: float = 2.0,
    ):
        super().__init__()
        if min(in_channels, companion_dim, nc, rank) <= 0:
            raise ValueError("in_channels, companion_dim, nc and rank must be positive")
        if beta_max <= 0:
            raise ValueError("beta_max must be positive")
        self.in_channels = int(in_channels)
        self.companion_dim = int(companion_dim)
        self.nc = int(nc)
        self.rank = int(rank)
        self.beta_max = float(beta_max)

        self.local_projection = nn.Conv2d(self.in_channels, self.rank, kernel_size=1, bias=False)
        self.companion_projection = nn.Linear(self.companion_dim, self.rank)
        self.class_embedding = nn.Parameter(torch.empty(self.nc, self.rank))
        self.rho = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.class_embedding, mean=0.0, std=0.02)

    @property
    def residual_strength(self) -> Tensor:
        """Return beta_l = beta_max * tanh(rho_l)."""
        return self.beta_max * torch.tanh(self.rho)

    def compatibility(self, target_feature: Tensor, companion_descriptor: Tensor) -> Tensor:
        """Return bounded compatibility m with shape (B, nc, H, W)."""
        if target_feature.ndim != 4 or target_feature.shape[1] != self.in_channels:
            raise ValueError(
                f"target_feature must have shape (B, {self.in_channels}, H, W), got {tuple(target_feature.shape)}"
            )
        if companion_descriptor.ndim != 2 or companion_descriptor.shape[-1] != self.companion_dim:
            raise ValueError(
                f"companion_descriptor must have shape (B, {self.companion_dim}), "
                f"got {tuple(companion_descriptor.shape)}"
            )
        if target_feature.shape[0] != companion_descriptor.shape[0]:
            raise ValueError("target feature and companion descriptor batch sizes must match")

        z = F.normalize(self.local_projection(target_feature), dim=1, eps=1e-6)
        g = F.normalize(self.companion_projection(companion_descriptor), dim=-1, eps=1e-6)
        e = F.normalize(self.class_embedding, dim=-1, eps=1e-6)
        t = F.normalize(g[:, None, :] * e[None, :, :], dim=-1, eps=1e-6)
        m = torch.einsum("brhw,bcr->bchw", z, t).relu()
        return m.clamp(max=1.0)

    def forward(self, target_feature: Tensor, companion_descriptor: Tensor, coefficient: Tensor) -> Tensor:
        if coefficient.ndim != 2 or coefficient.shape[-1] != self.nc:
            raise ValueError(f"coefficient must have shape (B, {self.nc}), got {tuple(coefficient.shape)}")
        if coefficient.shape[0] != target_feature.shape[0]:
            raise ValueError("coefficient and target feature batch sizes must match")
        m = self.compatibility(target_feature, companion_descriptor)
        return self.residual_strength * coefficient[:, :, None, None] * m


class VGRAResidualAssistant(nn.Module):
    """Apply shared V1 residual mathematics across all Detect input feature levels."""

    def __init__(
        self,
        in_channels: Sequence[int],
        companion_dim: int,
        nc: int,
        rank: int = 16,
        beta_max: float = 2.0,
    ):
        super().__init__()
        self.in_channels = tuple(int(c) for c in in_channels)
        self.levels = nn.ModuleList(
            VGRALevelResidual(c, companion_dim, nc, rank=rank, beta_max=beta_max)
            for c in self.in_channels
        )

    def forward(
        self,
        target_features: Sequence[Tensor],
        companion_descriptor: Tensor,
        coefficient: Tensor,
    ) -> list[Tensor]:
        if len(target_features) != len(self.levels):
            raise ValueError(f"expected {len(self.levels)} feature levels, got {len(target_features)}")
        return [
            module(feature, companion_descriptor, coefficient)
            for module, feature in zip(self.levels, target_features)
        ]


class VGRA(nn.Module):
    """Standalone frozen VGRA V1 mathematical core for paired AP/LAT features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        nc: int,
        descriptor_rank: int = 32,
        pair_hidden_dim: int = 128,
        cross_view_rank: int = 16,
        beta_max: float = 2.0,
    ):
        super().__init__()
        self.in_channels = tuple(int(c) for c in in_channels)
        self.nc = int(nc)
        self.descriptor = VGRAViewDescriptor(self.in_channels, descriptor_rank=descriptor_rank)
        self.visibility = VGRAPairVisibilityPredictor(
            view_dim=self.descriptor.output_dim,
            nc=self.nc,
            hidden_dim=pair_hidden_dim,
        )
        self.residual = VGRAResidualAssistant(
            self.in_channels,
            companion_dim=self.descriptor.output_dim,
            nc=self.nc,
            rank=cross_view_rank,
            beta_max=beta_max,
        )

    def forward_pair(self, ap_features: Sequence[Tensor], lat_features: Sequence[Tensor]) -> dict[str, object]:
        """Compute paired visibility and AP/LAT residuals without changing detector tensors."""
        h_ap = self.descriptor(ap_features)
        h_lat = self.descriptor(lat_features)
        visibility_logits = self.visibility(h_ap, h_lat)
        visibility_probabilities = self.visibility.probabilities(visibility_logits)
        k_ap, k_lat = visibility_coefficients(visibility_probabilities, detach=True)

        ap_residuals = self.residual(ap_features, h_lat, k_ap)
        lat_residuals = self.residual(lat_features, h_ap, k_lat)

        return {
            "ap_descriptor": h_ap,
            "lat_descriptor": h_lat,
            "visibility_logits": visibility_logits,
            "visibility_probabilities": visibility_probabilities,
            "k_ap": k_ap,
            "k_lat": k_lat,
            "ap_residuals": ap_residuals,
            "lat_residuals": lat_residuals,
        }


__all__ = (
    "VGRA",
    "VGRAViewDescriptor",
    "VGRAPairVisibilityPredictor",
    "VGRALevelResidual",
    "VGRAResidualAssistant",
    "visibility_state_from_presence",
    "visibility_state_weights",
    "visibility_coefficients",
    "VGRA_NUM_VISIBILITY_STATES",
    "VGRA_STATE_NONE",
    "VGRA_STATE_AP_ONLY",
    "VGRA_STATE_LAT_ONLY",
    "VGRA_STATE_BOTH",
)
