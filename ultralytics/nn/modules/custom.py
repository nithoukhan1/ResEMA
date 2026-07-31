"""Custom modules for the C3k2-DySample-ResEMA detector."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("DySample", "ResEMA")


def _normal_init(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    bias: float = 0.0,
) -> None:
    """Initialize weights normally and initialize bias to a constant."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(
            module.weight,
            mean=mean,
            std=std,
        )

    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(
            module.bias,
            bias,
        )


def _constant_init(
    module: nn.Module,
    value: float,
    bias: float = 0.0,
) -> None:
    """Initialize weights and bias to constants."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(
            module.weight,
            value,
        )

    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(
            module.bias,
            bias,
        )


class DySample(nn.Module):
    """
    Dynamic upsampling by learned point sampling.

    Adapted from the official implementation of:
    Learning to Upsample by Learning to Sample, ICCV 2023.
    """

    def __init__(
        self,
        c1: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()

        if scale < 1:
            raise ValueError(
                f"scale must be at least 1, but received {scale}."
            )

        if style not in {"lp", "pl"}:
            raise ValueError(
                f"style must be 'lp' or 'pl', but received {style!r}."
            )

        if groups < 1:
            raise ValueError(
                f"groups must be at least 1, but received {groups}."
            )

        if c1 % groups != 0:
            raise ValueError(
                f"Input channels ({c1}) must be divisible "
                f"by groups ({groups})."
            )

        if style == "pl":
            scale_squared = scale**2

            if c1 < scale_squared:
                raise ValueError(
                    f"For style='pl', input channels ({c1}) must be "
                    f"at least scale² ({scale_squared})."
                )

            if c1 % scale_squared != 0:
                raise ValueError(
                    f"For style='pl', input channels ({c1}) must be "
                    f"divisible by scale² ({scale_squared})."
                )

        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = dyscope

        if style == "pl":
            offset_input_channels = c1 // (scale**2)
            offset_output_channels = 2 * groups
        else:
            offset_input_channels = c1
            offset_output_channels = 2 * groups * (scale**2)

        self.offset = nn.Conv2d(
            offset_input_channels,
            offset_output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Reference DySample initialization.
        _normal_init(
            self.offset,
            std=0.001,
            bias=0.0,
        )

        if dyscope:
            self.scope = nn.Conv2d(
                offset_input_channels,
                offset_output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

            _constant_init(
                self.scope,
                value=0.0,
            )

        self.register_buffer(
            "init_pos",
            self._init_pos(),
        )

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
        ) / self.scale

        return (
            torch.stack(
                torch.meshgrid(
                    h,
                    h,
                    indexing="ij",
                )
            )
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = offset.shape

        offset = offset.view(
            batch_size,
            2,
            -1,
            height,
            width,
        )

        coords_h = (
            torch.arange(
                height,
                device=x.device,
                dtype=x.dtype,
            )
            + 0.5
        )

        coords_w = (
            torch.arange(
                width,
                device=x.device,
                dtype=x.dtype,
            )
            + 0.5
        )

        coords = (
            torch.stack(
                torch.meshgrid(
                    coords_w,
                    coords_h,
                    indexing="ij",
                )
            )
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )

        normalizer = torch.tensor(
            [width, height],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 2, 1, 1, 1)

        coords = (
            2.0 * (coords + offset) / normalizer
            - 1.0
        )

        coords = (
            F.pixel_shuffle(
                coords.view(
                    batch_size,
                    -1,
                    height,
                    width,
                ),
                self.scale,
            )
            .view(
                batch_size,
                2,
                -1,
                self.scale * height,
                self.scale * width,
            )
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        sampled = F.grid_sample(
            x.reshape(
                batch_size * self.groups,
                -1,
                height,
                width,
            ),
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        return sampled.view(
            batch_size,
            -1,
            self.scale * height,
            self.scale * width,
        )

    def forward_lp(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "scope"):
            offset = (
                self.offset(x)
                * self.scope(x).sigmoid()
                * 0.5
                + self.init_pos
            )
        else:
            offset = (
                self.offset(x) * 0.25
                + self.init_pos
            )

        return self.sample(
            x,
            offset,
        )

    def forward_pl(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_high_resolution = F.pixel_shuffle(
            x,
            self.scale,
        )

        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_high_resolution)
                    * self.scope(x_high_resolution).sigmoid(),
                    self.scale,
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_high_resolution),
                    self.scale,
                )
                * 0.25
                + self.init_pos
            )

        return self.sample(
            x,
            offset,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.style == "pl":
            return self.forward_pl(x)

        return self.forward_lp(x)


class ResEMA(nn.Module):
    """
    Residual Efficient Multi-Scale Attention.

    The module applies:
    1. preliminary channel transformation;
    2. grouped directional encoding;
    3. normalized and 3×3 feature branches;
    4. channel-softmax branch reweighting;
    5. an external residual connection.
    """

    def __init__(
        self,
        c1: int,
        groups: int = 8,
        reduction: int = 2,
    ) -> None:
        super().__init__()

        if groups < 1:
            raise ValueError(
                f"groups must be at least 1, but received {groups}."
            )

        if c1 % groups != 0:
            raise ValueError(
                f"Input channels ({c1}) must be divisible "
                f"by groups ({groups})."
            )

        if reduction < 1:
            raise ValueError(
                f"reduction must be at least 1, but received {reduction}."
            )

        self.groups = groups
        self.group_channels = c1 // groups
        self.reduction = reduction

        intermediate_channels = max(
            c1 // reduction,
            1,
        )

        self.pre_transform = nn.Sequential(
            nn.Conv2d(
                c1,
                intermediate_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                intermediate_channels,
                c1,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(c1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Preserves the normalization behavior of the original
        # ResEMA implementation while guaranteeing divisibility.
        normalization_groups = math.gcd(
            self.group_channels,
            32,
        )

        self.group_norm = nn.GroupNorm(
            normalization_groups,
            self.group_channels,
        )

        self.conv1x1 = nn.Conv2d(
            self.group_channels,
            self.group_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv3x3 = nn.Conv2d(
            self.group_channels,
            self.group_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        transformed = self.pre_transform(x)

        batch_size, channels, height, width = transformed.shape

        grouped = transformed.reshape(
            batch_size * self.groups,
            self.group_channels,
            height,
            width,
        )

        feature_h = self.pool_h(grouped)
        feature_w = self.pool_w(grouped).permute(
            0,
            1,
            3,
            2,
        )

        directional = self.conv1x1(
            torch.cat(
                (feature_h, feature_w),
                dim=2,
            )
        )

        feature_h, feature_w = torch.split(
            directional,
            (height, width),
            dim=2,
        )

        branch_1 = self.group_norm(
            grouped
            * feature_h.sigmoid()
            * feature_w.permute(0, 1, 3, 2).sigmoid()
        )

        branch_2 = self.conv3x3(grouped)

        weight_1 = self.softmax(
            self.global_pool(branch_1)
            .flatten(2)
            .transpose(1, 2)
        )

        weight_2 = self.softmax(
            self.global_pool(branch_2)
            .flatten(2)
            .transpose(1, 2)
        )

        weight_1 = (
            weight_1
            .transpose(1, 2)
            .reshape(
                batch_size * self.groups,
                self.group_channels,
                1,
                1,
            )
        )

        weight_2 = (
            weight_2
            .transpose(1, 2)
            .reshape(
                batch_size * self.groups,
                self.group_channels,
                1,
                1,
            )
        )

        output = (
            branch_1 * weight_1
            + branch_2 * weight_2
        ).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return residual + output