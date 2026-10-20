# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA class-visibility targets, TRAIN-only counts, and frozen weighted objective.

D4-C5A is a standalone objective-contract implementation. It does NOT
integrate model routing, native detection loss, trainer, or validator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ultralytics.nn.modules.vgra import (
    visibility_state_from_presence,
    visibility_state_weights,
)

VIEW_AP = 1
VIEW_LAT = 2
VISIBILITY_LOSS_WEIGHT_V1 = 0.25


def _validated_integer_vector(x: Tensor, name: str, upper: int | None = None) -> Tensor:
    """Fail closed on fractional, negative, nonfinite, or out-of-range indices."""
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tuple(x.shape)}")
    if x.is_floating_point():
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains nonfinite values")
        if not torch.all(x == x.round()):
            raise ValueError(f"{name} contains non-integer values")
    values = x.to(dtype=torch.long)
    if (values < 0).any() or (upper is not None and (values >= upper).any()):
        raise ValueError(f"{name} contains out-of-range values")
    return values


def canonical_pair_indices(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """Return each valid pair once, ordered (AP index, LAT index).

    Requires reciprocal companion positions, equal pair_id strings,
    AP/LAT view codes 1/2, and strict -1 sentinel for singles.
    """
    for field in ("vgra_pair_index", "vgra_pair_valid", "vgra_view_code", "vgra_pair_id"):
        if field not in batch:
            raise KeyError(f"VGRA metadata missing: {field}")

    partner = torch.as_tensor(batch["vgra_pair_index"])
    valid = torch.as_tensor(batch["vgra_pair_valid"])
    code = torch.as_tensor(batch["vgra_view_code"])
    pair_ids = batch["vgra_pair_id"]

    if partner.ndim != 1 or valid.ndim != 1 or code.ndim != 1:
        raise ValueError("VGRA partner, valid and view-code tensors must be 1D")
    n = partner.numel()
    if len(pair_ids) != n or valid.numel() != n or code.numel() != n:
        raise ValueError("VGRA metadata batch-length mismatch")
    if valid.dtype != torch.bool:
        raise TypeError("vgra_pair_valid must be Boolean")

    if partner.is_floating_point():
        if not torch.isfinite(partner).all() or not torch.all(partner == partner.round()):
            raise ValueError("vgra_pair_index must contain finite integer values")
    partner = partner.to(torch.long)

    if code.is_floating_point():
        if not torch.isfinite(code).all() or not torch.all(code == code.round()):
            raise ValueError("vgra_view_code must contain finite integer values")
    code = code.to(torch.long)
    if not torch.all((code >= 0) & (code <= 3)):
        raise ValueError("vgra_view_code outside supported range")

    ap_indices: list[int] = []
    lat_indices: list[int] = []
    observed_pair_ids: set[str] = set()

    for i in range(n):
        p = int(partner[i])
        is_pair = bool(valid[i])
        if not is_pair:
            if p != -1:
                raise ValueError(f"single index {i} has a companion {p}")
            continue
        if p == i or p < 0 or p >= n:
            raise ValueError(f"invalid companion index {p} for image {i}")
        if not bool(valid[p]) or int(partner[p]) != i:
            raise ValueError("VGRA companion relation must be reciprocal and valid")
        pid = str(pair_ids[i])
        if not pid or pid != str(pair_ids[p]):
            raise ValueError("VGRA paired members must share nonempty pair_id")
        c1, c2 = int(code[i]), int(code[p])
        if {c1, c2} != {VIEW_AP, VIEW_LAT}:
            raise ValueError(f"VGRA pair must be AP/LAT; got {c1}/{c2}")
        if c1 == VIEW_AP:
            if pid in observed_pair_ids:
                raise ValueError(f"duplicate VGRA pair_id in batch: {pid}")
            observed_pair_ids.add(pid)
            ap_indices.append(i)
            lat_indices.append(p)

    return (torch.tensor(ap_indices, device=partner.device, dtype=torch.long),
            torch.tensor(lat_indices, device=partner.device, dtype=torch.long))


def class_presence_from_collated_labels(batch: Mapping[str, Any], nc: int) -> Tensor:
    """Derive per-image class presence from collated YOLO GT classes only."""
    if nc <= 0:
        raise ValueError("nc must be positive")
    images = batch.get("img")
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise ValueError("batch['img'] must be a BCHW tensor")
    bs = images.shape[0]

    for name in ("cls", "batch_idx"):
        if name not in batch:
            raise KeyError(f"missing YOLO GT field: {name}")
    cls = torch.as_tensor(batch["cls"], device=images.device).reshape(-1)
    idx = torch.as_tensor(batch["batch_idx"], device=images.device).reshape(-1)
    if cls.numel() != idx.numel():
        raise ValueError("YOLO classes and batch_idx length mismatch")

    classes = _validated_integer_vector(cls, "class indices", nc)
    batch_indices = _validated_integer_vector(idx, "batch indices", bs)
    presence = torch.zeros((bs, nc), dtype=torch.bool, device=images.device)
    if classes.numel():
        presence[batch_indices, classes] = True
    return presence


def visibility_targets_from_batch(batch: Mapping[str, Any], nc: int) -> tuple[Tensor, Tensor, Tensor]:
    """Create 00/10/01/11 class targets for valid pairs and their AP/LAT positions."""
    ap, lat = canonical_pair_indices(batch)
    presence = class_presence_from_collated_labels(batch, nc)
    ap = ap.to(presence.device)
    lat = lat.to(presence.device)
    target = visibility_state_from_presence(presence[ap], presence[lat])
    return target, ap, lat


def visibility_weighted_ce(logits: Tensor, targets: Tensor, state_weights: Tensor) -> Tensor:
    """Frozen L_vis = mean_pair,class(w[c,target] * CE(logits, target)).

    No loss is produced for empty pair sets. Callers must independently handle
    unpaired detection loss, without using missing-view synthetic visibility labels.
    """
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError(f"logits must be (pairs,nc,4), got {tuple(logits.shape)}")
    n, nc, _ = logits.shape
    if targets.shape != (n, nc):
        raise ValueError("visibility target shape mismatch")
    if state_weights.shape != (nc, 4):
        raise ValueError("state weights must have shape (nc,4)")
    if not torch.isfinite(logits).all() or not torch.isfinite(state_weights).all():
        raise ValueError("nonfinite visibility logits or weights")
    if (state_weights <= 0).any():
        raise ValueError("state weights must be positive")
    t = _validated_integer_vector(targets.reshape(-1), "visibility targets", 4)
    if not n:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits.reshape(-1, 4), t, reduction="none").view(n, nc)
    weights = state_weights.to(device=logits.device, dtype=logits.dtype)
    selected = weights[torch.arange(nc, device=logits.device)[None, :], t.view(n, nc)]
    return (selected * ce).mean()


def train_state_counts_from_dataset(dataset: Any, nc: int, *, split: str) -> Tensor:
    """Count 00/10/01/11 from in-memory TRAIN label records (never VAL/TEST).

    Requires the VGRAYOLODataset in-memory .labels, .vgra_metadata and .vgra_units.
    Does not open image or annotation files itself. The dataset constructor may load
    annotations when the separately authorized C5B/C5C pipeline is used.
    """
    if split != "train" or getattr(dataset, "vgra_split", None) != "train":
        raise ValueError("state-weight count source must be exactly B-TRAIN")
    if nc <= 0:
        raise ValueError("nc must be positive")
    labels = dataset.labels
    metadata = dataset.vgra_metadata
    units = dataset.vgra_units
    if len(labels) != len(metadata):
        raise ValueError("dataset labels/metadata cardinality mismatch")

    counts = torch.zeros((nc, 4), dtype=torch.long)
    paired_count = 0
    for unit in units:
        if len(unit) == 1:
            continue
        if len(unit) != 2:
            raise ValueError("pairing unit must contain one or two images")
        a, b = unit
        if metadata[a]["view_code"] != "AP" or metadata[b]["view_code"] != "LAT":
            raise ValueError("pair units must be ordered AP then LAT")
        if not (metadata[a]["pair_valid"] and metadata[b]["pair_valid"]):
            raise ValueError("paired dataset unit has invalid member")
        if metadata[a]["pair_id"] != metadata[b]["pair_id"]:
            raise ValueError("pair ID mismatch")

        pres = []
        for i in (a, b):
            raw = torch.as_tensor(labels[i]["cls"]).reshape(-1)
            cl = _validated_integer_vector(raw, "dataset YOLO classes", nc)
            present = torch.zeros((nc,), dtype=torch.bool)
            if cl.numel():
                present[cl] = True
            pres.append(present)
        targets = visibility_state_from_presence(pres[0], pres[1])
        counts[torch.arange(nc), targets] += 1
        paired_count += 1

    if paired_count == 0 or not torch.all(counts.sum(dim=1) == paired_count):
        raise ValueError("TRAIN pair-state accounting mismatch")
    return counts


def train_state_weight_binding(dataset: Any, nc: int, *, split: str) -> tuple[Tensor, Tensor]:
    """Return TRAIN-derived state counts and frozen-normalized positive weights."""
    counts = train_state_counts_from_dataset(dataset, nc, split=split)
    return counts, visibility_state_weights(counts)


__all__ = (
    "VIEW_AP", "VIEW_LAT", "VISIBILITY_LOSS_WEIGHT_V1",
    "canonical_pair_indices", "class_presence_from_collated_labels",
    "visibility_targets_from_batch", "visibility_weighted_ce",
    "train_state_counts_from_dataset", "train_state_weight_binding",
)
