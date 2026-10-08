# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA pair-aware dataset and batching utilities.

D4-C4 scope:
- bind runtime images to the frozen D4-C2 assignment manifest;
- preserve valid AP/LAT pairs as indivisible batch units;
- retain single-view samples;
- disable cross-study composition augmentations;
- emit explicit companion/view metadata after collation.

No trainer, criterion, validator, model-loss, or B-test behavior is implemented here.
"""

from __future__ import annotations

import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import RANK


VGRA_VIEW_SINGLE = 0
VGRA_VIEW_AP = 1
VGRA_VIEW_LAT = 2
VGRA_VIEW_OTHER = 3

PAIR_SAFE_DISABLED_AUGMENTATIONS = ("mosaic", "mixup", "cutmix", "copy_paste")


def _view_code_id(view_code: str) -> int:
    code = str(view_code).strip().upper()
    if code == "AP":
        return VGRA_VIEW_AP
    if code == "LAT":
        return VGRA_VIEW_LAT
    if code == "OTHER":
        return VGRA_VIEW_OTHER
    return VGRA_VIEW_SINGLE


def pair_safe_hyp(hyp: Any) -> Any:
    """Return a copy of hyperparameters with cross-study composition disabled."""
    safe = deepcopy(hyp)
    for name in PAIR_SAFE_DISABLED_AUGMENTATIONS:
        if isinstance(safe, dict):
            if name in safe:
                safe[name] = 0.0
        elif hasattr(safe, name):
            setattr(safe, name, 0.0)
    return safe


def load_assignment_rows(path: str | Path, split: str) -> list[dict[str, str]]:
    """Load frozen D4-C2 image assignments for one split."""
    split = str(split).strip().lower()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"].strip().lower() == split]
    if not rows:
        raise ValueError(f"no assignment rows found for split={split!r} in {path}")
    stems = [r["filestem"].strip() for r in rows]
    if len(stems) != len(set(stems)):
        raise ValueError(f"duplicate assignment filestems for split={split!r}")
    return rows


def _runtime_pair_valid(row: dict[str, str], split: str) -> bool:
    split = split.strip().lower()
    structural = row["structural_role"].strip().upper()
    operational = row["operational_role"].strip().upper()
    if structural != "PAIRED":
        return False
    if split == "train":
        return operational == "PAIR_CANDIDATE"
    if split == "val":
        return operational == "PAIRED"
    raise ValueError(f"unsupported VGRA split: {split!r}")


def bind_vgra_assignments(
    im_files: Iterable[str | Path],
    assignment_rows: list[dict[str, str]],
    split: str,
) -> tuple[list[dict[str, Any]], list[tuple[int, ...]]]:
    """Bind current dataset image order to frozen assignment rows and build indivisible units."""
    split = str(split).strip().lower()
    row_by_stem = {r["filestem"].strip(): r for r in assignment_rows}
    files = list(im_files)
    stems = [Path(str(p)).stem for p in files]

    if len(stems) != len(set(stems)):
        raise ValueError("runtime dataset has duplicate filestems")
    missing = sorted(set(stems) - set(row_by_stem))
    if missing:
        raise ValueError(f"runtime dataset contains filestems absent from C2 assignments: {missing[:10]}")

    metadata: list[dict[str, Any]] = []
    pair_members: dict[str, list[int]] = {}
    singles: list[int] = []

    for index, stem in enumerate(stems):
        row = row_by_stem[stem]
        valid = _runtime_pair_valid(row, split)
        pair_id = row["pair_id"].strip()
        view_code = row["view_code"].strip().upper()

        meta = {
            "index": index,
            "filestem": stem,
            "pair_id": pair_id,
            "pair_valid": valid,
            "view_code": view_code,
            "view_code_id": _view_code_id(view_code),
            "structural_role": row["structural_role"].strip(),
            "operational_role": row["operational_role"].strip(),
        }
        metadata.append(meta)

        if valid:
            if not pair_id:
                raise ValueError(f"valid pair member {stem} has empty pair_id")
            pair_members.setdefault(pair_id, []).append(index)
        else:
            singles.append(index)

    units: list[tuple[int, ...]] = []
    for pair_id in sorted(pair_members):
        indices = pair_members[pair_id]
        if len(indices) != 2:
            raise ValueError(
                f"runtime pair {pair_id} has {len(indices)} available valid members; expected exactly 2"
            )
        codes = {metadata[i]["view_code"] for i in indices}
        if codes != {"AP", "LAT"}:
            raise ValueError(f"runtime pair {pair_id} must contain one AP and one LAT, got {sorted(codes)}")
        indices = sorted(indices, key=lambda i: metadata[i]["view_code_id"])
        units.append(tuple(indices))

    units.extend((i,) for i in sorted(singles, key=lambda j: metadata[j]["filestem"]))

    flattened = [i for unit in units for i in unit]
    if sorted(flattened) != list(range(len(files))):
        raise ValueError("VGRA units do not account for every runtime image exactly once")

    return metadata, units


class VGRAPairBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Deterministic batch sampler that never separates a valid AP/LAT pair."""

    def __init__(
        self,
        units: Iterable[tuple[int, ...]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.units = [tuple(int(i) for i in u) for u in units]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.batch_size < 2:
            raise ValueError("VGRA pair-preserving batching requires batch_size >= 2")
        if not self.units:
            raise ValueError("VGRA batch sampler requires at least one unit")
        if any(len(u) not in {1, 2} for u in self.units):
            raise ValueError("VGRA units must contain one single index or one two-image pair")

        flat = [i for u in self.units for i in u]
        if len(flat) != len(set(flat)):
            raise ValueError("VGRA units contain duplicate dataset indices")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _permute(items: list[tuple[int, ...]], generator: torch.Generator) -> list[tuple[int, ...]]:
        if len(items) <= 1:
            return list(items)
        order = torch.randperm(len(items), generator=generator).tolist()
        return [items[i] for i in order]

    def _build_batches(self, epoch: int) -> list[list[int]]:
        pairs = [u for u in self.units if len(u) == 2]
        singles = [u for u in self.units if len(u) == 1]

        g = torch.Generator()
        g.manual_seed(self.seed + int(epoch))
        if self.shuffle:
            pairs = self._permute(pairs, g)
            singles = self._permute(singles, g)

        batches: list[list[int]] = []
        singles_cursor = 0
        pairs_per_batch = max(1, self.batch_size // 2)

        for start in range(0, len(pairs), pairs_per_batch):
            chunk = pairs[start : start + pairs_per_batch]
            batch = [idx for pair in chunk for idx in pair]
            while len(batch) < self.batch_size and singles_cursor < len(singles):
                batch.append(singles[singles_cursor][0])
                singles_cursor += 1
            batches.append(batch)

        while singles_cursor < len(singles):
            batch = [u[0] for u in singles[singles_cursor : singles_cursor + self.batch_size]]
            singles_cursor += len(batch)
            batches.append(batch)

        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]

        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]

        for batch in batches:
            if not batch or len(batch) > self.batch_size:
                raise RuntimeError("invalid VGRA batch produced")
        return batches

    def __iter__(self):
        epoch = self.epoch
        batches = self._build_batches(epoch)
        if self.shuffle:
            self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches(self.epoch))


class VGRAYOLODataset(YOLODataset):
    """YOLO detection dataset with frozen VGRA pair metadata and pair-safe transforms."""

    def __init__(
        self,
        *args,
        vgra_assignment_manifest: str | Path,
        vgra_split: str,
        **kwargs,
    ):
        self.vgra_assignment_manifest = Path(vgra_assignment_manifest)
        self.vgra_split = str(vgra_split).strip().lower()
        if self.vgra_split not in {"train", "val"}:
            raise ValueError("vgra_split must be 'train' or 'val'")
        super().__init__(*args, **kwargs)

        rows = load_assignment_rows(self.vgra_assignment_manifest, self.vgra_split)
        self.vgra_metadata, self.vgra_units = bind_vgra_assignments(
            self.im_files, rows, self.vgra_split
        )

    def build_transforms(self, hyp: Any = None):
        """Build ordinary YOLO transforms with cross-study composition disabled."""
        return super().build_transforms(pair_safe_hyp(hyp))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        meta = self.vgra_metadata[index]
        sample["vgra_filestem"] = meta["filestem"]
        sample["vgra_pair_id"] = meta["pair_id"]
        sample["vgra_pair_valid"] = meta["pair_valid"]
        sample["vgra_view_code"] = meta["view_code_id"]
        sample["vgra_operational_role"] = meta["operational_role"]
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate ordinary YOLO labels plus explicit VGRA companion indices."""
        if not batch:
            raise ValueError("cannot collate an empty VGRA batch")

        meta_keys = {
            "vgra_filestem",
            "vgra_pair_id",
            "vgra_pair_valid",
            "vgra_view_code",
            "vgra_operational_role",
        }
        core_batch = []
        metadata = []
        for sample in batch:
            item = dict(sample)
            missing = meta_keys - set(item)
            if missing:
                raise ValueError(f"VGRA sample missing metadata keys: {sorted(missing)}")
            metadata.append({k: item.pop(k) for k in meta_keys})
            core_batch.append(item)

        out = YOLODataset.collate_fn(core_batch)
        bs = len(metadata)

        pair_index = torch.full((bs,), -1, dtype=torch.long)
        pair_valid = torch.tensor([bool(m["vgra_pair_valid"]) for m in metadata], dtype=torch.bool)
        view_code = torch.tensor([int(m["vgra_view_code"]) for m in metadata], dtype=torch.long)

        positions_by_pair: dict[str, list[int]] = {}
        for i, m in enumerate(metadata):
            if pair_valid[i]:
                pair_id = str(m["vgra_pair_id"])
                if not pair_id:
                    raise ValueError("valid VGRA batch member has empty pair_id")
                positions_by_pair.setdefault(pair_id, []).append(i)

        for pair_id, positions in positions_by_pair.items():
            if len(positions) != 2:
                raise ValueError(
                    f"VGRA pair {pair_id} was split across batches; found {len(positions)} member(s)"
                )
            a, b = positions
            codes = {int(view_code[a]), int(view_code[b])}
            if codes != {VGRA_VIEW_AP, VGRA_VIEW_LAT}:
                raise ValueError(f"VGRA pair {pair_id} must contain AP and LAT view codes")
            pair_index[a] = b
            pair_index[b] = a

        out["vgra_pair_index"] = pair_index
        out["vgra_pair_valid"] = pair_valid
        out["vgra_view_code"] = view_code
        out["vgra_pair_id"] = tuple(str(m["vgra_pair_id"]) for m in metadata)
        out["vgra_filestem"] = tuple(str(m["vgra_filestem"]) for m in metadata)
        out["vgra_operational_role"] = tuple(str(m["vgra_operational_role"]) for m in metadata)
        return out


def build_vgra_yolo_dataset(
    cfg: Any,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    assignment_manifest: str | Path,
    mode: str = "train",
    stride: int = 32,
) -> VGRAYOLODataset:
    """Build pair-aware VGRA dataset.

    Rectangular batching is intentionally disabled in both train and val because pair-preserving
    batches do not follow the stock aspect-ratio batch ordering.
    """
    if mode not in {"train", "val"}:
        raise ValueError("mode must be 'train' or 'val'")
    return VGRAYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=False,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        vgra_assignment_manifest=assignment_manifest,
        vgra_split=mode,
    )


def build_vgra_dataloader(
    dataset: VGRAYOLODataset,
    batch: int,
    workers: int,
    shuffle: bool,
    seed: int,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    """Build single-process VGRA pair-preserving dataloader.

    V1 deliberately fails closed for DDP. Distributed pair sharding may be added only in a later
    governed transaction if multi-GPU training becomes necessary.
    """
    if rank != -1:
        raise NotImplementedError("VGRA V1 pair-preserving dataloader supports single-process training only")
    if batch < 2:
        raise ValueError("VGRA requires image batch size >= 2")

    nd = torch.cuda.device_count()
    nw = min((os.cpu_count() or 1) // max(nd, 1), int(workers))
    pair_batch_sampler = VGRAPairBatchSampler(
        dataset.vgra_units,
        batch_size=min(int(batch), len(dataset)),
        shuffle=shuffle,
        seed=int(seed),
        drop_last=drop_last,
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK + int(seed))

    loader = InfiniteDataLoader(
        dataset=dataset,
        batch_sampler=pair_batch_sampler,
        num_workers=nw,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=nd > 0 and pin_memory,
        collate_fn=dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loader.vgra_pair_batch_sampler = pair_batch_sampler
    return loader


__all__ = (
    "VGRAYOLODataset",
    "VGRAPairBatchSampler",
    "bind_vgra_assignments",
    "build_vgra_dataloader",
    "build_vgra_yolo_dataset",
    "load_assignment_rows",
    "pair_safe_hyp",
    "PAIR_SAFE_DISABLED_AUGMENTATIONS",
    "VGRA_VIEW_SINGLE",
    "VGRA_VIEW_AP",
    "VGRA_VIEW_LAT",
    "VGRA_VIEW_OTHER",
)
