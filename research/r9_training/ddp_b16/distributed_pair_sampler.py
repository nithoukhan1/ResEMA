"""Two-rank, pair-preserving batch sharding for VGRA DDP-B16.

Additive candidate: does not import or modify the frozen C1 Ultralytics fork.
Takes the already-verified VGRAYOLODataset.vgra_units: tuples of one
single-view index or two AP/LAT indices (in that order).

One complete pair is ALWAYS assigned to exactly one rank in one step.
All indices are covered once per epoch. Every rank has the same number of
nonempty steps, with <= 8 images per rank. No synthetic padding/duplication.

This is a SAMPLER COMPONENT ONLY. DDP trainer, AMP, loss normalization,
validation and checkpoint/resume are NOT implemented or authorized here.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator


class DistributedPairBatchSampler:
    """Pure-Python PyTorch DataLoader batch_sampler interface for 2 ranks.

    The iterator increments epoch on TRAIN (shuffle=True), matching the frozen
    single-process VGRA sampler's epoch cursor convention. The future native
    DDP loop MUST invoke .set_epoch(epoch) on THIS sampler, not the default
    DataLoader.sampler, and MUST prove correct InfiniteDataLoader reset semantics.
    """

    def __init__(
        self,
        units: Iterable[tuple[int, ...]],
        *,
        rank: int,
        world_size: int = 2,
        per_rank_batch: int = 8,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if type(world_size) is not int or world_size != 2:
            raise ValueError("DDP_B16 requires exactly 2 ranks")
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError("DDP_B16 rank must be 0 or 1")
        if type(per_rank_batch) is not int or per_rank_batch != 8:
            raise ValueError("DDP_B16 local image batch must be exactly 8")
        if type(seed) is not int or seed < 0:
            raise ValueError("DDP_B16 seed must be a nonnegative integer")

        data = [tuple(unit) for unit in units]
        if not data:
            raise ValueError("DDP_B16 units cannot be empty")
        if any(len(unit) not in (1, 2) for unit in data):
            raise ValueError("DDP_B16 units must have one or two indices")
        if any(type(i) is not int or i < 0 for unit in data for i in unit):
            raise ValueError("DDP_B16 invalid dataset index")
        flat = [i for unit in data for i in unit]
        if len(flat) != len(set(flat)):
            raise ValueError("DDP_B16 duplicated dataset index in units")
        if set(flat) != set(range(len(flat))):
            raise ValueError("DDP_B16 units must cover the complete dense dataset index range")

        self.units = data
        self.rank = rank
        self.world_size = world_size
        self.per_rank_batch = per_rank_batch
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.epoch = 0
        self.num_images = len(flat)
        self.num_pairs = sum(len(unit) == 2 for unit in data)
        self.num_singles = sum(len(unit) == 1 for unit in data)

        # Both ranks must participate in every step, including partial batches.
        # This also rejects toy populations too small to shard safely.
        self._build_plan(0)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("DDP_B16 epoch must be a nonnegative integer")
        self.epoch = epoch

    @staticmethod
    def _balanced_chunks(items: list[tuple[int, ...]], capacity: int) -> list[list[tuple[int, ...]]]:
        """Balanced step sizes avoid a pathological 1-image tail on either rank."""
        if not items:
            return []
        steps = (len(items) + capacity - 1) // capacity
        q, r = divmod(len(items), steps)
        sizes = [q + (i < r) for i in range(steps)]
        chunks = []
        cursor = 0
        for n in sizes:
            chunks.append(items[cursor:cursor + n])
            cursor += n
        if cursor != len(items):
            raise RuntimeError("DDP_B16 chunk accounting failed")
        return chunks

    def _build_plan(self, epoch: int) -> list[tuple[list[int], list[int]]]:
        pairs = [u for u in self.units if len(u) == 2]
        singles = [u for u in self.units if len(u) == 1]
        rng = random.Random(self.seed + epoch)
        if self.shuffle:
            rng.shuffle(pairs)
            rng.shuffle(singles)

        # Each global pair step holds <=8 complete pair units (16 images).
        # Each global single step holds <=16 distinct single images.
        categories = [
            self._balanced_chunks(pairs, 8),
            self._balanced_chunks(singles, 16),
        ]
        steps: list[tuple[list[int], list[int]]] = []
        uneven_index = 0
        for chunks in categories:
            for chunk in chunks:
                if len(chunk) % 2:
                    # Alternate which rank receives the extra unit. This
                    # makes the epoch-level image imbalance <=1 for singles.
                    left_count = (len(chunk) + 1) // 2 if uneven_index % 2 == 0 else len(chunk) // 2
                    uneven_index += 1
                else:
                    left_count = len(chunk) // 2
                left = [image for unit in chunk[:left_count] for image in unit]
                right = [image for unit in chunk[left_count:] for image in unit]
                if not left or not right or len(left) > 8 or len(right) > 8:
                    raise ValueError("DDP_B16 nonempty balanced per-rank step impossible")
                steps.append((left, right))

        if self.shuffle:
            rng.shuffle(steps)

        flattened = [i for left, right in steps for b in (left, right) for i in b]
        if len(flattened) != self.num_images or set(flattened) != set(range(self.num_images)):
            raise RuntimeError("DDP_B16 epoch coverage or duplication failure")

        # Every pair must reside in one local batch on one rank, never
        # across steps or ranks. Check positions, not just total membership.
        location = {}
        for step_index, (left, right) in enumerate(steps):
            for rank_index, batch in enumerate((left, right)):
                for idx in batch:
                    if idx in location:
                        raise RuntimeError("DDP_B16 duplicated runtime index")
                    location[idx] = (step_index, rank_index)
        for pair in pairs:
            if location[pair[0]] != location[pair[1]]:
                raise RuntimeError("DDP_B16 pair split across GPU ranks or steps")
        return steps

    def __iter__(self) -> Iterator[list[int]]:
        steps = self._build_plan(self.epoch)
        if self.shuffle:
            self.epoch += 1
        for local in steps:
            yield list(local[self.rank])

    def __len__(self) -> int:
        return (self.num_pairs + 7) // 8 + (self.num_singles + 15) // 16

    def audit(self, epoch: int) -> dict[str, int]:
        """Safe aggregate counts; no study IDs, filenames, or image content."""
        steps = self._build_plan(epoch)
        return {
            "epoch": epoch,
            "steps_per_rank": len(steps),
            "images_total": self.num_images,
            "complete_pairs_total": self.num_pairs,
            "singles_total": self.num_singles,
            "rank0_images": sum(len(x[0]) for x in steps),
            "rank1_images": sum(len(x[1]) for x in steps),
            "min_local_batch": min(len(b) for step in steps for b in step),
            "max_local_batch": max(len(b) for step in steps for b in step),
        }
