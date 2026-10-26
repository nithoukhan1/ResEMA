"""Provisional DDP-B16 training-loader integration for frozen VGRA datasets.

C1 is immutable; this module is separate execution-source candidate code.
Only the training loader is constructed here. Validation, rank barriers, AMP,
loss scaling, recovery, and executable two-GPU launch are NOT implemented.

The frozen InfiniteDataLoader wraps the supplied batch_sampler in a repeating
sampler and caches an iterator. Its regular .sampler is NOT the VGRA pair
batch_sampler. Every epoch must call begin_distributed_pair_epoch() instead
of the stock trainer's train_loader.sampler.set_epoch(epoch).
"""
from __future__ import annotations

from typing import Any

try:
    from .distributed_pair_sampler import DistributedPairBatchSampler
except ImportError:
    # Standalone file imports (without a package) are intentionally not enabled:
    # runtime must import from the pinned ddp_b16 package.
    raise


class DDPBatchLoaderContractError(RuntimeError):
    """Incorrect/mismatched rank-local pair-loader integration."""


def verify_ddp_training_loader(loader: Any, sampler: DistributedPairBatchSampler) -> None:
    """Validate direct sampler wiring against Ultralytics InfiniteDataLoader."""
    if loader is None or sampler is None:
        raise DDPBatchLoaderContractError("loader and sampler are required")
    if getattr(loader, "vgra_pair_batch_sampler", None) is not sampler:
        raise DDPBatchLoaderContractError("pair sampler not attached to loader")
    repeat = getattr(loader, "batch_sampler", None)
    if getattr(repeat, "sampler", None) is not sampler:
        raise DDPBatchLoaderContractError("InfiniteDataLoader is not repeating pair sampler")
    if getattr(loader, "num_workers", None) != 0:
        raise DDPBatchLoaderContractError("frozen recovery requires workers=0")
    if not callable(getattr(loader, "reset", None)):
        raise DDPBatchLoaderContractError("loader does not provide reset()")
    if len(loader) != len(sampler):
        raise DDPBatchLoaderContractError("loader/sampler step count mismatch")


def begin_distributed_pair_epoch(loader: Any, epoch: int) -> None:
    """Set the real pair sampler epoch and discard the cached old iterator.

    This must be invoked on BOTH ranks after the model is ready and before the
    FIRST batch of each epoch. At resume, the separately verified distributed
    recovery mechanism must restore states before this call. This function
    never touches optimizer, GradScaler, DDP state, or model weights.
    """
    sampler = getattr(loader, "vgra_pair_batch_sampler", None)
    if not isinstance(sampler, DistributedPairBatchSampler):
        raise DDPBatchLoaderContractError("wrong pair sampler class")
    verify_ddp_training_loader(loader, sampler)
    sampler.set_epoch(epoch)
    loader.reset()  # InfiniteDataLoader caches an iterator; reset is mandatory.
    if sampler.epoch != epoch:
        raise DDPBatchLoaderContractError("pair sampler epoch changed unexpectedly")
    verify_ddp_training_loader(loader, sampler)


def build_distributed_pair_training_loader(
    dataset: Any,
    *,
    rank: int,
    global_batch: int = 16,
    seed: int = 42,
    workers: int = 0,
) -> Any:
    """Create rank-local InfiniteDataLoader from authenticated VGRAYOLODataset.

    Requires the *already authenticated* dataset and frozen Ultralytics runtime.
    No dataset discovery, default download or new pairing inference occurs here.
    """
    if global_batch != 16 or type(global_batch) is not int:
        raise ValueError("DDP-B16 requires global batch exactly 16")
    if type(workers) is not int or workers != 0:
        raise ValueError("DDP-B16 workers=0 until recovery acceptance")
    if not hasattr(dataset, "vgra_units") or not callable(getattr(dataset, "collate_fn", None)):
        raise TypeError("DDP-B16 requires the verified VGRAYOLODataset")
    if not hasattr(dataset, "__len__"):
        raise TypeError("DDP-B16 missing dataset length")

    # Torch/Ultralytics are imported only when the real runtime builder is
    # invoked. Pure source-unit tests run without CUDA, dataset or GPU access.
    import torch
    from ultralytics.data.build import InfiniteDataLoader, seed_worker

    sampler = DistributedPairBatchSampler(
        dataset.vgra_units,
        rank=rank,
        world_size=2,
        per_rank_batch=global_batch // 2,
        seed=seed,
        shuffle=True,
    )
    if sampler.num_images != len(dataset):
        raise DDPBatchLoaderContractError("sampler does not cover authenticated dataset")
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + rank + seed)

    loader = InfiniteDataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loader.vgra_pair_batch_sampler = sampler
    verify_ddp_training_loader(loader, sampler)
    return loader
