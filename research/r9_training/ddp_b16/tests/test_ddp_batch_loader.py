"""CPU-only tests of DDP-B16 rank-local loader wiring (no torch required)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from research.r9_training.ddp_b16.distributed_pair_sampler import DistributedPairBatchSampler
from research.r9_training.ddp_b16.ddp_batch_loader import (
    DDPBatchLoaderContractError,
    begin_distributed_pair_epoch,
    verify_ddp_training_loader,
)


class FakeInfiniteLoader:
    """Models the frozen InfiniteDataLoader's repeat wrapper and reset API."""

    def __init__(self, sampler):
        self.vgra_pair_batch_sampler = sampler
        self.batch_sampler = SimpleNamespace(sampler=sampler)
        self.sampler = object()  # Stock trainer .sampler is NOT pair batch sampler.
        self.num_workers = 0
        self.resets = 0

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def reset(self):
        self.resets += 1


def units(pairs=16, singles=31):
    return [(2*i, 2*i + 1) for i in range(pairs)] + [
        (2*pairs + j,) for j in range(singles)
    ]


class DDPBatchLoaderTests(unittest.TestCase):
    def test_both_ranks_move_to_correct_epoch(self):
        data = units()
        samplers = [
            DistributedPairBatchSampler(data, rank=rank, seed=42)
            for rank in (0, 1)
        ]
        loaders = [FakeInfiniteLoader(s) for s in samplers]
        for epoch in (0, 1, 19):
            for loader in loaders:
                begin_distributed_pair_epoch(loader, epoch)
            self.assertEqual([s.epoch for s in samplers], [epoch, epoch])
            self.assertEqual([loader.resets for loader in loaders], [((0, 1, 19).index(epoch) + 1)] * 2)
            streams = [list(s) for s in samplers]
            self.assertEqual(len(streams[0]), len(streams[1]))
            self.assertEqual([s.epoch for s in samplers], [epoch+1, epoch+1])

    def test_rejects_wrong_sampler_or_missing_wiring(self):
        sampler = DistributedPairBatchSampler(units(), rank=0)
        loader = FakeInfiniteLoader(sampler)
        verify_ddp_training_loader(loader, sampler)

        loader.vgra_pair_batch_sampler = object()
        with self.assertRaises(DDPBatchLoaderContractError):
            begin_distributed_pair_epoch(loader, 1)
        loader.vgra_pair_batch_sampler = sampler

        loader.batch_sampler = SimpleNamespace(sampler=object())
        with self.assertRaises(DDPBatchLoaderContractError):
            verify_ddp_training_loader(loader, sampler)
        loader.batch_sampler = SimpleNamespace(sampler=sampler)

        loader.num_workers = 2
        with self.assertRaises(DDPBatchLoaderContractError):
            begin_distributed_pair_epoch(loader, 1)

    def test_refuses_mismatched_lengths(self):
        sampler = DistributedPairBatchSampler(units(), rank=0)
        loader = FakeInfiniteLoader(sampler)
        loader.batch_sampler = SimpleNamespace(sampler=sampler)
        loader.batch_sampler = SimpleNamespace(sampler=sampler)
        loader.__len__ = lambda: 999
        # Special methods are looked up on the class, so override via a subclass.
        class WrongLengthLoader(FakeInfiniteLoader):
            def __len__(self):
                return 999
        loader = WrongLengthLoader(sampler)
        with self.assertRaises(DDPBatchLoaderContractError):
            verify_ddp_training_loader(loader, sampler)


if __name__ == "__main__":
    unittest.main()
