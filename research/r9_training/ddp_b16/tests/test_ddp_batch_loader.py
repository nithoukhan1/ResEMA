"""CPU-only tests of DDP-B16 rank-local loader wiring (no torch required)."""
from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from research.r9_training.ddp_b16.distributed_pair_sampler import DistributedPairBatchSampler
from research.r9_training.ddp_b16.ddp_batch_loader import (
    DDPBatchLoaderContractError,
    begin_distributed_pair_epoch,
    build_distributed_pair_training_loader,
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
        # Special methods are looked up on the class, so override via a subclass.
        class WrongLengthLoader(FakeInfiniteLoader):
            def __len__(self):
                return 999
        loader = WrongLengthLoader(sampler)
        with self.assertRaises(DDPBatchLoaderContractError):
            verify_ddp_training_loader(loader, sampler)


    def test_runtime_builder_wires_pair_sampler_without_stock_sampler(self):
        captured = {}

        class FakeGenerator:
            def manual_seed(self, seed):
                captured["seed"] = seed

        class FakeRuntimeLoader(FakeInfiniteLoader):
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                super().__init__(kwargs["batch_sampler"])

        fake_torch = ModuleType("torch")
        fake_torch.Generator = FakeGenerator
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
        fake_ultra = ModuleType("ultralytics")
        fake_data = ModuleType("ultralytics.data")
        fake_build = ModuleType("ultralytics.data.build")
        fake_build.InfiniteDataLoader = FakeRuntimeLoader
        fake_build.seed_worker = lambda _: None

        dataset = SimpleNamespace(vgra_units=units(), collate_fn=lambda x: x)
        class BoundDataset:
            vgra_units = dataset.vgra_units
            collate_fn = staticmethod(dataset.collate_fn)
            def __len__(self):
                return 63
        fake_modules = {
            "torch": fake_torch,
            "ultralytics": fake_ultra,
            "ultralytics.data": fake_data,
            "ultralytics.data.build": fake_build,
        }
        with patch.dict(sys.modules, fake_modules):
            a = build_distributed_pair_training_loader(BoundDataset(), rank=0)
        self.assertEqual(captured["kwargs"]["num_workers"], 0)
        self.assertTrue(captured["kwargs"]["pin_memory"])
        self.assertEqual(captured["seed"], 6148914691236517205 + 42)
        self.assertEqual(len(a), 4)
        self.assertEqual(a.vgra_pair_batch_sampler.rank, 0)
        begin_distributed_pair_epoch(a, 11)
        self.assertEqual(a.resets, 1)
        self.assertEqual(a.vgra_pair_batch_sampler.epoch, 11)

    def test_runtime_builder_rejects_recipe_drift(self):
        class Dataset:
            vgra_units = units()
            collate_fn = staticmethod(lambda x: x)
            def __len__(self):
                return 63
        for kwargs in ({"workers": 2}, {"global_batch": 32}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_distributed_pair_training_loader(Dataset(), rank=0, **kwargs)


if __name__ == "__main__":
    unittest.main()
