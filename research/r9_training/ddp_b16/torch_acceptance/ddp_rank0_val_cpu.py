"""Real CPU/Gloo two-rank validation-boundary acceptance.

Only rank 0 may call the fake pair-aware validator. Both ranks must receive
identical finite metrics, fitness, and B-VAL=3049. Test also makes a rank-0
cardinality failure propagate to rank 1 without a collective deadlock.

Not real radiograph evaluation; CUDA and actual VGRA forward remain untested.
"""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from research.r9_training.ddp_b16.ddp_epoch_boundary import (
    DDPValidationBoundaryError, rank0_validate_and_sync,
)


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakePairValidator:
    def __init__(self, fail_cardinality: bool):
        self.calls = 0
        self.seen = 0
        self._d4_fixed_seen = 0
        self.dataloader = SimpleNamespace(dataset=range(3049))
        self.fail_cardinality = fail_cardinality

    def __call__(self, *, trainer):
        assert dist.get_rank() == 0
        assert trainer._ddp_b16_rank0_validation_ready is True
        self.calls += 1
        self.seen = 3048 if self.fail_cardinality else 3049
        self._d4_fixed_seen = self.seen
        return {"fitness": 0.4612, "metrics/mAP50-95(B)": 0.3995, "metrics/precision(B)": 0.611}


def _worker(rank, port, bad_count, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2)
    try:
        validator = FakePairValidator(bad_count)
        trainer = SimpleNamespace(
            validator=validator, world_size=2, epoch=3,
            best_fitness=0.15, metrics={}, fitness=0.0,
            _ddp_b16_rank0_validation_ready=False,
        )
        try:
            report = rank0_validate_and_sync(trainer, epoch=3)
            queue.put((rank, "PASS", report, validator.calls,
                       trainer._ddp_b16_rank0_validation_ready))
        except DDPValidationBoundaryError as exc:
            queue.put((rank, "ERROR", str(exc), validator.calls,
                       trainer._ddp_b16_rank0_validation_ready))
    finally:
        dist.destroy_process_group()


class RankZeroValidationTests(unittest.TestCase):
    def run_pair(self, bad_count=False):
        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        mp.spawn(_worker, args=(_port(), bad_count, queue), nprocs=2, join=True)
        return dict((rank, (status, result, called, ready))
                    for rank, status, result, called, ready
                    in (queue.get() for _ in range(2)))

    def test_only_rank_zero_validates_and_both_receive_same_metrics(self):
        output = self.run_pair()
        self.assertEqual({v[0] for v in output.values()}, {"PASS"})
        self.assertEqual(output[0][2], 1)
        self.assertEqual(output[1][2], 0)
        self.assertEqual(output[0][1]["metrics"], output[1][1]["metrics"])
        self.assertEqual(output[0][1]["fitness"], output[1][1]["fitness"])
        self.assertEqual(output[0][1]["operational_val_images"], 3049)
        self.assertEqual(output[1][1]["operational_val_images"], 3049)
        self.assertTrue(output[0][1]["validation_was_executed_here"])
        self.assertFalse(output[1][1]["validation_was_executed_here"])
        self.assertIs(output[0][3], False)
        self.assertIs(output[1][3], False)

    def test_bad_cardinality_fails_both_ranks_without_hang(self):
        output = self.run_pair(bad_count=True)
        self.assertEqual({v[0] for v in output.values()}, {"ERROR"})
        self.assertEqual(output[0][2], 1)
        self.assertEqual(output[1][2], 0)
        self.assertIn("EVALUATED_CARDINALITY", output[0][1])
        self.assertIn("EVALUATED_CARDINALITY", output[1][1])
        self.assertIs(output[0][3], False)
        self.assertIs(output[1][3], False)


if __name__ == "__main__":
    unittest.main()
