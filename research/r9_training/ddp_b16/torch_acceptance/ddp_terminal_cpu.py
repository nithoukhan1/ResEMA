"""Real two-rank CPU/Gloo selected-BEST terminal orchestration smoke tests.

Synthetic checkpoint bytes and fake paired validation only: this does not
establish real YOLO/VGRA, CUDA, AMP, NCCL or B-VAL model correctness.
"""
from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from research.r9_training.ddp_b16.ddp_terminal_boundary import (
    DDPB16TerminalError, rank0_selected_best_terminal,
)


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeValidator:
    def __init__(self):
        self.seen = 0
        self._d4_fixed_seen = 0
        self.dataloader = SimpleNamespace(dataset=range(3049))


class FakeTrainer:
    def __init__(self, root, *, bad_count=False, no_authority=False):
        self.world_size = 2
        self._ddp_b16_terminal_candidate_ready = not no_authority
        self._ddp_b16_g4_native_bound = True
        self._ddp_b16_rank0_validation_ready = False
        self._d4e_recovery_configured = False
        self.data = {"train": "/mock/train", "val": "/mock/val"}
        self.validator = FakeValidator()
        self.best = Path(root) / "best.pt"
        self.last = Path(root) / "last.pt"
        self.metrics = {}
        self.calls = 0
        self.bad_count = bad_count

    def final_eval(self):
        assert dist.get_rank() == 0
        assert self._ddp_b16_rank0_validation_ready is True
        self.calls += 1
        self.validator.seen = 3048 if self.bad_count else 3049
        self.validator._d4_fixed_seen = self.validator.seen
        self.metrics = {"metrics/mAP50-95(B)": 0.414, "metrics/precision(B)": 0.70}


def _worker(rank, port, root, bad_count, no_authority, q):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2)
    try:
        t = FakeTrainer(root, bad_count=bad_count, no_authority=no_authority)
        try:
            result = rank0_selected_best_terminal(t)
            q.put((rank, "PASS", result, t.calls,
                   t._ddp_b16_rank0_validation_ready))
        except DDPB16TerminalError as exc:
            q.put((rank, "ERROR", str(exc), t.calls,
                   t._ddp_b16_rank0_validation_ready))
    finally:
        dist.destroy_process_group()


class TerminalBoundaryTests(unittest.TestCase):
    def run_pair(self, *, bad_count=False, no_authority=False):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            (p / "best.pt").write_bytes(b"synthetic-best-only")
            (p / "last.pt").write_bytes(b"synthetic-last-only")
            ctx = mp.get_context("spawn")
            q = ctx.SimpleQueue()
            mp.spawn(_worker, args=(_port(), root, bad_count, no_authority, q),
                     nprocs=2, join=True)
            return {rank: (status, report, calls, ready)
                    for rank, status, report, calls, ready
                    in (q.get() for _ in range(2))}

    def test_only_rank_zero_selected_best_eval_and_identical_metrics(self):
        output = self.run_pair()
        self.assertEqual({r[0] for r in output.values()}, {"PASS"})
        self.assertEqual((output[0][2], output[1][2]), (1, 0))
        self.assertEqual(output[0][1]["metrics"], output[1][1]["metrics"])
        self.assertEqual(output[0][1]["selected_best_sha256"],
                         output[1][1]["selected_best_sha256"])
        self.assertEqual(output[0][1]["operational_val_images"], 3049)
        self.assertFalse(output[0][1]["terminal_export_authorized"])
        self.assertFalse(output[1][1]["terminal_export_authorized"])

    def test_rank0_bad_count_propagates_and_does_not_hang(self):
        output = self.run_pair(bad_count=True)
        self.assertEqual({r[0] for r in output.values()}, {"ERROR"})
        self.assertEqual((output[0][2], output[1][2]), (1, 0))
        self.assertIn("FINAL_BVAL_CARDINALITY", output[0][1])
        self.assertIn("FINAL_BVAL_CARDINALITY", output[1][1])
        self.assertFalse(output[0][3])
        self.assertFalse(output[1][3])

    def test_missing_terminal_authority_fails_both_ranks(self):
        output = self.run_pair(no_authority=True)
        self.assertEqual({r[0] for r in output.values()}, {"ERROR"})
        self.assertIn("TERMINAL_NOT_AUTHORIZED", output[0][1])
        self.assertIn("TERMINAL_NOT_AUTHORIZED", output[1][1])


if __name__ == "__main__":
    unittest.main()
