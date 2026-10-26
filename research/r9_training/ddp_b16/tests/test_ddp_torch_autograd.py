"""Real two-process CPU/Gloo gradient smoke test for the DDP-B16 scaling rule.

This is a tiny synthetic autograd test, NOT a VGRA model benchmark.
Exercises unequal local batch sizes and optional visibility parameters that
are unused on one rank; CUDA AMP and true dataset/trainers remain unverified.
"""
from __future__ import annotations

import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


class ToyPairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Parameter(torch.tensor(0.35, dtype=torch.float64))
        self.visibility = nn.Parameter(torch.tensor(-0.21, dtype=torch.float64))

    def forward(self, images, pair_count: int):
        native_loss = self.native * images.sum()
        if pair_count:
            pairs = images[: 2 * pair_count].reshape(pair_count, 2)
            a = pairs.sum(-1)
            logits = torch.stack(
                (self.visibility * a,
                 -0.2 * self.visibility * a,
                 0.11 * a,
                 torch.zeros_like(a)),
                dim=-1,
            )
            labels = torch.arange(pair_count, dtype=torch.long) % 4
            vis_mean = nn.functional.cross_entropy(logits, labels)
            vis_loss = 0.25 * (2 * pair_count) * vis_mean
            return native_loss + vis_loss
        # No dependency on self.visibility: legitimately unused parameter.
        return native_loss


def _inputs(rank: int, zero_pair_second_rank: bool):
    if rank == 0:
        return torch.tensor([0.1, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8], dtype=torch.float64), 4
    x = torch.tensor([0.12, 0.32, 0.52, 0.72, 0.92, 0.22, 0.42], dtype=torch.float64)
    return x, 0 if zero_pair_second_rank else 3


def _worker(rank, port, zero_pair_second_rank, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank)
    try:
        model = ToyPairModel()
        ddp = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        x, pairs = _inputs(rank, zero_pair_second_rank)
        loss = ddp(x, pairs)
        # Existing native trainer multiplies rank loss by world_size
        # before backward; DDP then averages gradients across ranks.
        (2 * loss).backward()
        grads = (float(model.native.grad.item()), float(model.visibility.grad.item()))
        queue.put((rank, grads))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _expected(zero_pair_second_rank):
    ref = ToyPairModel()
    for rank in (0, 1):
        x, pairs = _inputs(rank, zero_pair_second_rank)
        ref(x, pairs).backward()  # gradients SUM across rank-local objectives
    return (float(ref.native.grad.item()), float(ref.visibility.grad.item()))


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RealDDPAutogradTests(unittest.TestCase):
    def test_gloo_two_rank_numerical_gradient_with_and_without_pairs(self):
        for zero in (False, True):
            with self.subTest(rank1_has_zero_pairs=zero):
                ctx = mp.get_context("spawn")
                q = ctx.SimpleQueue()
                mp.spawn(_worker, args=(_port(), zero, q), nprocs=2, join=True)
                got = dict(q.get() for _ in range(2))
                expected = _expected(zero)
                self.assertEqual(set(got), {0, 1})
                for rank in (0, 1):
                    for actual, want in zip(got[rank], expected):
                        self.assertAlmostEqual(actual, want, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
