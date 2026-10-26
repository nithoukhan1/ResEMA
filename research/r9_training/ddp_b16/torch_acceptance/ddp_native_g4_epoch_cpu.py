"""Two-process native G4 epoch-hook smoke tests (synthetic CPU/Gloo only).

Tests are NOT an authenticated real-data/GPU training replay.
"""
from __future__ import annotations

import copy
import hashlib
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

from research.r9_training.ddp_b16.distributed_pair_sampler import DistributedPairBatchSampler
from research.r9_training.ddp_b16.ddp_native_g4_hook import (
    DDPG4NativeHookError, ddp_b16_g4_epoch_end,
)


def _identity():
    return {
        "arm": "MV01", "run_id": "DDP_B16_NATIVE_G4_CPU",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": "b" * 64,
        "assignment_sha256": "c" * 64,
        "pretrained_sha256": "d" * 64,
        "stage_source_manifest_sha256": "e" * 64,
    }


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sampler(rank, epoch):
    units = [(2 * i, 2 * i + 1) for i in range(6496)]
    units += [(12992 + j,) for j in range(1235)]
    sampler = DistributedPairBatchSampler(units, rank=rank, seed=42)
    sampler.set_epoch(epoch + 1)
    return sampler


def _trainer(rank, root, epoch, scenario):
    torch.manual_seed(100 + rank)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.99 ** e)
    ema_model = copy.deepcopy(model)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1.0)
    sampler = _sampler(rank, epoch)
    loader = FakeLoader(
        vgra_pair_batch_sampler=sampler,
        generator=torch.Generator().manual_seed(42 + rank),
        dataset=SimpleNamespace(transforms=object()),
    )
    options = SimpleNamespace(
        batch=16, imgsz=1024, epochs=100, workers=0, seed=42, nbs=64, amp=True,
    )
    steps = 2 if scenario == "step_drift" and rank == 1 else 1
    return SimpleNamespace(
        args=options, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, ema=SimpleNamespace(ema=ema_model, updates=1),
        train_loader=loader, epoch=epoch, world_size=2,
        last=str(Path(root) / "last.pt"),
        best=str(Path(root) / ("missing_best.pt" if scenario == "no_best" else "best.pt")),
        stopper=SimpleNamespace(best_epoch=0, possible_stop=False),
        best_fitness=0.48, fitness=0.49, metrics={"map": 0.42},
        stop=False, _d4_optimizer_steps_observed=steps, accumulate=4,
        _d4e_recovery_configured=False,
        _ddp_b16_g4_native_bound=True, _ddp_b16_g4_identity=_identity(),
        _ddp_b16_g4_root=root,
    )


class FakeLoader(SimpleNamespace):
    def __len__(self):
        return 890


def _worker(rank, port, root, scenario, queue):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
    )
    try:
        epoch = 99 if scenario == "terminal" else 0
        trainer = _trainer(rank, root, epoch, scenario)
        try:
            result = ddp_b16_g4_epoch_end(
                trainer, epoch=epoch, nb=890, last_opt_step=3,
                _cpu_test_augment_capture=lambda root: {
                    "schema": "D4E_TRAIN_ALBUMENTATIONS_RNG_R2",
                    "entries": [{"path": "fixture", "python_state": None}],
                },
            )
            queue.put((rank, "PASS", result["status"]))
        except DDPG4NativeHookError as exc:
            queue.put((rank, "ERROR", str(exc)))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class NativeG4EpochHookTests(unittest.TestCase):
    def run_case(self, scenario):
        with tempfile.TemporaryDirectory() as base:
            p = Path(base)
            (p / "last.pt").write_bytes(b"synthetic-native-last")
            if scenario != "no_best":
                (p / "best.pt").write_bytes(b"synthetic-native-selected-best")
            q = mp.get_context("spawn").SimpleQueue()
            mp.spawn(_worker, args=(_port(), base, scenario, q), nprocs=2, join=True)
            messages = dict((rank, (status, msg)) for rank, status, msg in
                            (q.get() for _ in range(2)))
            snapshot_files = list(p.glob("*rank*.pt"))
            manifest_files = list(p.glob("*two_rank_manifest.json"))
            return messages, len(snapshot_files), len(manifest_files)

    def test_nonterminal_both_ranks_publish(self):
        messages, snapshots, manifests = self.run_case("ordinary")
        self.assertEqual([messages[i][0] for i in (0, 1)], ["PASS", "PASS"])
        self.assertEqual(snapshots, 2)
        self.assertEqual(manifests, 1)
        for _, status in messages.values():
            self.assertEqual(status, "LOCAL_SYNCHRONIZED_DDP_CAPSULE_EXTERNAL_AUTHENTICATION_PENDING")

    def test_missing_rank0_best_fails_both_ranks(self):
        messages, snapshots, manifests = self.run_case("no_best")
        self.assertEqual([messages[i][0] for i in (0, 1)], ["ERROR", "ERROR"])
        self.assertEqual(snapshots, 0)
        self.assertEqual(manifests, 0)
        self.assertIn("RANK0_CHECKPOINT", messages[1][1])

    def test_optimizer_step_mismatch_fails_both_ranks(self):
        messages, snapshots, manifests = self.run_case("step_drift")
        self.assertEqual([messages[i][0] for i in (0, 1)], ["ERROR", "ERROR"])
        self.assertEqual(snapshots, 0)
        self.assertEqual(manifests, 0)
        self.assertIn("RANK_STEP_DIVERGENCE", messages[0][1])
        self.assertIn("RANK_STEP_DIVERGENCE", messages[1][1])

    def test_terminal_epoch_refuses_resume_capsule(self):
        messages, snapshots, manifests = self.run_case("terminal")
        self.assertEqual([messages[i][0] for i in (0, 1)], ["PASS", "PASS"])
        self.assertEqual([messages[i][1] for i in (0, 1)],
                         ["TERMINAL_EPOCH_G4_CAPSULE_NOT_WRITTEN"] * 2)
        self.assertEqual(snapshots, 0)
        self.assertEqual(manifests, 0)


if __name__ == "__main__":
    unittest.main()
