"""Two-rank CPU/Gloo native FRESH/G4 RESUME state lifecycle tests.

Fixtures use synthetic model tensors, fake best.pt and a locally calculated
manifest digest standing in for an external pin. They do NOT establish
off-session authenticity, real YOLO/VGRA, CUDA, NCCL, or run authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from research.r9_training.ddp_b16.distributed_pair_sampler import DistributedPairBatchSampler
from research.r9_training.ddp_b16.ddp_g4_recovery import distributed_capture_and_publish
from research.r9_training.ddp_b16.ddp_native_state import (
    SCHEMA, DDPNativeStateError, ddp_b16_native_state_start, ddp_b16_begin_epoch,
)
from research.r9_training.ddp_b16.ddp_startup_boundary import require_ddp_b16_native_preloop


def identity():
    return {
        "arm": "MV00", "run_id": "DDP_NATIVE_CPU_FIXTURE",
        "ddp_release_sha": "a" * 40, "ddp_recipe_sha256": "b" * 64,
        "assignment_sha256": "c" * 64, "pretrained_sha256": "d" * 64,
        "stage_source_manifest_sha256": "e" * 64,
    }


def cursor():
    return {
        "completed_epoch": 0, "next_epoch": 1,
        "steps_per_rank": 890, "global_batch": 16,
        "optimizer_updates": 1, "last_opt_step": 3,
        "accumulate": 4, "sampler_next_epoch": 1,
    }


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeLoader:
    def __init__(self, rank):
        units = [(2 * i, 2 * i + 1) for i in range(6496)]
        units.extend((12992 + i,) for i in range(1235))
        sampler = DistributedPairBatchSampler(units, rank=rank, seed=42)
        self.vgra_pair_batch_sampler = sampler
        self.batch_sampler = SimpleNamespace(sampler=sampler)
        self.num_workers = 0
        self.generator = torch.Generator().manual_seed(700 + rank)
        self.dataset = SimpleNamespace(transforms=SimpleNamespace(restored=None))
        self.reset_calls = 0

    def __len__(self):
        return 890

    def reset(self):
        self.reset_calls += 1


class EarlyPairedTrainer:
    def __init__(self, root, rank):
        torch.manual_seed(250 + rank)
        self.model = nn.Linear(3, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.937)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
        self.scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1.0)
        self.ema = SimpleNamespace(ema=copy.deepcopy(self.model), updates=0)
        self.stopper = SimpleNamespace(best_fitness=0.0, best_epoch=0, possible_stop=False)
        self.world_size = 2
        self.start_epoch = 0
        self.accumulate = 1
        self.best_fitness = 0.0
        self._d4_optimizer_steps_observed = 0
        self._ddp_b16_g4_identity = identity()
        self._ddp_b16_g4_root = str(root)
        self._ddp_b16_g4_native_bound = True
        self._ddp_b16_verified_worker_ready = True
        self._d4e_recovery_configured = False
        self._g3b4_ready = False
        self.args = SimpleNamespace(
            batch=16, imgsz=1024, epochs=100, seed=42, workers=0,
            nbs=64, optimizer="SGD", amp=True, resume=False, val=True, save=True)
        self.train_loader = FakeLoader(rank)
        self.best = Path(root) / "best.pt"
        self.last = Path(root) / "last.pt"
        self._ddp_b16_native_state_request = None


def _state_request(root, mode, pin=None):
    return {
        "schema": SCHEMA, "mode": mode, "identity": identity(),
        "checkpoint_root": str(root),
        "cursor": cursor() if mode == "RESUME_FROM_DDP_B16" else None,
        "externally_pinned_manifest_sha256": pin,
    }


def _install_fake_frozen_augmentation_restorer():
    # No Ultralytics installed on CPU Gloo CI. The production module imports
    # the frozen restore_augmentation_rng; this is a TEST-ONLY injected stub.
    names = ["ultralytics", "ultralytics.models", "ultralytics.models.yolo",
             "ultralytics.models.yolo.detect"]
    for name in names:
        obj = types.ModuleType(name)
        obj.__path__ = []
        sys.modules[name] = obj
    module = types.ModuleType("ultralytics.models.yolo.detect.d4e_transform_rng_state")
    module.restore_augmentation_rng = (
        lambda transforms, state: setattr(transforms, "restored", state))
    sys.modules[module.__name__] = module


def _publish_fixture(root, rank):
    before = EarlyPairedTrainer(root, rank)
    # Synthetic optimizer/EMA/G4 state, never patient data or real validation.
    loss = before.model(torch.ones(2, 3) * (rank + 1)).square().sum()
    before.scaler.scale(loss).backward()
    before.scaler.step(before.optimizer)
    before.scaler.update()
    before.scheduler.step()
    before.optimizer.zero_grad(set_to_none=True)
    before.ema.ema.load_state_dict(before.model.state_dict())
    before.ema.updates = 1
    before.train_loader.vgra_pair_batch_sampler.set_epoch(1)
    before._d4_optimizer_steps_observed = 1
    dist.barrier()
    best_hash = hashlib.sha256((Path(root) / "best.pt").read_bytes()).hexdigest()
    result = distributed_capture_and_publish(
        root, rank=rank, capture_kwargs={
            "identity": identity(), "cursor": cursor(),
            "model": before.model, "optimizer": before.optimizer,
            "ema_model": before.ema.ema, "ema_updates": before.ema.updates,
            "scheduler": before.scheduler, "scaler": before.scaler,
            "loader_generator": before.train_loader.generator,
            "augmentation_rng": {
                "schema": "D4E_TRAIN_ALBUMENTATIONS_RNG_R2",
                "entries": [{"path": "fixture", "python_state": None,
                             "numpy_state": None}],
            },
            "sampler": before.train_loader.vgra_pair_batch_sampler,
            "stopper": {"best_fitness": 0.5, "best_epoch": 0,
                        "possible_stop": False,
                        "selected_best_sha256": best_hash},
        },
    )
    assert result["status"] == "LOCAL_SYNCHRONIZED_DDP_CAPSULE_EXTERNAL_AUTHENTICATION_PENDING"
    path = Path(root) / "DDP_NATIVE_CPU_FIXTURE_epoch001_two_rank_manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker(rank, port, root, scenario, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2)
    try:
        random.seed(900 + rank)
        np.random.seed(1200 + rank)
        _install_fake_frozen_augmentation_restorer()
        if scenario == "fresh":
            trainer = EarlyPairedTrainer(root, rank)
            trainer._ddp_b16_native_state_request = _state_request(
                root, "FRESH_FROM_EARLY")
            result = ddp_b16_native_state_start(trainer)
            assert result == -1
            require_ddp_b16_native_preloop(trainer)
            ddp_b16_begin_epoch(trainer, 0)
            assert trainer.train_loader.reset_calls == 1
            assert trainer.start_epoch == 0
            queue.put((rank, "PASS"))
        else:
            if rank == 0:
                (Path(root) / "best.pt").write_bytes(b"SYNTHETIC_SELECTED_BEST_ONLY")
            dist.barrier()
            pin = _publish_fixture(root, rank)
            if scenario == "best_tamper" and rank == 0:
                (Path(root) / "best.pt").write_bytes(b"TAMPERED_SELECTED_BEST")
            dist.barrier()
            if scenario == "bad_pin":
                pin = "0" * 64
            trainer = EarlyPairedTrainer(root, rank)
            trainer._ddp_b16_native_state_request = _state_request(
                root, "RESUME_FROM_DDP_B16", pin)
            try:
                last = ddp_b16_native_state_start(trainer)
                if scenario != "resume":
                    raise AssertionError("Expected resume refusal")
                assert last == 3 and trainer.start_epoch == 1
                assert trainer._d4_optimizer_steps_observed == 1
                assert trainer.ema.updates == 1
                assert trainer.accumulate == 4
                assert trainer.train_loader.dataset.transforms.restored is not None
                require_ddp_b16_native_preloop(trainer)
                assert trainer.train_loader.reset_calls == 1
                ddp_b16_begin_epoch(trainer, 1)
                assert trainer.train_loader.reset_calls == 1
                ddp_b16_begin_epoch(trainer, 2)
                assert trainer.train_loader.reset_calls == 2
                queue.put((rank, "PASS"))
            except DDPNativeStateError as exc:
                if scenario == "resume":
                    raise
                assert "PREFLIGHT_PEER_FAILED" in str(exc)
                queue.put((rank, "REFUSED"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class NativeDDPStateLifecycleTests(unittest.TestCase):
    def run_pair(self, scenario):
        with tempfile.TemporaryDirectory() as t:
            ctx = mp.get_context("spawn")
            q = ctx.SimpleQueue()
            mp.spawn(_worker, args=(_port(), t, scenario, q),
                     nprocs=2, join=True)
            return dict(q.get() for _ in range(2))

    def test_fresh_two_rank_initial_state_and_sampler_begin(self):
        self.assertEqual(self.run_pair("fresh"), {0: "PASS", 1: "PASS"})

    def test_full_rank_specific_g4_resume_cursor_and_first_epoch_reset(self):
        self.assertEqual(self.run_pair("resume"), {0: "PASS", 1: "PASS"})

    def test_untrusted_manifest_pin_fails_both_ranks(self):
        self.assertEqual(self.run_pair("bad_pin"), {0: "REFUSED", 1: "REFUSED"})

    def test_selected_best_hash_mismatch_fails_both_ranks(self):
        self.assertEqual(self.run_pair("best_tamper"), {0: "REFUSED", 1: "REFUSED"})


if __name__ == "__main__":
    unittest.main()
