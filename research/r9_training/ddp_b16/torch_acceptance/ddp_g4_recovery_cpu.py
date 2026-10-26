"""CPU Gloo two-rank checkpoint save/verification/restore acceptance.

Tests the candidate capsule library only. Not a real YOLO/VGRA training
resume, not an independent offsite pin, and not CUDA/AMP acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from research.r9_training.ddp_b16.ddp_g4_recovery import (
    DDPG4Error, capture_rank, publish_pair, read_pinned_rank,
    restore_rank, save_rank, validate_cursor, validate_identity,
)


def _identity():
    return {
        "arm": "MV01", "run_id": "DDP_B16_TEST_01",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": "b" * 64,
        "assignment_sha256": "c" * 64,
        "pretrained_sha256": "d" * 64,
        "stage_source_manifest_sha256": "e" * 64,
    }


def _cursor():
    return {
        "completed_epoch": 0, "next_epoch": 1,
        "steps_per_rank": 890, "global_batch": 16,
        "optimizer_updates": 1, "last_opt_step": 3,
        "accumulate": 4, "sampler_next_epoch": 1,
    }


def _fresh_components(rank):
    torch.manual_seed(100 + rank)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99**epoch)
    ema = copy.deepcopy(model)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1.0)
    gen = torch.Generator().manual_seed(42 + rank)
    return model, optimizer, scheduler, ema, scaler, gen


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, port, root, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank)
    try:
        torch.manual_seed(111 + rank)
        random.seed(400 + rank)
        np.random.seed(500 + rank)
        model, optimizer, scheduler, ema, scaler, gen = _fresh_components(rank)
        optimizer.zero_grad(set_to_none=True)
        values = torch.ones(2, 3) * (rank + 1)
        loss = model(values).square().sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.load_state_dict(model.state_dict())
        optimizer.zero_grad(set_to_none=True)
        augmentation = {
            "schema": "D4E_TRAIN_ALBUMENTATIONS_RNG_R2",
            "entries": [{"path": "fixture-only", "python_state": None, "numpy_state": None}],
        }
        stopper = {
            "best_fitness": 0.5, "best_epoch": 0,
            "possible_stop": False, "selected_best_sha256": "f" * 64,
        }
        payload = capture_rank(
            rank=rank, identity=_identity(), cursor=_cursor(), model=model,
            optimizer=optimizer, ema_model=ema, ema_updates=1,
            scheduler=scheduler, scaler=scaler, loader_generator=gen,
            augmentation_rng=augmentation, stopper=stopper,
        )
        save_rank(payload, root)
        dist.barrier()
        if rank == 0:
            publish_pair(root, identity=_identity(), cursor=_cursor())
        dist.barrier()

        # In this CPU test only, the locally produced digest simulates an
        # externally pinned digest. Do NOT call this off-session proof.
        stem = "DDP_B16_TEST_01_epoch001"
        manifest = Path(root) / (stem + "_two_rank_manifest.json")
        expected_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result = read_pinned_rank(
            root, rank=rank, identity=_identity(), cursor=_cursor(),
            externally_pinned_manifest_sha256=expected_sha,
        )
        fresh_model, fresh_opt, fresh_sched, fresh_ema, fresh_scaler, fresh_gen = _fresh_components(rank)
        seen = []
        restored = restore_rank(
            result, rank=rank, model=fresh_model, optimizer=fresh_opt,
            ema_model=fresh_ema, scheduler=fresh_sched, scaler=fresh_scaler,
            loader_generator=fresh_gen, restore_augmentation=lambda x: seen.append(x),
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(fresh_model.state_dict()[name], value, rtol=0, atol=0)
        for name, value in ema.state_dict().items():
            torch.testing.assert_close(fresh_ema.state_dict()[name], value, rtol=0, atol=0)
        assert fresh_scaler.state_dict() == scaler.state_dict()
        assert fresh_sched.state_dict() == scheduler.state_dict()
        assert fresh_gen.get_state().equal(gen.get_state())
        assert seen == [augmentation]
        assert restored["next_epoch"] == 1 and restored["optimizer_updates"] == 1
        assert fresh_opt.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
        assert len(fresh_opt.state_dict()["state"]) == len(optimizer.state_dict()["state"])
        queue.put((rank, True))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DDPG4CapsuleTests(unittest.TestCase):
    def test_two_process_rank_capsules_and_exact_state_restore(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = mp.get_context("spawn")
            q = ctx.SimpleQueue()
            mp.spawn(_worker, args=(_port(), t, q), nprocs=2, join=True)
            self.assertEqual(dict(q.get() for _ in range(2)), {0: True, 1: True})
            files = sorted(p.name for p in Path(t).iterdir())
            self.assertEqual(len(files), 5)
            self.assertTrue(any("two_rank_manifest" in x for x in files))
            digest = hashlib.sha256(
                (Path(t) / "DDP_B16_TEST_01_epoch001_two_rank_manifest.json").read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(DDPG4Error, "EXTERNAL_MANIFEST_SHA"):
                read_pinned_rank(
                    t, rank=0, identity=_identity(), cursor=_cursor(),
                    externally_pinned_manifest_sha256="0" * 64,
                )
            with self.assertRaisesRegex(DDPG4Error, "EXTERNAL_MANIFEST_PIN"):
                read_pinned_rank(
                    t, rank=0, identity=_identity(), cursor=_cursor(),
                    externally_pinned_manifest_sha256="",
                )
            with self.assertRaisesRegex(DDPG4Error, "FILE_OVERWRITE"):
                publish_pair(t, identity=_identity(), cursor=_cursor())
            # Tamper a rank checkpoint after pinning the manifest:
            rank1 = Path(t) / "DDP_B16_TEST_01_epoch001_rank1.pt"
            with rank1.open("ab") as f:
                f.write(b"malicious mutation")
            with self.assertRaisesRegex(DDPG4Error, "CHECKPOINT_SHA"):
                read_pinned_rank(
                    t, rank=0, identity=_identity(), cursor=_cursor(),
                    externally_pinned_manifest_sha256=digest,
                )

    def test_reject_wrong_recipe_identity_and_cursor(self):
        wrong = _identity()
        wrong["arm"] = "MV02"
        with self.assertRaises(DDPG4Error):
            validate_identity(wrong)
        cursor = _cursor()
        cursor["global_batch"] = 8
        with self.assertRaisesRegex(DDPG4Error, "BATCH_RECIPE"):
            validate_cursor(cursor)
        cursor = _cursor()
        cursor["sampler_next_epoch"] = 0
        with self.assertRaisesRegex(DDPG4Error, "SAMPLER_EPOCH"):
            validate_cursor(cursor)


if __name__ == "__main__":
    unittest.main()
