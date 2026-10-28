"""True CPU/Gloo two-rank native-binding consensus (synthetic data only)."""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from research.r9_training.ddp_b16.ddp_native_binding_consensus import (
    DDPNativeBindingConsensusError, require_native_binding_consensus,
)
from research.r9_training.ddp_b16.ddp_native_binding import SCHEMA, CLASS
from research.r9_training.ddp_b16.ddp_verified_admission import RECIPE, sha_object


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake(rank, mode):
    arm = "MV00" if mode != "arm" or rank == 0 else "MV01"
    identity = {
        "arm": arm, "run_id": "FIXTURE",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": sha_object(RECIPE),
        "assignment_sha256": "c" * 64,
        "pretrained_sha256": "d" * 64,
        "stage_source_manifest_sha256": "e" * 64,
    }
    body = {
        "arm": arm, "identity": identity,
        "source_root": "/fixture/source",
        "modules_root": "/fixture/modules",
        "trainer_class": CLASS[arm],
        "trainer_module": "fixture.module",
        "constructor": {"pretrained_best": "/fixture/best.pt"},
        "overrides": {"batch": 16, "device": "0,1", "epochs": 100,
                      "resume": False, "amp": True},
        "r0_record_sha256": "f" * 64,
        "stage_receipt_sha256": "1" * 64,
    }
    if rank == 1 and mode == "constructor":
        body["constructor"] = {"pretrained_best": "/fixture/different.pt"}
    value = {
        "schema": SCHEMA, **body,
        "binding_sha256": sha_object(body),
        "status": "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY",
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }
    if rank == 1 and mode == "tampered_digest":
        value["binding_sha256"] = "0" * 64
    if rank == 1 and mode == "policy":
        value["gpu_execution_allowed"] = True
    return value


def _worker(rank, port, mode, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2)
    try:
        try:
            answer = require_native_binding_consensus(_fake(rank, mode))
            queue.put((rank, "PASS", answer["status"]))
        except DDPNativeBindingConsensusError as exc:
            queue.put((rank, "REFUSED", str(exc)))
    finally:
        dist.destroy_process_group()


class NativeBindingConsensusTests(unittest.TestCase):
    def run_pair(self, mode):
        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        mp.spawn(_worker, args=(_port(), mode, queue),
                 nprocs=2, join=True)
        return {rank: (state, msg) for rank, state, msg
                in (queue.get() for _ in range(2))}

    def test_identical_constructor_reviews_agree_only_without_authority(self):
        result = self.run_pair("same")
        self.assertEqual({v[0] for v in result.values()}, {"PASS"})
        self.assertTrue(all("REVIEW_ONLY" in v[1] for v in result.values()))

    def test_mismatched_constructor_arm_forged_digest_or_policy_refused(self):
        for mode in ("constructor", "arm", "tampered_digest", "policy"):
            with self.subTest(mode=mode):
                result = self.run_pair(mode)
                self.assertEqual({v[0] for v in result.values()}, {"REFUSED"})


if __name__ == "__main__":
    unittest.main()
