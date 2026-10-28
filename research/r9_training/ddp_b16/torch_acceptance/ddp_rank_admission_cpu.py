"""True CPU/Gloo two-rank admission-consensus cases with fixture identities."""
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

from research.r9_training.ddp_b16.ddp_rank_admission import (
    DDPRankAdmissionError, require_review_consensus,
)
from research.r9_training.ddp_b16.ddp_verified_admission import (
    ADMISSION_SCHEMA, ASSIGNMENT, PRETRAINED, RECIPE, sha_object,
)


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fixture(rank, tamper):
    identity = {
        "arm": "MV00", "run_id": "CPU_FIXTURE",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": sha_object(RECIPE),
        "assignment_sha256": ASSIGNMENT,
        "pretrained_sha256": PRETRAINED,
        "stage_source_manifest_sha256": "b" * 64,
    }
    if rank == 1 and tamper == "identity":
        identity["stage_source_manifest_sha256"] = "f" * 64
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "R0_AND_PRIVATE_STAGE_BYTES_VERIFIED_REVIEW_ONLY",
        "identity": identity, "r0_record_sha256": "c" * 64,
        "stage_receipt_sha256": (
            "e" * 64 if rank == 1 and tamper == "receipt" else "d" * 64),
        "gpu_execution_allowed": rank == 1 and tamper == "policy",
        "official_training_authorized": False, "b_test_access": "NONE",
    }


def worker(rank, port_number, tamper, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",
                            init_method=f"tcp://127.0.0.1:{port_number}",
                            rank=rank, world_size=2)
    try:
        local_error = "simulated_bad_local_R0" if rank == 1 and tamper == "local" else None
        try:
            r = require_review_consensus(fixture(rank, tamper),
                                         local_error=local_error)
            queue.put((rank, "PASS", r["status"]))
        except DDPRankAdmissionError as exc:
            queue.put((rank, "REFUSED", str(exc)))
    finally:
        dist.destroy_process_group()


class DDPConsensusTests(unittest.TestCase):
    def run_pair(self, tamper):
        ctx = mp.get_context("spawn")
        q = ctx.SimpleQueue()
        mp.spawn(worker, args=(port(), tamper, q), nprocs=2, join=True)
        return dict(q.get() for _ in range(2))

    def test_matching_verified_fixture_identity_only(self):
        r = self.run_pair(None)
        self.assertEqual({v[0] for v in r.values()}, {"PASS"})
        self.assertTrue(all("REVIEW_ONLY" in v[1] for v in r.values()))

    def test_mismatch_and_local_failure_rejected_on_both_ranks(self):
        for change in ("identity", "receipt", "policy", "local"):
            with self.subTest(change=change):
                r = self.run_pair(change)
                self.assertEqual({v[0] for v in r.values()}, {"REFUSED"})


if __name__ == "__main__":
    unittest.main()
