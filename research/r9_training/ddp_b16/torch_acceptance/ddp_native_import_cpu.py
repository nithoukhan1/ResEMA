"""Two-rank Gloo import-only probe contracts with mock native class imports.

No real GPU, Ultralytics import, checkpoint deserialization or training.
"""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from research.r9_training.ddp_b16.ddp_native_binding import (
    CLASS, SOURCE_MOD, SCHEMA,
)
from research.r9_training.ddp_b16.ddp_native_import_probe import (
    DDPNativeImportProbeError, collective_native_import_probe,
)
from research.r9_training.ddp_b16.ddp_verified_admission import (
    RECIPE, sha_object,
)


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _binding(rank, mode):
    arm = "MV01" if mode == "cross_rank_drift" and rank == 1 else "MV00"
    identity = {
        "arm": arm, "run_id": "CPU_GLOO_IMPORT_FIXTURE",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": sha_object(RECIPE),
        "assignment_sha256": "b" * 64,
        "pretrained_sha256": "c" * 64,
        "stage_source_manifest_sha256": "d" * 64,
    }
    body = {
        "arm": arm, "identity": identity,
        "source_root": "/fixture/stage/source",
        "modules_root": "/fixture/stage/modules",
        "checkout_root": "/fixture/checkout",
        "trainer_class": CLASS[arm],
        "trainer_module": SOURCE_MOD[arm],
        "constructor": {
            "train_assignment_manifest": "/fixture/assignment.csv",
            "val_assignment_manifest": "/fixture/assignment.csv",
            "train_visibility_evidence": "/fixture/visibility.json",
            "pretrained_archive": "/fixture/early.zip",
            "pretrained_best": "/fixture/early.pt",
        },
        "overrides": {
            "batch": 16, "device": "0,1", "epochs": 100,
            "workers": 0, "amp": True, "resume": False,
        },
        "r0_record_sha256": "e" * 64,
        "stage_receipt_sha256": "f" * 64,
    }
    return {
        "schema": SCHEMA, **body, "binding_sha256": sha_object(body),
        "status": "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY",
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }


def _worker(rank, port, scenario, queue):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=2,
    )
    try:
        binding = _binding(rank, scenario)

        def fake_load(record):
            if scenario == "source_reject" and rank == 1:
                raise RuntimeError("simulated independently detected source tampering")
            return type(
                record["trainer_class"], (object,),
                {"__module__": record["trainer_module"]},
            )

        def fake_t4(_cuda):
            if scenario == "not_t4" and rank == 1:
                raise RuntimeError("simulated topology mismatch")
            return ["Tesla T4", "Tesla T4"]

        with patch(
            "research.r9_training.ddp_b16.ddp_native_import_probe.verify_two_t4_runtime",
            side_effect=fake_t4,
        ), patch(
            "research.r9_training.ddp_b16.ddp_native_import_probe.load_pinned_native_class",
            side_effect=fake_load,
        ):
            try:
                report = collective_native_import_probe(binding)
                queue.put((rank, "PASS", report))
            except DDPNativeImportProbeError as exc:
                queue.put((rank, "REFUSED", str(exc)))
    finally:
        dist.destroy_process_group()


class NativeImportProbeTests(unittest.TestCase):
    def _pair(self, scenario):
        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        mp.spawn(_worker, args=(_port(), scenario, queue),
                 nprocs=2, join=True)
        return {rank: (status, data)
                for rank, status, data in (queue.get() for _ in range(2))}

    def test_same_native_class_imported_on_both_ranks_without_training(self):
        output = self._pair("same")
        self.assertEqual({x[0] for x in output.values()}, {"PASS"})
        for status, report in output.values():
            self.assertFalse(report["native_train_invoked"])
            self.assertFalse(report["gpu_training_authorized"])
            self.assertFalse(report["official_training_authorized"])
            self.assertFalse(report["model_deserialization_performed"])
            self.assertEqual(report["b_test_access"], "NONE")
            self.assertEqual(report["visible_gpu_names"], ["Tesla T4", "Tesla T4"])
            self.assertEqual(report["process_group_backend"], "gloo")

    def test_rank_local_topology_source_or_binding_drift_fails_both(self):
        for scenario in ("not_t4", "source_reject", "cross_rank_drift"):
            with self.subTest(scenario=scenario):
                output = self._pair(scenario)
                self.assertEqual({x[0] for x in output.values()}, {"REFUSED"})


if __name__ == "__main__":
    unittest.main()
