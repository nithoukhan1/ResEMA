"""Review native AMP class MRO, topology, and immutable identity binding.

All trainer classes are fake; no pretrained deserialization, model
construction, datasets, Ultralytics import, or GPU training occurs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.r9_training.ddp_b16.ddp_native_bootstrap_candidate import (
    DDPNativeBootstrapError, assemble_amp_ready_native_class,
    bind_candidate_fresh_state, load_pinned_native_class,
    reauthenticate_private_stage_for_import,
    require_native_binding_identity, verify_two_t4_runtime,
)
from research.r9_training.ddp_b16.ddp_native_binding import (
    SCHEMA, CLASS, SOURCE_MOD,
)
from research.r9_training.ddp_b16.ddp_verified_admission import (
    RECIPE, sha_object, audit_existing_stage, DDPAdmissionError,
)
from research.r9_training.ddp_b16.ddp_worker_stage import create_private_stage


def sample(arm="MV00"):
    identity = {
        "arm": arm, "run_id": "CPU_ONLY",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": sha_object(RECIPE),
        "assignment_sha256": "b" * 64,
        "pretrained_sha256": "c" * 64,
        "stage_source_manifest_sha256": "d" * 64,
    }
    body = {
        "arm": arm, "identity": identity,
        "source_root": "/kaggle/working/stage/source",
        "modules_root": "/kaggle/working/stage/modules",
        "checkout_root": "/kaggle/working/pinned_checkout",
        "trainer_class": CLASS[arm],
        "trainer_module": SOURCE_MOD[arm],
        "constructor": {
            "train_assignment_manifest": "/fixture/assignment",
            "val_assignment_manifest": "/fixture/assignment",
            "train_visibility_evidence": "/fixture/visibility",
            "pretrained_archive": "/fixture/early.zip",
            "pretrained_best": "/fixture/early.pt",
        },
        "overrides": {
            "device": "0,1", "batch": 16, "epochs": 100,
            "resume": False, "amp": True, "workers": 0,
        },
        "r0_record_sha256": "e" * 64,
        "stage_receipt_sha256": "f" * 64,
    }
    return {
        "schema": SCHEMA, **body,
        "binding_sha256": sha_object(body),
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
        "status": "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY",
    }


class EarlyPairedTrainer:
    def __init__(self, **kwargs):
        self.overrides_seen = kwargs
        self.amp_flag_pre_native = self._r7c_amp_gate_ready


class FakeGPU:
    def __init__(self, count=2, names=("Tesla T4", "Tesla T4")):
        self.count, self.names = count, names
    def is_available(self):
        return self.count > 0
    def device_count(self):
        return self.count
    def get_device_name(self, idx):
        return self.names[idx]


class DDPNativeBootstrapTests(unittest.TestCase):
    def test_exact_class_name_and_amp_before_native_constructor(self):
        Impl = assemble_amp_ready_native_class(EarlyPairedTrainer,
                                                "EarlyPairedTrainer")
        obj = Impl(example="ok")
        self.assertTrue(obj.amp_flag_pre_native)
        self.assertEqual(type(obj).__name__, "EarlyPairedTrainer")
        self.assertEqual(obj.overrides_seen, {"example": "ok"})
        self.assertEqual(Impl.mro()[2], EarlyPairedTrainer)

    def test_topology_requires_real_two_rank_mapping_and_two_t4s(self):
        for rank in (0, 1):
            with patch.dict(os.environ, {
                "WORLD_SIZE": "2", "RANK": str(rank),
                "LOCAL_RANK": str(rank), "CUDA_VISIBLE_DEVICES": "0,1",
            }, clear=True):
                verify_two_t4_runtime(FakeGPU())
                for wrong in (FakeGPU(count=1), FakeGPU(names=("Tesla T4", "A100"))):
                    with self.assertRaises(DDPNativeBootstrapError):
                        verify_two_t4_runtime(wrong)
        with patch.dict(os.environ, {
            "WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }, clear=True):
            with self.assertRaises(DDPNativeBootstrapError):
                verify_two_t4_runtime(FakeGPU())

    def test_forged_constructor_and_recovery_flags_fail_closed(self):
        b = sample()
        require_native_binding_identity(b)
        corrupted = copy.deepcopy(b)
        corrupted["overrides"]["batch"] = 8
        with self.assertRaises(DDPNativeBootstrapError):
            require_native_binding_identity(corrupted)
        corrupted = copy.deepcopy(b)
        corrupted["identity"]["arm"] = "MV01"
        with self.assertRaises(DDPNativeBootstrapError):
            require_native_binding_identity(corrupted)

    def test_real_c1_source_rechecked_before_import_and_forged_receipt_rejected(self):
        # Actual 336 original source files + seven integration modules, not mocks.
        repo_root = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as temp_dir:
            stage = Path(temp_dir) / "private"
            create_private_stage(repo_root, stage)
            audited = audit_existing_stage(stage, repo_root)
            b = sample()
            b["source_root"] = str(stage / "source")
            b["modules_root"] = str(stage / "modules")
            b["checkout_root"] = str(repo_root)
            b["identity"]["stage_source_manifest_sha256"] = (
                audited["source_manifest_sha256"])
            b["stage_receipt_sha256"] = audited["stage_receipt_sha256"]
            b["binding_sha256"] = sha_object({
                k: v for k, v in b.items()
                if k not in ("schema", "binding_sha256", "status",
                             "gpu_execution_allowed", "official_training_authorized",
                             "b_test_access")
            })
            report = reauthenticate_private_stage_for_import(b)
            self.assertEqual(
                report["status"],
                "C1_DERIVED_SOURCE_REVERIFIED_BEFORE_NATIVE_IMPORT")
            self.assertFalse(report["gpu_execution_allowed"])

            # A forged receipt that agrees with a modified source must
            # still fail against independently recomputed frozen C1 bytes.
            target = stage / "source/ultralytics/engine/trainer.py"
            target.write_bytes(target.read_bytes() + b"# forged source\\n")
            receipt_file = stage / "STAGE_RECEIPT.json"
            receipt = json.loads(receipt_file.read_text("utf-8"))
            mutation_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            rel = "ultralytics/engine/trainer.py"
            receipt["stage_source_sha256"][rel] = mutation_sha
            receipt["ddp_adapted_files"][rel] = mutation_sha
            receipt_file.write_text(json.dumps(receipt), encoding="utf-8")
            with patch(
                "research.r9_training.ddp_b16.ddp_native_bootstrap_candidate.importlib.import_module",
                side_effect=AssertionError("import must not be attempted"),
            ):
                with self.assertRaisesRegex(
                    DDPAdmissionError, "PINNED_C1_SOURCE_DERIVATION"):
                    load_pinned_native_class(b)

    def test_native_import_disables_bytecode_before_stage_reauthentication(self):
        from research.r9_training.ddp_b16 import ddp_native_bootstrap_candidate as module
        binding = sample()
        observed = []
        prior = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = False
            def reject_before_import(_binding):
                observed.append(sys.dont_write_bytecode)
                raise DDPNativeBootstrapError("TEST_STOPS_BEFORE_IMPORT")
            with patch.object(module, "reauthenticate_private_stage_for_import",
                              side_effect=reject_before_import):
                with self.assertRaisesRegex(DDPNativeBootstrapError,
                                            "TEST_STOPS_BEFORE_IMPORT"):
                    module.load_pinned_native_class(binding)
            self.assertEqual(observed, [True])
        finally:
            sys.dont_write_bytecode = prior

    def test_candidate_fresh_state_does_not_grant_execution(self):
        with tempfile.TemporaryDirectory() as t:
            trainer_type = assemble_amp_ready_native_class(
                EarlyPairedTrainer, "EarlyPairedTrainer")
            trainer = trainer_type()
            trainer.world_size = 2
            trainer.ddp = False
            trainer.args = SimpleNamespace(batch=16, device="0,1", resume=False)
            bind_candidate_fresh_state(trainer, sample(), Path(t))
            self.assertEqual(trainer._ddp_b16_g4_identity["arm"], "MV00")
            self.assertIs(trainer._ddp_b16_g4_native_bound, True)
            self.assertIs(trainer._ddp_b16_verified_worker_ready, True)
            self.assertIs(trainer._ddp_b16_terminal_candidate_ready, False)
            self.assertEqual(trainer._ddp_b16_native_state_request["mode"],
                             "FRESH_FROM_EARLY")
            trainer.args.batch = 8
            with self.assertRaises(DDPNativeBootstrapError):
                bind_candidate_fresh_state(trainer, sample(), Path(t))


if __name__ == "__main__":
    unittest.main()
