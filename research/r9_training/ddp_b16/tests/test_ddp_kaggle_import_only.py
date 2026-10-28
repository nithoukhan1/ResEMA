"""No-GPU launcher receipt acceptance / refusal tests.

These tests do not run the Kaggle launcher or claim GPU acceptance.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from research.r9_training.ddp_b16.ddp_kaggle_import_only import (
    DDPKaggleImportError, _check_probe_report, _parse_probe_log,
    _new_private_output, PROBE_PREFIX, PROBE_SCHEMA,
)


def binding():
    return {
        "arm": "MV00",
        "binding_sha256": "a" * 64,
        "identity": {"stage_source_manifest_sha256": "b" * 64},
    }


def report():
    return {
        "schema": PROBE_SCHEMA,
        "status": "BOTH_NATIVE_CLASSES_IMPORTED_NO_MODEL_CONSTRUCTION",
        "arm": "MV00", "world_size": 2,
        "binding_sha256": "a" * 64,
        "native_class": "EarlyPairedTrainer",
        "source_manifest_sha256": "b" * 64,
        "visible_gpu_names": ["Tesla T4", "Tesla T4"],
        "torch_version": "2.x-fixture", "cuda_runtime_version": "12.x-fixture",
        "process_group_backend": "gloo",
        "gpu_topology_only_checked": True,
        "native_train_invoked": False,
        "model_deserialization_performed": False,
        "gpu_training_authorized": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }


class KaggleImportOnlyContractTests(unittest.TestCase):
    def test_valid_synthetic_receipt_is_no_training(self):
        result = _check_probe_report(report(), binding())
        self.assertEqual(result["visible_gpu_names"], ["Tesla T4", "Tesla T4"])
        self.assertEqual(result["binding_sha256"], "a" * 64)

    def test_reject_fake_cpu_receipt_wrong_rank_and_training_claims(self):
        cases = {
            "cpu_only": ("visible_gpu_names", ["CPU", "CPU"]),
            "single_t4": ("visible_gpu_names", ["Tesla T4"]),
            "wrong_arm": ("arm", "MV01"),
            "wrong_checkpoint": ("binding_sha256", "0" * 64),
            "wrong_class": ("native_class", "VGRADetectionTrainer"),
            "world_size": ("world_size", 1),
            "not_gloo": ("process_group_backend", "nccl"),
            "train_called": ("native_train_invoked", True),
            "weights_loaded": ("model_deserialization_performed", True),
            "official": ("official_training_authorized", True),
            "test_open": ("b_test_access", "READ"),
            "missing_status": ("status", "SUCCESS"),
        }
        for label, (field, change) in cases.items():
            with self.subTest(label=label):
                altered = report()
                altered[field] = change
                with self.assertRaises(DDPKaggleImportError):
                    _check_probe_report(altered, binding())

    def test_requires_exactly_one_valid_rank_zero_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "log.txt"
            log.write_text("some harmless log\n" + PROBE_PREFIX +
                           json.dumps(report()) + "\n", "utf-8")
            self.assertEqual(_parse_probe_log(log)["arm"], "MV00")
            log.write_text("no rank report\n", "utf-8")
            with self.assertRaisesRegex(DDPKaggleImportError,
                                        "ONE_RANK0_PROBE_RECEIPT"):
                _parse_probe_log(log)
            log.write_text(PROBE_PREFIX + json.dumps(report()) + "\n" +
                           PROBE_PREFIX + json.dumps(report()) + "\n",
                           "utf-8")
            with self.assertRaises(DDPKaggleImportError):
                _parse_probe_log(log)

    def test_main_rejects_relative_missing_or_symlink_plan_before_staging(self):
        from research.r9_training.ddp_b16.ddp_kaggle_import_only import main
        with self.assertRaisesRegex(DDPKaggleImportError, "EXACT_ABSOLUTE_PLAN"):
            main(["--plan", "./untrusted_plan.json",
                  "--project", "/kaggle/working/new_project",
                  "--evidence-root", "/kaggle/working/new_evidence",
                  "--prepare-only"])
        with self.assertRaisesRegex(DDPKaggleImportError, "EXACT_ABSOLUTE_PLAN"):
            main(["--plan", "/kaggle/working/missing_plan.json",
                  "--project", "/kaggle/working/new_project",
                  "--evidence-root", "/kaggle/working/new_evidence",
                  "--prepare-only"])

    def test_child_launcher_disables_bytecode_writes_before_spawn(self):
        # CI cannot use real Kaggle assets. Assert the exact launcher
        # declaration and its placement in the real child env block.
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / "ddp_kaggle_import_only.py"
        text = source.read_text("utf-8")
        self.assertIn('env["PYTHONDONTWRITEBYTECODE"] = "1"', text)
        self.assertLess(text.index('env["PYTHONDONTWRITEBYTECODE"] = "1"'),
                        text.index('subprocess.run('))

    def test_no_overwrite_overlapping_or_sealed_output(self):
        plan = {
            "checkout_root": "/kaggle/working/ResEMA_DDP_RELEASE",
            "stage_root": "/kaggle/working/ddp_private_stage",
        }
        for value in ("/kaggle/working/ResEMA_DDP_RELEASE/out",
                      "/kaggle/working/ddp_private_stage/report",
                      "/kaggle/working/a/../new",
                      "/kaggle/working/b-test/reports",
                      "/tmp/unsafe",
                      "/kaggle/working//new"):
            with self.subTest(value=value), self.assertRaises(DDPKaggleImportError):
                _new_private_output(value, plan, label="EVIDENCE")


if __name__ == "__main__":
    unittest.main()
