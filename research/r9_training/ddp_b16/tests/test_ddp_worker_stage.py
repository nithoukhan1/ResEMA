"""CPU-only verified private-stage acceptance; no model/GPU/data access."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.r9_training.ddp_b16.ddp_worker_stage import (
    DDPStageError, create_private_stage, file_sha256, verify_original_source,
)
from research.r9_training.ddp_b16.native_source_adapter import BASE_BLOBS


ROOT = Path(__file__).resolve().parents[4]


class DDPPrivateSourceStageTests(unittest.TestCase):
    def test_complete_c1_inventory_and_private_five_file_stage(self):
        original_audit = verify_original_source(ROOT)
        self.assertEqual(original_audit["source_files_verified"], 336)
        self.assertEqual(original_audit["integration_modules_verified"], 7)
        before = {name: file_sha256(
            ROOT / "research/r9_training/frozen_source" / name
        ) for name in BASE_BLOBS}

        with tempfile.TemporaryDirectory() as t:
            dest = Path(t) / "fresh_ddp_private_source"
            receipt = create_private_stage(ROOT, dest)
            self.assertEqual(receipt["status"], "EXACT_SOURCE_STAGED_CPU_ONLY_NOT_GPU_AUTHORITY")
            self.assertIs(receipt["official_training_authorized"], False)
            self.assertIs(receipt["gpu_execution_allowed"], False)
            self.assertEqual(receipt["b_test_access"], "NONE")
            self.assertEqual(receipt["science_files_verified"], 336)
            self.assertEqual(receipt["integration_modules_verified"], 7)
            self.assertEqual(len(receipt["ddp_adapted_files"]), 5)
            on_disk = json.loads((dest / "STAGE_RECEIPT.json").read_text("utf8"))
            self.assertEqual(receipt, on_disk)
            for rel, digest in receipt["ddp_adapted_files"].items():
                self.assertEqual(file_sha256(dest / "source" / rel), digest)
                self.assertNotEqual(file_sha256(
                    ROOT / "research/r9_training/frozen_source" / rel
                ), digest)
            with self.assertRaises(DDPStageError):
                create_private_stage(ROOT, dest)  # cannot replace existing stage

        after = {name: file_sha256(
            ROOT / "research/r9_training/frozen_source" / name
        ) for name in BASE_BLOBS}
        self.assertEqual(before, after)

    def test_refuse_existing_and_checkout_destination(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            with self.assertRaisesRegex(DDPStageError, "NEW_ABSOLUTE"):
                create_private_stage(ROOT, p)
        target = ROOT / "research/r9_training/ddp_b16/__forbidden_stage__"
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(DDPStageError, "REFUSE_WRITE_INSIDE_CHECKOUT"):
            create_private_stage(ROOT, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
