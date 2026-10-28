"""CPU-only source-and-asset admission tests. Mock R0 is a fixture, NOT authority."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.r9_training.ddp_b16.ddp_verified_admission import (
    DDPAdmissionError, audit_existing_stage, verify_staged_admission,
    PRETRAINED, ASSIGNMENT, RECIPE,
)
from research.r9_training.ddp_b16.ddp_worker_stage import create_private_stage

ROOT = Path(__file__).resolve().parents[4]


def plan(stage):
    return {"stage_root": str(stage), "mode": "FRESH_FROM_EARLY",
            "arm": "MV00", "run_id": "fixture_only", "b_test_access": "NONE",
            "expected_git_head": "a" * 40}


def review():
    return {"historical_input_identity_verified": True,
            "historical_scientific_source_commit":
                "fa6a7675daae385a04c7b77e69a9de29f9e1b2f3",
            "historical_input_recipe_is_ddp_training_authority": False,
            "historical_pretrained_sha256": PRETRAINED,
            "historical_assignment_sha256": ASSIGNMENT,
            "historical_r0_record_sha256": "b" * 64,
            "arm": "MV00", "run_id": "fixture_only",
            "gpu_execution_allowed": False, "official_training_authorized": False}


class DDPAdmissionTests(unittest.TestCase):
    def test_stage_is_fully_rehashed_and_has_no_training_authority(self):
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t) / "stage"
            create_private_stage(ROOT, stage)
            record = verify_staged_admission(plan(stage), review())
            self.assertEqual(record["identity"]["ddp_release_sha"], "a" * 40)
            self.assertEqual(record["identity"]["assignment_sha256"], ASSIGNMENT)
            self.assertEqual(record["identity"]["pretrained_sha256"], PRETRAINED)
            self.assertEqual(RECIPE["global_batch"], 16)
            self.assertFalse(record["gpu_execution_allowed"])
            self.assertFalse(record["official_training_authorized"])
            self.assertEqual(record["b_test_access"], "NONE")

    def test_tampered_file_or_extra_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t) / "stage"
            create_private_stage(ROOT, stage)
            source = stage / "source/ultralytics/engine/trainer.py"
            source.write_bytes(source.read_bytes() + b"# unauthorized\n")
            with self.assertRaisesRegex(DDPAdmissionError, "BYTES_OR_FILESET_DRIFT"):
                audit_existing_stage(stage)
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t) / "stage"
            create_private_stage(ROOT, stage)
            (stage / "modules/extra.py").write_text("pass\n", "utf-8")
            with self.assertRaisesRegex(DDPAdmissionError, "BYTES_OR_FILESET_DRIFT"):
                audit_existing_stage(stage)

    def test_wrong_identity_legacy_authority_or_test_access_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t) / "stage"
            create_private_stage(ROOT, stage)
            wrong = review()
            wrong["historical_pretrained_sha256"] = "0" * 64
            with self.assertRaisesRegex(DDPAdmissionError, "R0_ASSET_PINS"):
                verify_staged_admission(plan(stage), wrong)
            wrong = review()
            wrong["historical_input_recipe_is_ddp_training_authority"] = True
            with self.assertRaisesRegex(DDPAdmissionError, "R0_SOURCE_STATUS"):
                verify_staged_admission(plan(stage), wrong)
            incorrect = plan(stage)
            incorrect["b_test_access"] = "READ"
            with self.assertRaisesRegex(DDPAdmissionError, "PLAN"):
                verify_staged_admission(incorrect, review())

    def test_duplicate_receipt_keys_and_false_authorization_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t) / "stage"
            create_private_stage(ROOT, stage)
            receipt = stage / "STAGE_RECEIPT.json"
            data = json.loads(receipt.read_text("utf-8"))
            data["official_training_authorized"] = True
            receipt.write_text(json.dumps(data), "utf-8")
            with self.assertRaisesRegex(DDPAdmissionError, "STAGE_NOT_EXECUTABLE"):
                audit_existing_stage(stage)
        with tempfile.TemporaryDirectory() as t:
            stage = Path(t) / "stage"
            create_private_stage(ROOT, stage)
            receipt = stage / "STAGE_RECEIPT.json"
            s = receipt.read_text("utf-8")
            receipt.write_text(s.replace('"status":', '"status": "dup", "status":', 1), "utf-8")
            with self.assertRaisesRegex(DDPAdmissionError, "DUPLICATE_JSON_KEY"):
                audit_existing_stage(stage)


if __name__ == "__main__":
    unittest.main()
