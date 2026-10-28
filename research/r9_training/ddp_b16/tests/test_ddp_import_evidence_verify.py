"""Pure CPU import-only evidence consistency tests; no GPU attestation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.r9_training.ddp_b16.ddp_import_evidence_verify import (
    ImportEvidenceError, verify_evidence,
)
from research.r9_training.ddp_b16.ddp_kaggle_import_only import (
    PROBE_PREFIX, PROBE_SCHEMA, SCHEMA as EVIDENCE_SCHEMA, sha_file,
)
from research.r9_training.ddp_b16.ddp_native_binding import (
    SCHEMA as BINDING_SCHEMA,
)
from research.r9_training.ddp_b16.ddp_verified_admission import RECIPE, sha_object


HEAD = "a" * 40


def fixture(root: Path):
    log = root / "TWO_RANK_IMPORT_ONLY.log"
    receipt = root / "IMPORT_ONLY_EVIDENCE.json"
    plan = root / "PLAN.json"
    plan.write_text(json.dumps({
        "expected_git_head": HEAD, "arm": "MV00",
        "run_id": "DUMMY", "mode": "FRESH_FROM_EARLY",
        "b_test_access": "NONE",
    }), encoding="utf-8")
    child = {
        "schema": PROBE_SCHEMA,
        "status": "BOTH_NATIVE_CLASSES_IMPORTED_NO_MODEL_CONSTRUCTION",
        "arm": "MV00", "world_size": 2,
        "binding_sha256": "b" * 64,
        "native_class": "EarlyPairedTrainer",
        "source_manifest_sha256": "c" * 64,
        "visible_gpu_names": ["Tesla T4", "Tesla T4"],
        "torch_version": "fixture-not-actual", "cuda_runtime_version": "fixture",
        "process_group_backend": "gloo",
        "gpu_topology_only_checked": True,
        "native_train_invoked": False,
        "model_deserialization_performed": False,
        "gpu_training_authorized": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }
    log.write_text(PROBE_PREFIX + json.dumps(child) + "\n", encoding="utf-8")
    proof = {
        "schema": EVIDENCE_SCHEMA,
        "timestamp_utc": "2026-10-28T00:00:00+00:00",
        "status": "LIVE_T4X2_IMPORT_ONLY_PASS_NOT_TRAINING",
        "plan_sha256": sha_file(plan),
        "release_git_head": HEAD, "arm": "MV00", "run_id": "DUMMY",
        "historical_r0_record_sha256": "d" * 64,
        "stage_receipt_sha256": "e" * 64,
        "binding": {
            "schema": BINDING_SCHEMA,
            "status": "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY",
            "arm": "MV00", "binding_sha256": "b" * 64,
            "source_manifest_sha256": "c" * 64,
            "ddp_recipe_sha256": sha_object(RECIPE),
            "gpu_execution_allowed": False,
            "official_training_authorized": False,
            "b_test_access": "NONE",
        },
        "source_stage_sha256": "c" * 64,
        "probe": {
            "arm": "MV00",
            "binding_sha256": "b" * 64,
            "native_class": "EarlyPairedTrainer",
            "visible_gpu_names": ["Tesla T4", "Tesla T4"],
            "torch_version": "fixture-not-actual",
            "cuda_runtime_version": "fixture",
            "process_group_backend": "gloo",
        },
        "probe_log_sha256": sha_file(log), "error_code": None,
        "native_model_constructed": False,
        "pretrained_model_deserialized": False,
        "native_training_invoked": False, "optimizer_steps": 0,
        "official_training_authorized": False,
        "gpu_training_authorized": False,
        "offsession_resume_verified": False,
        "b_test_access": "NONE",
    }
    receipt.write_text(json.dumps(proof), encoding="utf-8")
    return receipt, log, plan, proof, child


class EvidenceConsistencyTests(unittest.TestCase):
    def test_fixture_is_only_internal_consistency_not_real_gpu_proof(self):
        with tempfile.TemporaryDirectory() as t:
            e, l, p, _, _ = fixture(Path(t))
            report = verify_evidence(e, l, expected_head=HEAD, plan_path=p)
            self.assertTrue(report["original_plan_bytes_checked"])
            self.assertFalse(report["actual_kaggle_gpu_execution_independently_attested"])
            self.assertFalse(report["official_training_authorized"])
            self.assertEqual(report["b_test_access"], "NONE")

    def test_modified_raw_log_is_rejected_even_if_receipt_retained(self):
        with tempfile.TemporaryDirectory() as t:
            e, l, p, _, _ = fixture(Path(t))
            l.write_text(l.read_text("utf-8") + "forged extra\n", "utf-8")
            with self.assertRaisesRegex(ImportEvidenceError, "RAW_LOG_DIGEST"):
                verify_evidence(e, l, expected_head=HEAD, plan_path=p)

    def test_duplicate_json_and_false_training_flag_refused(self):
        with tempfile.TemporaryDirectory() as t:
            e, l, p, receipt, _ = fixture(Path(t))
            e.write_text('{"schema":"x","schema":"y"}', "utf-8")
            with self.assertRaisesRegex(ImportEvidenceError, "DUPLICATE_JSON_KEY"):
                verify_evidence(e, l, expected_head=HEAD)
            receipt["native_training_invoked"] = True
            e.write_text(json.dumps(receipt), "utf-8")
            with self.assertRaisesRegex(ImportEvidenceError, "TRAINING_OR_TEST_FIREWALL"):
                verify_evidence(e, l, expected_head=HEAD)

    def test_plan_or_release_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            e, l, p, _, _ = fixture(Path(t))
            with self.assertRaisesRegex(ImportEvidenceError, "GIT_HEAD_MISMATCH"):
                verify_evidence(e, l, expected_head="f" * 40)
            p.write_text(p.read_text("utf-8") + " ", "utf-8")
            with self.assertRaisesRegex(ImportEvidenceError, "PLAN_CONTENT_HASH"):
                verify_evidence(e, l, expected_head=HEAD, plan_path=p)

    def test_self_consistent_synthetic_log_cannot_be_called_independent_attestation(self):
        with tempfile.TemporaryDirectory() as t:
            e, l, _, _, _ = fixture(Path(t))
            result = verify_evidence(e, l, expected_head=HEAD)
            self.assertEqual(result["status"],
                             "INTERNAL_CONSISTENCY_VERIFIED_NOT_GPU_ATTESTATION")
            self.assertFalse(result["original_plan_bytes_checked"])


if __name__ == "__main__":
    unittest.main()
