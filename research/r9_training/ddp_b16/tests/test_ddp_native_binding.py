"""CPU, mock-file-pin negative tests for fixed DDP-B16 constructor contract.

Mocking source and file checks is fixture-only; this is not R0 provenance
evidence or execution acceptance.
"""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from research.r9_training.ddp_b16.ddp_native_binding import (
    ARCHIVE_SHA, ASSIGNMENT, CLASS, DDPBindingError, PRETRAINED,
    SCHEMA, VISIBILITY_SHA, YAML_SHA, fixed_native_overrides,
    make_native_binding, safe_binding_summary,
)
from research.r9_training.ddp_b16.ddp_verified_admission import (
    ADMISSION_SCHEMA, RECIPE, sha_object,
)


def plan(arm="MV00"):
    return {
        "arm": arm, "run_id": "REVIEW_TWO_T4",
        "mode": "FRESH_FROM_EARLY",
        "b_test_access": "NONE",
        "expected_git_head": "a" * 40,
        "checkout_root": "/kaggle/working/exact_release",
        "stage_root": "/kaggle/working/exact_private_stage",
    }


def admission(arm="MV00"):
    identity = {
        "arm": arm, "run_id": "REVIEW_TWO_T4",
        "ddp_release_sha": "a" * 40,
        "ddp_recipe_sha256": sha_object(RECIPE),
        "assignment_sha256": ASSIGNMENT,
        "pretrained_sha256": PRETRAINED,
        "stage_source_manifest_sha256": "b" * 64,
    }
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "R0_AND_PRIVATE_STAGE_BYTES_VERIFIED_REVIEW_ONLY",
        "identity": identity, "r0_record_sha256": "c" * 64,
        "stage_receipt_sha256": "d" * 64,
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }


def config():
    assets = {
        "assignment_csv": "/kaggle/input/frozen/assignment.csv",
        "early_best_pt": "/kaggle/input/frozen/best.pt",
        "early_archive_zip": "/kaggle/input/frozen/archive.zip",
        "visibility_counts_json": "/kaggle/input/frozen/visibility.json",
    }
    return {
        "assets": assets, "overlays": {}, "r21_code": {},
        "native_original": "/kaggle/input/frozen/native.py",
        "runtime_yaml": "/kaggle/input/frozen/train_val.yaml",
    }


class NativeBindingTests(unittest.TestCase):
    def fake_stage(self):
        return {"source_manifest_sha256": "b" * 64,
                "stage_receipt_sha256": "d" * 64}

    def fake_file(self, path, digest):
        assert digest in (YAML_SHA, ASSIGNMENT, PRETRAINED, ARCHIVE_SHA, VISIBILITY_SHA)
        return path

    def test_fixed_recipe_and_arm_native_constructor(self):
        for arm, classname in CLASS.items():
            options = fixed_native_overrides(
                arm=arm, run_id="REVIEW_TWO_T4",
                yaml_path="/kaggle/input/frozen/train_val.yaml",
                project="/kaggle/working/new_run")
            self.assertEqual(options["imgsz"], 1024)
            self.assertEqual(options["batch"], 16)
            self.assertEqual(options["device"], "0,1")
            self.assertEqual(options["workers"], 0)
            self.assertFalse(options["resume"])
            self.assertTrue(options["amp"])
            self.assertEqual(options["epochs"], 100)
            self.assertNotIn("test", options)
            self.assertIn(arm, options["name"])

    def test_actual_binding_structure_with_mocked_file_checks(self):
        # No GRAZPEDWRI data or pretrained weights accessible on GitHub CI.
        for arm in CLASS:
            p = plan(arm)
            a = admission(arm)
            with patch("research.r9_training.ddp_b16.ddp_native_binding.audit_existing_stage",
                       return_value=self.fake_stage()), patch(
                "research.r9_training.ddp_b16.ddp_native_binding.verified_file",
                side_effect=self.fake_file), patch(
                "research.r9_training.ddp_b16.ddp_native_binding.Path.resolve",
                return_value=__import__("pathlib").Path("/kaggle/working/nonexistent")):
                result = make_native_binding(
                    p, a, config(), project="/kaggle/working/new_run")
            self.assertEqual(result["schema"], SCHEMA)
            self.assertEqual(result["trainer_class"], CLASS[arm])
            self.assertEqual(result["constructor"]["train_assignment_manifest"],
                             result["constructor"]["val_assignment_manifest"])
            self.assertEqual(result["binding_sha256"], sha_object({
                k: v for k, v in result.items() if k not in (
                    "schema", "binding_sha256", "gpu_execution_allowed",
                    "official_training_authorized", "b_test_access", "status")
            }))
            self.assertFalse(result["gpu_execution_allowed"])
            self.assertFalse(result["official_training_authorized"])
            brief = safe_binding_summary(result)
            self.assertNotIn("constructor", brief)
            self.assertEqual(brief["b_test_access"], "NONE")

    def test_bad_arm_resume_and_test_are_rejected_early(self):
        p, a, cfg = plan(), admission(), config()
        for key, value in (("mode", "RESUME_FROM_DDP_B16"),
                           ("arm", "MV02"), ("b_test_access", "READ")):
            corrupted = dict(p)
            corrupted[key] = value
            with self.subTest(key=key), self.assertRaises(DDPBindingError):
                make_native_binding(corrupted, a, cfg,
                                    project="/kaggle/working/new_run")
        a["identity"]["ddp_recipe_sha256"] = "f" * 64
        with self.assertRaisesRegex(DDPBindingError, "DDP_IDENTITY_DRIFT"):
            make_native_binding(p, a, cfg, project="/kaggle/working/new_run")

    def test_no_arbitrary_recipe_or_hidden_test_path(self):
        for path in ("/kaggle/working/b-test/run",
                     "/kaggle/working/a/../b",
                     "/kaggle/working/another\\name",
                     "/tmp/not_kaggle"):
            with self.subTest(path=path), self.assertRaises(DDPBindingError):
                fixed_native_overrides(
                    arm="MV00", run_id="REVIEW_TWO_T4",
                    yaml_path="/kaggle/input/frozen/train_val.yaml",
                    project=path)


if __name__ == "__main__":
    unittest.main()
