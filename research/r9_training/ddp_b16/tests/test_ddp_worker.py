"""CPU-only explicit worker policy checks; no assets/GPU required."""
from __future__ import annotations

import copy
import unittest

from research.r9_training.ddp_b16.ddp_worker import (
    DDPWorkerError, SCHEMA, main, validate_plan,
)


def plan(arm="MV00"):
    return {
        "schema": SCHEMA,
        "arm": arm,
        "run_id": "DDP_B16_T4X2_S42",
        "mode": "FRESH_FROM_EARLY",
        "checkout_root": "/kaggle/working/ResEMA_DDP_RELEASE",
        "expected_git_head": "a" * 40,
        "r0_input_config": "/kaggle/input/pinned-assets/r0.json",
        "stage_root": "/kaggle/working/ddp_b16_private_stage",
        "b_test_access": "NONE",
    }


class ExplicitDDPWorkerTests(unittest.TestCase):
    def test_pinned_two_rank_parameters_and_arm_mapping(self):
        for arm, expected in (("MV00", "SCConv-Early"),
                              ("MV01", "SCConv-Early+VGRA-V1")):
            summary = validate_plan(plan(arm))
            self.assertEqual(summary["model"], expected)
            self.assertEqual(summary["world_size"], 2)
            self.assertEqual(summary["global_batch"], 16)
            self.assertEqual(summary["batch_per_rank"], 8)
            self.assertEqual(summary["epochs"], 100)
            self.assertEqual(summary["imgsz"], 1024)
            self.assertEqual(summary["seed"], 42)
            self.assertEqual(summary["b_test_access"], "NONE")
            self.assertIs(summary["gpu_execution_allowed"], False)
            self.assertIs(summary["official_training_authorized"], False)

    def test_execution_is_locked_before_plan_is_read(self):
        with self.assertRaisesRegex(DDPWorkerError, "EXECUTION_LOCKED"):
            main(["--plan", "/file-that-does-not-exist", "--execute"])

    def test_no_extra_fields_or_user_training_overrides(self):
        for key, value in (("batch", 32), ("epochs", 200), ("device", "0,1"),
                           ("resume_checkpoint", "anything")):
            p = plan()
            p[key] = value
            with self.subTest(key=key), self.assertRaises(DDPWorkerError):
                validate_plan(p)

    def test_prohibit_wrong_arm_resume_and_btest(self):
        for key, value in (("arm", "MV02"),
                           ("mode", "RESUME_FROM_OUR_OWN_RUN"),
                           ("b_test_access", "READ_ONLY"),
                           ("expected_git_head", "research/vgra-ddp-b16"),
                           ("stage_root", "/kaggle/working/ResEMA_DDP_RELEASE/private"),
                           ("stage_root", "/kaggle/working/b-test/artifacts"),
                           ("r0_input_config", "/kaggle/working/unpinned.json"),
                           ("checkout_root", "/kaggle/input/repo")):
            p = plan()
            p[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(DDPWorkerError):
                validate_plan(p)

    def test_rejects_traversal_windows_and_noncanonical_paths(self):
        for key, value in (("stage_root", "/kaggle/working/a/../b"),
                           ("stage_root", "/kaggle/working/a//b"),
                           ("r0_input_config", "/kaggle/input/a\\b"),
                           ("r0_input_config", "relative.json")):
            p = plan()
            p[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(DDPWorkerError):
                validate_plan(p)


if __name__ == "__main__":
    unittest.main()
