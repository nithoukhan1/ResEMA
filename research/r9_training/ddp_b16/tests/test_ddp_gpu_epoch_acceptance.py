"""CPU refusal/interface checks for the bounded MV00 GPU acceptance entrypoint."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.r9_training.ddp_b16.ddp_gpu_epoch_acceptance import (
    RUN_SCHEMA, OUTPUT_SCHEMA, AcceptanceFailure, safe_output_path,
)


class GPUEpochAcceptanceContractTests(unittest.TestCase):
    def test_module_is_parseable_and_explicitly_bounded(self):
        path = Path(__file__).resolve().parents[1] / "ddp_gpu_epoch_acceptance.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("epoch == 0", source)
        self.assertIn("trainer.train()", source)
        self.assertIn("bind_candidate_fresh_state", source)
        self.assertIn("review_worker_inputs(plan, stage_source=True)", source)
        self.assertIn("NONOFFICIAL_EPOCH1_G4_SAVED_NOT_OFFSESSION_VERIFIED", source)
        self.assertIn("False", source)
        self.assertEqual(RUN_SCHEMA, "DDP_B16_FRESH_ONE_EPOCH_ACCEPTANCE_V1")
        self.assertEqual(OUTPUT_SCHEMA, "DDP_B16_EPOCH1_G4_CANDIDATE_V1")

    def test_rejects_unsafe_outputs_without_writes(self):
        for candidate in (
            "/kaggle/input/myoutput",
            "/kaggle/working/../test",
            "/kaggle/working/B-TEST",
            "/kaggle/working/B_TEST/weights",
            "/kaggle/working/safe/../../private",
            "C:\\Users\\me\\checkpoint",
            "/kaggle/working",
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(AcceptanceFailure):
                    safe_output_path(candidate, label="OUTPUT")

    def test_rejects_existing_output(self):
        with TemporaryDirectory() as tmp:
            # The check must fail for already existing targets before any writes.
            location = Path(tmp) / "g4"
            location.mkdir()
            with self.assertRaises(AcceptanceFailure):
                safe_output_path(str(location), label="OUTPUT")


if __name__ == "__main__":
    unittest.main()
