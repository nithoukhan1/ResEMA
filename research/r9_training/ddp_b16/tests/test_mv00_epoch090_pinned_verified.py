"""MV00 verified epoch-90 Kaggle resume: fail-closed static contract."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = "2ebf42201f37bf9ab065552ed5113b672f22dad65863e4f8a06a22447daef217"

class MV00Epoch90VerifiedResume(unittest.TestCase):
    def test_checkpoint_verification_and_training_contract(self):
        notebook = json.loads(
            (ROOT / "kaggle_mv00_resume_epoch090_pinned_verified.ipynb").read_text("utf8")
        )
        self.assertEqual([cell["cell_type"] for cell in notebook["cells"]],
                         ["markdown", "code", "code"])
        preflight = "".join(notebook["cells"][1]["source"])
        code = "".join(notebook["cells"][2]["source"])
        ast.parse(preflight)
        tree = ast.parse(code)
        worker = ast.literal_eval(tree.body[0].value)
        ast.parse(worker)
        for check in (
            f'PREVIOUS_MANIFEST_SHA256 = "{PIN}"',
            'EXPECTED_COMPLETED_EPOCH = 89',
            'EXPECTED_OPTIMIZER_UPDATES = None',
            'EXPECTED_OPTIMIZER_UPDATES = receipt["optimizer_updates"]',
            'metadata.get("completed_epoch") == EXPECTED_COMPLETED_EPOCH',
            'metadata["optimizer_updates"] > 14647',
            'MV00 EPOCH-89 OFF-SESSION G4 AUTHENTICATION: PASS',
            'MV00_EPOCH89_INPUT_NOT_ATTACHED_OR_WRONG_VERSION',
            'MANIFEST_CURSOR_WRONG_RESUME_EPOCH',
            'STRICT_NATIVE_RESUME_ACCEPTED=',
            'SOURCE_HEAD = "f440d329d4544ec08ad63c1972bcf9a8b8a7eed8"',
        ):
            self.assertIn(check, code)
        for check in (
            PIN, "MV00 EPOCH-90 READ-ONLY PRE-TRAINING VERIFICATION: PASS",
            'cursor["completed_epoch"]==88', 'cursor["next_epoch"]==89',
            'cursor["optimizer_updates"]==updates', "GATE 10",
            "RANK_RECEIPT_IDENTITY",
        ):
            self.assertIn(check, preflight)
        self.assertNotIn("torch.load(", preflight)
        self.assertNotIn("trainer.train(", preflight)
        for guard in (
            'identity["arm"] == "MV00"',
            'native.__name__ == "EarlyPairedTrainer"',
            '"mode": "RESUME_FROM_DDP_B16"',
            "REAL_NATIVE_RESTORE_WRONG_EPOCH",
            "REAL_NATIVE_OPTIMIZER_CURSOR_DRIFT",
            "MV00_CONTINUOUS_RESULT=",
        ):
            self.assertIn(guard, worker)

if __name__ == "__main__":
    unittest.main()
