"""MV00 epoch-64 externally pinned continuation notebook contract."""
import ast
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1]
PIN = "5976a4a3fe6e1b17d70687f0298dacb046198ac79ee74d08ba1184ed535c5a7a"

class Epoch64PinTest(unittest.TestCase):
    def test_fixed_checkpoint_and_native_restore_contract(self):
        notebook = json.loads((BASE / "kaggle_mv00_resume_epoch064_pinned.ipynb").read_text("utf-8"))
        code = "".join(notebook["cells"][1]["source"])
        ast.parse(code)
        first = ast.parse(code).body[0]
        worker = ast.literal_eval(first.value)
        ast.parse(worker)
        self.assertIn('PREVIOUS_OUTPUT_DATASET = "/kaggle/input/notebooks/nithoukhan/grazpedwri-dx-ddp-b16-training"', code)
        self.assertIn(f'PREVIOUS_MANIFEST_SHA256 = "{PIN}"', code)
        self.assertIn('EXPECTED_COMPLETED_EPOCH = 63', code)
        self.assertIn('EXPECTED_OPTIMIZER_UPDATES = 14647', code)
        self.assertIn('SOURCE_HEAD = "f440d329d4544ec08ad63c1972bcf9a8b8a7eed8"', code)
        self.assertIn('REAL_NATIVE_RESTORE_WRONG_EPOCH',worker)
        self.assertIn('REAL_NATIVE_OPTIMIZER_CURSOR_DRIFT',worker)
        self.assertIn('STRICT_NATIVE_RESUME_ACCEPTED=',worker)
        self.assertIn('MV00_CONTINUOUS_RESULT=',worker)
        self.assertIn('assert manifest["cursor"]["next_epoch"] == EXPECTED_COMPLETED_EPOCH',code)
        self.assertNotIn('PREVIOUS_OUTPUT_DATASET = "AUTO"',code)

if __name__ == "__main__":
    unittest.main()
