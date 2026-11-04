"""Static contract: MV01 pinned epoch-34 continuation is strictly hash-derived."""
import ast
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1]
PIN = "098d87afbb74717b6ee8c349d335a3b2d63ad11eb21f554a1ef67e65b16470a7"

class MV01Epoch34PinnedContract(unittest.TestCase):
    def test_strict_resume(self):
        nb = json.loads((BASE/"kaggle_mv01_resume_epoch034_sha_pinned.ipynb").read_text("utf8"))
        code = "".join(nb["cells"][1]["source"])
        parsed = ast.parse(code)
        worker = ast.literal_eval(parsed.body[0].value)
        ast.parse(worker)
        for val in (
            f'PREVIOUS_MANIFEST_SHA256 = "{PIN}"',
            'EXPECTED_COMPLETED_EPOCH = 33',
            'EXPECTED_OPTIMIZER_UPDATES = None',
            'EXPECTED_OPTIMIZER_UPDATES = receipt["optimizer_updates"]',
            '"MV01_DDP_B16_CONTINUOUS_OUTPUT_V1"',
            'MV01 EPOCH-33 OFF-SESSION G4 AUTHENTICATION: PASS',
            'metadata.get("source_release_sha") == SOURCE_HEAD',
            'metadata.get("completed_epoch") == EXPECTED_COMPLETED_EPOCH',
            'MANIFEST_CURSOR_WRONG_RESUME_EPOCH',
            'STRICT_NATIVE_RESUME_ACCEPTED=',
        ):
            self.assertIn(val, code)
        for val in (
            'native.__name__ == "VGRADetectionTrainer"',
            'REAL_NATIVE_RESTORE_WRONG_EPOCH',
            'REAL_NATIVE_OPTIMIZER_CURSOR_DRIFT',
            '"mode": "RESUME_FROM_DDP_B16"',
            'MV01_CONTINUOUS_RESULT=',
        ):
            self.assertIn(val, worker)
        self.assertNotIn('MV00', code + worker)

if __name__ == "__main__":
    unittest.main()
