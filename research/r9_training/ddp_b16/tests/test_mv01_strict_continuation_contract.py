"""Static regression test for MV01's authentic epoch-2 G4 continuation notebook."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = "8282d20fec970c60cfcc783c2e902fb456372c0dfab3a03fadffd0585ddb8d0b"

class MV01StrictContinuationContract(unittest.TestCase):
    def test_worker_notebook_identity_and_resume_guards(self):
        note = json.loads((ROOT / "kaggle_mv01_continuous_strict_resume.ipynb").read_text("utf8"))
        code = "".join(note["cells"][1]["source"])
        tree = ast.parse(code)
        worker = ast.literal_eval(tree.body[0].value)
        ast.parse(worker)
        self.assertEqual(worker, (ROOT / "ddp_mv01_multiepoch_worker_strict_resume.py").read_text("utf8"))
        for s in (
            PIN, 'EXPECTED_COMPLETED_EPOCH = 1', 'EXPECTED_OPTIMIZER_UPDATES = 667',
            '/kaggle/input/notebooks/abasitive/grazpedwri-dx-mv01-ddp-b16',
            'DDP_B16_EPOCH1_G4_CANDIDATE_V1', 'MV01_DDP_B16_CONTINUOUS_OUTPUT_V1',
            'MANIFEST_CURSOR_WRONG_RESUME_EPOCH', 'STRICT_NATIVE_RESUME_ACCEPTED=',
            'SOURCE_HEAD = "f440d329d4544ec08ad63c1972bcf9a8b8a7eed8"',
        ):
            self.assertIn(s, code)
        for s in (
            'identity["arm"] == "MV01"',
            'native.__name__ == "VGRADetectionTrainer"',
            '"mode": "RESUME_FROM_DDP_B16"',
            'REAL_NATIVE_RESTORE_WRONG_EPOCH',
            'REAL_NATIVE_OPTIMIZER_CURSOR_DRIFT',
            'STRICT_NATIVE_RESUME_ACCEPTED=',
            'MV01_CONTINUOUS_RESULT=',
        ):
            self.assertIn(s, worker)
        self.assertNotIn('MV00', code + worker)
        self.assertIn('if receipt["schema"] == "DDP_B16_EPOCH1_G4_CANDIDATE_V1":',code)
        self.assertIn('elif receipt["schema"] == "MV01_DDP_B16_CONTINUOUS_OUTPUT_V1":',code)

if __name__ == "__main__":
    unittest.main()
