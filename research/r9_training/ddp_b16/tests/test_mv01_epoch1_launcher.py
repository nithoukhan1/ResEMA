"""Static subprocess invocation contract for Kaggle MV01 epoch-1 acceptance."""
import ast
import json
from pathlib import Path
import unittest

FOLDER = Path(__file__).resolve().parents[1]

class MV01LauncherContract(unittest.TestCase):
    def test_parent_and_worker_file_script_invocations(self):
        nb = json.loads((FOLDER / "kaggle_mv01_epoch1_acceptance.ipynb").read_text("utf-8"))
        src = "".join(nb["cells"][1]["source"])
        tree = ast.parse(src)
        self.assertIsInstance(tree.body[0], ast.Assign)
        worker = ast.literal_eval(tree.body[0].value)
        ast.parse(worker)
        self.assertEqual(worker, (FOLDER / "ddp_gpu_epoch1_mv01.py").read_text("utf-8"))
        self.assertIn('sys.executable, str(worker_path),', src)
        self.assertNotIn('sys.executable, "-m",\n    str(worker_path)', src)
        self.assertIn('sys.executable, "-m", "torch.distributed.run"', worker)
        self.assertIn('"--max_restarts=0", str(Path(__file__).resolve())', worker)
        self.assertNotIn('"--module", str(Path(__file__).resolve())', worker)
        self.assertIn('"arm": "MV01"', src)
        self.assertIn('native.__name__ == "VGRADetectionTrainer"', worker)
        self.assertIn('f440d329d4544ec08ad63c1972bcf9a8b8a7eed8', src)
        self.assertIn('4dc26b7277d58ed0b1fbcc1f4fac68aaf7ad417951e735bc8be2b040764861bb', src)

if __name__ == "__main__":
    unittest.main()
