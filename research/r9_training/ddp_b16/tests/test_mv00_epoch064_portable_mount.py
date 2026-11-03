"""Static guard for MV00 epoch-64 checkpoint mount discovery."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MV00Epoch64MountGuard(unittest.TestCase):
    def test_pinned_identity_and_no_training_before_checkpoint_select(self):
        path = ROOT / "kaggle_mv00_resume_epoch064_portable_mount.ipynb"
        notebook = json.loads(path.read_text("utf8"))
        code = "".join(notebook["cells"][1]["source"])
        tree = ast.parse(code)
        ast.parse(ast.literal_eval(tree.body[0].value))
        for s in (
          'PREVIOUS_MANIFEST_SHA256 = "5976a4a3fe6e1b17d70687f0298dacb046198ac79ee74d08ba1184ed535c5a7a"',
          'EXPECTED_COMPLETED_EPOCH = 63',
          'EXPECTED_OPTIMIZER_UPDATES = 14647',
          'SOURCE_HEAD = "f440d329d4544ec08ad63c1972bcf9a8b8a7eed8"',
          'metadata.get("run_id") == RUN_ID',
          'metadata.get("source_release_sha") == SOURCE_HEAD',
          'metadata.get("optimizer_updates") == EXPECTED_OPTIMIZER_UPDATES',
          'MV00_EPOCH63_INPUT_NOT_ATTACHED_OR_WRONG_VERSION',
          'STRICT_NATIVE_RESUME_ACCEPTED=',
        ):
            self.assertIn(s,code)
        self.assertNotIn('assert candidates_root.is_dir(), "SELECTED_NOTEBOOK_OUTPUT_MISSING"', code)

if __name__ == "__main__":
    unittest.main()
