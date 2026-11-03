"""Static safety tests for strict external epoch-31 Kaggle continuation."""
import ast, json, pathlib, unittest

HERE=pathlib.Path(__file__).resolve().parents[1]
class StrictResumeNotebookContract(unittest.TestCase):
    def test_strict_notebook_is_parseable_and_embeds_exact_worker(self):
        note=json.loads((HERE/"kaggle_mv00_continuous_strict_resume.ipynb").read_text("utf-8"))
        code="".join(note["cells"][1]["source"])
        tree=ast.parse(code)
        first=tree.body[0]
        self.assertIsInstance(first,ast.Assign)
        worker=ast.literal_eval(first.value)
        ast.parse(worker)
        self.assertEqual(worker, (HERE/"ddp_multiepoch_kaggle_worker_strict_resume.py").read_text("utf-8"))
        for word in ("EXPECTED_COMPLETED_EPOCH = 30",
                     "EXPECTED_OPTIMIZER_UPDATES = 7305",
                     "05f702c3d4b09eb6310a4c153dcfbb2ea0f63c1d819f7b5d398abd7f134fc6d9",
                     "WRONG_OR_AMBIGUOUS_SAVED_OUTPUT",
                     "MANIFEST_CURSOR_WRONG_RESUME_EPOCH",
                     "STRICT_PRE_TRAINING_RESUME_PASS",
                     "STRICT_NATIVE_RESUME_ACCEPTED="):
            self.assertIn(word,code)
        for word in ("NATIVE_G4_RESTORATION_NOT_CONFIRMED",
                     "REAL_NATIVE_RESTORE_WRONG_EPOCH",
                     "REAL_NATIVE_OPTIMIZER_CURSOR_DRIFT",
                     "RESUME_FROM_DDP_B16",
                     "BUDGET_HOURS"):
            self.assertIn(word,worker)
        self.assertIn('SOURCE_HEAD = "f440d329d4544ec08ad63c1972bcf9a8b8a7eed8"',code)
        self.assertIn('PREVIOUS_OUTPUT_DATASET = "AUTO"',code)
        self.assertNotIn('PREVIOUS_OUTPUT_DATASET = "/kaggle/input/datasets/awaishit/mv00-b16-acceptance-output"',code)

if __name__=="__main__":
    unittest.main()
