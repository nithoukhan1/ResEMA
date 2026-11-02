"""Offline static contracts for the new multi-epoch runner and notebook."""
import ast
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class DDPB16MultiEpochContractTests(unittest.TestCase):
    def test_worker_static_contract(self):
        source=(ROOT/"ddp_multiepoch_kaggle_worker.py").read_text("utf-8")
        ast.parse(source)
        for term in ("RESUME_FROM_DDP_B16","ddp_b16_g4_epoch_end",
                     "persist_epoch","persist_terminal","EpochBoundaryStop",
                     "dist.broadcast_object_list","torch.serialization.safe_globals",
                     "BUDGET_HOURS","b_test_access"):
            self.assertIn(term,source)
        self.assertNotIn("B-TEST evaluation",source)

    def test_notebook_embeds_same_worker_and_pins_source(self):
        note=json.loads((ROOT/"kaggle_mv00_continuous.ipynb").read_text("utf-8"))
        self.assertEqual(len(note["cells"]),2)
        code="".join(note["cells"][1]["source"])
        ast.parse(code)
        expected=(ROOT/"ddp_multiepoch_kaggle_worker.py").read_text("utf-8")
        namespace={}
        ast_tree=ast.parse(code)
        first=ast_tree.body[0]
        self.assertIsInstance(first,ast.Assign)
        self.assertEqual(ast.literal_eval(first.value),expected)
        self.assertIn("4b8707bb9c4f16f42fa8796efd9777127e1cad9b9b5055a629e96b55607129b5",code)
        self.assertIn("f440d329d4544ec08ad63c1972bcf9a8b8a7eed8",code)
        self.assertIn("--max_restarts",code)

if __name__=="__main__":
    unittest.main()
