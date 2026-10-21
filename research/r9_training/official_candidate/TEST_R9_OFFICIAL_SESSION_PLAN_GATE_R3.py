#!/usr/bin/env python3
"""CPU regression and negative tests for R9 dual-session plan gate R3."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate", root / "R9_OFFICIAL_SESSION_PLAN_GATE_R3.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
contract = json.loads((root / "R9_OFFICIAL_RELEASE_CONTRACT_R1.json").read_text(encoding="utf8"))


def fresh():
    return {
        "schema": gate.PLAN_SCHEMA, "status": "REVIEW_ONLY_NO_GPU",
        "run_group_id": "R9OFFICIAL_FUTURE_CANDIDATE", "official_training_authorized": False,
        "run_executable": False, "release_base_commit": contract["source"]["baseline_git_commit"],
        "recipe_sha256": contract["source"]["frozen_recipe_sha256"],
        "data_yaml_sha256": contract["source"]["runtime_data_yaml_sha256"],
        "total_epochs": 100, "train_split": "B-TRAIN", "val_split": "B-VAL", "test_split": "SEALED",
        "arms": {a: {"arm": a, "run_id": "R9OFFICIAL_FUTURE_CANDIDATE_" + a,
                      "physical_gpu": g, "mode": "FRESH_FROM_APPROVED_ORIGINAL",
                      "start_epoch_one_based": 1, "target_epoch_one_based": 5,
                      "checkpoint_input": None,
                      "initialization_source": "APPROVED_ORIGINAL_SCCONV_EARLY_PRETRAINED"}
                 for a, g in gate.ARM_MAP.items()}}


def resume():
    plan = fresh()
    for a in plan["arms"]:
        s = plan["arms"][a]
        s.update(mode="RESUME_VERIFIED_OFFSITE", start_epoch_one_based=6,
                 target_epoch_one_based=10,
                 initialization_source="AUTHENTICATED_OWN_RUN_CAPSULE")
        s["checkpoint_input"] = {
            "archive_path": f"/kaggle/input/private/{a}.zipblob",
            "archive_sha256": "a" * 64,
            "sidecar_path": f"/kaggle/input/private/{a}.sha256",
            "external_attestation_path": f"/kaggle/input/private/{a}.json",
            "completed_epoch_zero_based": 4,
            "terminal": False, "b_test_access": "NONE", "source_run_id": s["run_id"]}
    return plan


class Tests(unittest.TestCase):
    def accepts(self, plan):
        return gate.validate(contract, plan)
    def rejects(self, plan):
        with self.assertRaises(RuntimeError): self.accepts(plan)
    def test_fresh(self):
        self.assertEqual(self.accepts(fresh())["status"], "PASS_STATIC_CONTRACT_NOT_EXECUTABLE")
    def test_resumed(self):
        self.assertEqual(self.accepts(resume())["modes_by_arm"]["MV00"], "RESUME_VERIFIED_OFFSITE")
    def test_terminal_plan_is_only_review(self):
        x = resume()
        for s in x["arms"].values(): s["target_epoch_one_based"] = 100
        self.assertTrue(all(self.accepts(x)["terminal_requires_separate_native_validation"].values()))
    def test_authorization_rejected(self):
        x = fresh(); x["official_training_authorized"] = True; self.rejects(x)
    def test_executable_rejected(self):
        x = fresh(); x["run_executable"] = True; self.rejects(x)
    def test_single_arm_rejected(self):
        x = fresh(); del x["arms"]["MV01"]; self.rejects(x)
    def test_gpu_swap_rejected(self):
        x = fresh(); x["arms"]["MV00"]["physical_gpu"] = 1; self.rejects(x)
    def test_rehearsal_pretrain_rejected(self):
        x = fresh(); x["arms"]["MV01"]["initialization_source"] = "R9_REHEARSAL_EPOCH2"; self.rejects(x)
    def test_resume_without_attestation_rejected(self):
        x = resume(); x["arms"]["MV01"]["checkpoint_input"]["external_attestation_path"] = None; self.rejects(x)
    def test_terminal_resume_rejected(self):
        x = resume(); x["arms"]["MV00"]["checkpoint_input"]["terminal"] = True; self.rejects(x)
    def test_btest_reference_rejected(self):
        x = resume(); x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = "/kaggle/input/B-TEST/model.zip"; self.rejects(x)
    def test_wrong_epoch_cursor_rejected(self):
        x = resume(); x["arms"]["MV00"]["checkpoint_input"]["completed_epoch_zero_based"] = 3; self.rejects(x)
    def test_stale_recipe_rejected(self):
        x = fresh(); x["recipe_sha256"] = "0" * 64; self.rejects(x)
    def test_run_id_drift_rejected(self):
        x = resume(); x["arms"]["MV01"]["checkpoint_input"]["source_run_id"] = "ANOTHER_RUN"; self.rejects(x)
    def test_independent_arm_modes_allowed(self):
        x = resume(); x["arms"]["MV00"] = fresh()["arms"]["MV00"]
        self.assertEqual(self.accepts(x)["modes_by_arm"]["MV00"], "FRESH_FROM_APPROVED_ORIGINAL")
    def test_independent_epoch_targets_allowed(self):
        x = fresh(); x["arms"]["MV00"]["target_epoch_one_based"] = 10
        self.assertEqual(self.accepts(x)["targets_by_arm"]["MV01"], 5)
    def test_early_stop_invalid(self):
        x = fresh(); x["arms"]["MV00"]["target_epoch_one_based"] = 101; self.rejects(x)
    def test_duplicate_json_keys_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "duplicated.json"; p.write_text('{"schema":1,"schema":2}')
            with self.assertRaises(RuntimeError): gate.strict_load(p)
    def test_kaggle_absolute_path_works_on_windows_semantics(self):
        # PureWindowsPath models the host-dependent behavior that broke R2.
        from pathlib import PurePosixPath, PureWindowsPath
        sample = "/kaggle/input/private/MV00.zipblob"
        self.assertFalse(PureWindowsPath(sample).is_absolute())
        self.assertTrue(PurePosixPath(sample).is_absolute())
        self.assertEqual(self.accepts(resume())["modes_by_arm"]["MV01"], "RESUME_VERIFIED_OFFSITE")

    def test_windows_drive_checkpoint_path_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = r"C:\kaggle\input\private\MV00.zipblob"
        self.rejects(x)

    def test_kaggle_working_not_offsite_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = "/kaggle/working/MV00.zip"
        self.rejects(x)

    def test_parent_traversal_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = "/kaggle/input/private/../outside.zip"
        self.rejects(x)

    def test_noncanonical_double_slash_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["sidecar_path"] = "/kaggle/input/private//MV00.sha256"
        self.rejects(x)

    def test_backslash_mixed_separator_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = r"/kaggle/input/private\MV00.zipblob"
        self.rejects(x)

    def test_double_root_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = "//kaggle/input/private/MV00.zipblob"
        self.rejects(x)

    def test_control_character_rejected(self):
        x = resume()
        x["arms"]["MV00"]["checkpoint_input"]["archive_path"] = "/kaggle/input/private/\x00.zip"
        self.rejects(x)

    def test_valid_datasets_mount_accepted(self):
        x = resume()
        for a in x["arms"]:
            x["arms"][a]["checkpoint_input"]["archive_path"] = "/kaggle/input/datasets/example/model.zipblob"
        self.assertEqual(self.accepts(x)["status"], "PASS_STATIC_CONTRACT_NOT_EXECUTABLE")

    def test_contract_override_rejected(self):
        c = copy.deepcopy(contract); c["official_training_authorized"] = True
        with self.assertRaises(RuntimeError): gate.validate(c, fresh())


if __name__ == "__main__": unittest.main(verbosity=2)
