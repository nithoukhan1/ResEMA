#!/usr/bin/env python3
"""CPU-only R9 official-session PLAN VALIDATOR (R3, review only).

This program never starts GPU work, never grants training authorization and never
accepts the historical rehearsal capsules as official starting checkpoints.
This is an additive plan contract for subsequent audited native-worker integration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

CONTRACT_SCHEMA = "DETECTION_R9_OFFICIAL_RELEASE_CONTRACT_PROPOSAL_R1"
PLAN_SCHEMA = "DETECTION_R9_OFFICIAL_DUAL_SESSION_PLAN_REVIEW_ONLY_R2"
ARM_MAP = {"MV00": 0, "MV01": 1}
TOP_KEYS = {"schema", "status", "run_group_id", "official_training_authorized", "run_executable", "release_base_commit", "recipe_sha256", "data_yaml_sha256", "total_epochs", "train_split", "val_split", "test_split", "arms"}
ARM_KEYS = {"arm", "run_id", "physical_gpu", "mode", "start_epoch_one_based", "target_epoch_one_based", "checkpoint_input", "initialization_source"}
CP_KEYS = {"archive_path", "archive_sha256", "sidecar_path", "external_attestation_path", "completed_epoch_zero_based", "terminal", "b_test_access", "source_run_id"}
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, code):
    if not condition:
        raise RuntimeError("R9_STEP2B_R3_FAIL_CLOSED_" + code)


def kaggle_offsite_path(value, arm, key):
    """Validate Kaggle POSIX input paths independent of the Windows host OS.

    This is only a syntactic static gate. It does not establish file existence,
    bytes, integrity, attestation authenticity, or permission to train.
    """
    require(type(value) is str and value != "", "CHECKPOINT_PATH_" + key + "_" + arm)
    require("\\" not in value and all(ord(ch) >= 32 and ord(ch) != 127 for ch in value),
            "CHECKPOINT_PATH_SYNTAX_" + key + "_" + arm)
    p = PurePosixPath(value)
    require(p.is_absolute() and p.parts[:3] == ("/", "kaggle", "input"),
            "CHECKPOINT_PATH_" + key + "_" + arm)
    require(p.as_posix() == value and ".." not in p.parts and len(p.parts) > 3,
            "CHECKPOINT_PATH_CANONICAL_" + key + "_" + arm)
    require(all(part.lower() not in {"b-test", "b_test", "test"} for part in p.parts),
            "BTEST_PATH_" + arm)


def strict_load(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), "JSON_INPUT_NOT_REGULAR")
    def no_duplicates(pairs):
        d = {}
        for k, v in pairs:
            require(k not in d, "JSON_DUPLICATE_KEY")
            d[k] = v
        return d
    return json.loads(p.read_text("utf-8"), object_pairs_hook=no_duplicates)


def validate(contract, plan):
    require(type(contract) is dict and contract.get("schema") == CONTRACT_SCHEMA, "R1_CONTRACT_SCHEMA")
    require(contract.get("status") == "ENGINEERING_CANDIDATE_NOT_AUTHORIZED", "CONTRACT_STATUS")
    require(contract.get("official_training_authorized") is False and contract.get("run_executable") is False, "R1_NOT_AUTHORIZED")
    require(type(plan) is dict and set(plan) == TOP_KEYS, "PLAN_TOP_KEYS")
    require(plan["schema"] == PLAN_SCHEMA and plan["status"] == "REVIEW_ONLY_NO_GPU", "PLAN_SCHEMA_OR_STATUS")
    require(plan["official_training_authorized"] is False and plan["run_executable"] is False, "PLAN_AUTHORITY")
    require(type(plan["run_group_id"]) is str and RUN_ID.fullmatch(plan["run_group_id"]) is not None, "RUN_GROUP_ID")
    require(plan["release_base_commit"] == contract["source"]["baseline_git_commit"], "GIT_BASE_ID")
    require(plan["recipe_sha256"] == contract["source"]["frozen_recipe_sha256"], "RECIPE_ID")
    require(plan["data_yaml_sha256"] == contract["source"]["runtime_data_yaml_sha256"], "DATA_ID")
    exp = contract["experiment"]
    require(type(plan["total_epochs"]) is int and plan["total_epochs"] == exp["epochs_per_arm"] == 100, "TOTAL_EPOCHS")
    require((plan["train_split"], plan["val_split"], plan["test_split"]) == ("B-TRAIN", "B-VAL", "SEALED"), "SPLIT_AND_TEST")
    arms = plan["arms"]
    require(type(arms) is dict and set(arms) == set(ARM_MAP), "TWO_ARMS_REQUIRED")
    modes = {}
    ids = []
    targets = {}
    for arm, gpu in ARM_MAP.items():
        s = arms[arm]
        require(type(s) is dict and set(s) == ARM_KEYS, "ARM_KEYS_" + arm)
        require(s["arm"] == arm and type(s["physical_gpu"]) is int and s["physical_gpu"] == gpu, "GPU_BINDING_" + arm)
        require(type(s["run_id"]) is str and RUN_ID.fullmatch(s["run_id"]) is not None, "RUN_ID_" + arm)
        require(s["run_id"].startswith(plan["run_group_id"] + "_" + arm), "RUN_GROUP_BIND_" + arm)
        ids.append(s["run_id"])
        mode = s["mode"]
        require(mode in ("FRESH_FROM_APPROVED_ORIGINAL", "RESUME_VERIFIED_OFFSITE"), "MODE_" + arm)
        modes[arm] = mode
        start = s["start_epoch_one_based"]
        target = s["target_epoch_one_based"]
        require(type(start) is int and type(target) is int and 1 <= start <= target <= 100, "CURSOR_" + arm)
        targets[arm] = target
        if mode == "FRESH_FROM_APPROVED_ORIGINAL":
            require(start == 1 and s["checkpoint_input"] is None, "FRESH_CURSOR_AND_RESUME_" + arm)
            require(s["initialization_source"] == "APPROVED_ORIGINAL_SCCONV_EARLY_PRETRAINED", "FRESH_PRETRAIN_" + arm)
        else:
            require(start >= 2 and s["initialization_source"] == "AUTHENTICATED_OWN_RUN_CAPSULE", "RESUME_INITIALIZATION_" + arm)
            cp = s["checkpoint_input"]
            require(type(cp) is dict and set(cp) == CP_KEYS, "CHECKPOINT_FIELDS_" + arm)
            require(cp["terminal"] is False and cp["b_test_access"] == "NONE", "TERMINAL_OR_BTEST_CHECKPOINT_" + arm)
            require(type(cp["completed_epoch_zero_based"]) is int and cp["completed_epoch_zero_based"] == start - 2, "RESUME_CURSOR_" + arm)
            require(cp["source_run_id"] == s["run_id"], "CHECKPOINT_OWN_RUN_ID_" + arm)
            require(type(cp["archive_sha256"]) is str and HEX64.fullmatch(cp["archive_sha256"]) is not None, "CHECKPOINT_SHA_" + arm)
            for key in ("archive_path", "sidecar_path", "external_attestation_path"):
                kaggle_offsite_path(cp[key], arm, key)
            # Path and receipt *contents* must be independently verified by the future GPU worker.
    require(len(set(ids)) == 2, "RUN_ID_COLLISION")
    # Independent workers can have different epoch cursors/modes after a partial failure.
    # Never require lockstep progress between GPU0/MV00 and GPU1/MV01.
    summary = {
        "schema": "R9_STEP2B_R3_CPU_PLAN_REVIEW_ONLY",
        "status": "PASS_STATIC_CONTRACT_NOT_EXECUTABLE",
        "both_arms": sorted(ARM_MAP),
        "modes_by_arm": modes,
        "targets_by_arm": targets,
        "terminal_requires_separate_native_validation": {arm: epoch == 100 for arm, epoch in targets.items()},
        "source_identity": plan["release_base_commit"],
        "resume_archive_byte_verification_performed": False,
        "pretrained_byte_verification_performed": False,
        "gpu_training_performed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True)
    p.add_argument("--plan", required=True)
    args = p.parse_args()
    summary = validate(strict_load(args.contract), strict_load(args.plan))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("REVIEW_ONLY; NO TRAINING; NO AUTHORIZATION; NO B-TEST")


if __name__ == "__main__":
    main()
