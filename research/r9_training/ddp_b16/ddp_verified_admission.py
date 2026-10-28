"""Bind verified R0 and staged C1-derived bytes to DDP-B16 candidate identity.

This module is PRE-EXECUTION only: no torch import, training, deserialization,
permission to execute, GPU approval, or release authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .ddp_g4_recovery import validate_identity
from .ddp_worker_stage import C1_COMMIT, file_sha256, verify_original_source
from .native_source_adapter import BASE_BLOBS, create_candidate_sources

SHA64 = re.compile(r"^[0-9a-f]{64}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
PRETRAINED = "620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1"
ASSIGNMENT = "87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4"
STAGE_SCHEMA = "VGRA_DDP_B16_PRIVATE_SOURCE_STAGE_CANDIDATE_V1"
RECIPE = {
    "schema": "VGRA_DDP_B16_FIXED_CANDIDATE_RECIPE_V1",
    "global_batch": 16, "per_rank_max_images": 8, "world_size": 2,
    "steps_per_rank": 890, "imgsz": 1024, "epochs": 100,
    "seed": 42, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
    "momentum": 0.937, "weight_decay": 0.0005, "cos_lr": True,
    "nbs": 64, "workers": 0, "amp": True,
    "deterministic": True, "patience": 100, "cache": False,
    "compile": False, "rect": False, "multi_scale": 0.0,
    "fraction": 1.0, "single_cls": False, "mosaic": 0.0,
    "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "plots": False, "save": True, "val": True, "resume": False,
    "exist_ok": False, "b_test_access": "NONE",
}
ADMISSION_SCHEMA = "DDP_B16_STAGE_ATTESTATION_REVIEW_NOT_GPU_AUTHORITY_V1"


class DDPAdmissionError(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise DDPAdmissionError("DDP_ADMISSION_" + code)


def sha_object(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def strict_json(path):
    require(path.is_file() and not path.is_symlink(), "RECEIPT_FILE")
    def pairs_only_once(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "DUPLICATE_JSON_KEY")
            obj[key] = value
        return obj
    data = json.loads(path.read_text("utf-8"), object_pairs_hook=pairs_only_once)
    require(type(data) is dict, "RECEIPT_MAPPING")
    return data


def audit_tree(root, expected, count, name):
    require(root.is_dir() and not root.is_symlink(), name + "_DIRECTORY")
    require(type(expected) is dict and len(expected) == count, name + "_COUNT")
    for relative, digest in expected.items():
        require(type(relative) is str and relative != "" and
                not relative.startswith("/") and "\\" not in relative and
                all(c not in ("", ".", "..") for c in relative.split("/")),
                name + "_RELATIVE_PATH")
        require(type(digest) is str and SHA64.fullmatch(digest) is not None,
                name + "_DIGEST_FORMAT")
    actual = {}
    for p in root.rglob("*"):
        require(not p.is_symlink(), name + "_SYMLINK")
        if p.is_file():
            actual[p.relative_to(root).as_posix()] = file_sha256(p)
        else:
            require(p.is_dir(), name + "_SPECIAL_FILE")
    require(actual == expected, name + "_BYTES_OR_FILESET_DRIFT")


def audit_existing_stage(path, checkout_root):
    # Never allow the mutable stage receipt to be its own ultimate authority.
    # Reproduce the exact five patched files from authenticated C1 Git blobs,
    # and derive the remaining 331 + seven module hashes from a fixed manifest.
    checkout = Path(checkout_root).resolve(strict=True)
    verify_original_source(checkout)
    original_base = checkout / "research/r9_training"
    original_manifest = json.loads(
        (original_base / "code/R9_CANDIDATE_MANIFEST_R1.json").read_text("utf-8")
    )
    originals = {
        rel: (original_base / "frozen_source" / rel).read_bytes()
        for rel in BASE_BLOBS
    }
    patched = create_candidate_sources(originals)
    expected_source = dict(original_manifest["candidate_source_sha256"])
    expected_source.update({
        rel: hashlib.sha256(value).hexdigest()
        for rel, value in patched.items()
    })
    expected_modules = original_manifest["unchanged_integration_modules_sha256"]
    candidate = Path(path)
    require(candidate.is_absolute() and ".." not in candidate.parts,
            "STAGE_ABSOLUTE_PATH")
    require(not any(p.is_symlink() for p in (candidate, *candidate.parents)),
            "STAGE_SYMLINK")
    stage = candidate.resolve(strict=True)
    require(stage.is_dir(), "STAGE_DIR")
    receipt_path = stage / "STAGE_RECEIPT.json"
    r = strict_json(receipt_path)
    require(r.get("schema") == STAGE_SCHEMA and
            r.get("status") == "EXACT_SOURCE_STAGED_CPU_ONLY_NOT_GPU_AUTHORITY",
            "STAGE_SCHEMA")
    require(r.get("base_c1_commit") == C1_COMMIT and
            r.get("science_files_verified") == 336 and
            r.get("integration_modules_verified") == 7, "C1_INVENTORY")
    require(r.get("official_training_authorized") is False and
            r.get("gpu_execution_allowed") is False and
            r.get("gpu_used") is False and
            r.get("b_test_access") == "NONE", "STAGE_NOT_EXECUTABLE")
    source = r.get("stage_source_sha256")
    modules = r.get("stage_modules_sha256")
    require(source == expected_source and modules == expected_modules,
            "PINNED_C1_SOURCE_DERIVATION")
    audit_tree(stage / "source", source, 336, "SOURCE")
    audit_tree(stage / "modules", modules, 7, "MODULES")
    patches = r.get("ddp_adapted_files")
    require(type(patches) is dict and len(patches) == 5 and
            all(type(k) is str and source.get(k) == v for k, v in patches.items()),
            "FIVE_PATCH_SOURCE_MAP")
    return {
        "source_manifest_sha256": sha_object({"source": source, "modules": modules}),
        "stage_receipt_sha256": file_sha256(receipt_path),
    }


def verify_staged_admission(plan, r0_review):
    """Must be invoked AFTER review_worker_inputs verified pinned R0 bytes."""
    require(type(plan) is dict and type(r0_review) is dict, "ARG_TYPES")
    require(plan.get("mode") == "FRESH_FROM_EARLY" and
            plan.get("arm") in ("MV00", "MV01") and
            plan.get("b_test_access") == "NONE", "PLAN")
    require(r0_review.get("historical_input_identity_verified") is True and
            r0_review.get("historical_scientific_source_commit") == C1_COMMIT and
            r0_review.get("historical_input_recipe_is_ddp_training_authority") is False,
            "R0_SOURCE_STATUS")
    require(r0_review.get("arm") == plan.get("arm") and
            r0_review.get("run_id") == plan.get("run_id"), "R0_ARM_OR_RUN")
    require(r0_review.get("historical_pretrained_sha256") == PRETRAINED and
            r0_review.get("historical_assignment_sha256") == ASSIGNMENT,
            "R0_ASSET_PINS")
    r0_sha = r0_review.get("historical_r0_record_sha256")
    require(type(r0_sha) is str and SHA64.fullmatch(r0_sha) is not None,
            "R0_RECORD_DIGEST")
    head = plan.get("expected_git_head")
    require(type(head) is str and SHA40.fullmatch(head) is not None,
            "RELEASE_COMMIT")
    require(r0_review.get("gpu_execution_allowed") is False and
            r0_review.get("official_training_authorized") is False, "R0_NO_AUTHORITY")
    stage = audit_existing_stage(plan["stage_root"], plan["checkout_root"])
    identity = {
        "arm": plan["arm"], "run_id": plan["run_id"],
        "ddp_release_sha": head, "ddp_recipe_sha256": sha_object(RECIPE),
        "assignment_sha256": ASSIGNMENT, "pretrained_sha256": PRETRAINED,
        "stage_source_manifest_sha256": stage["source_manifest_sha256"],
    }
    validate_identity(identity)
    return {
        "schema": ADMISSION_SCHEMA,
        "identity": identity, "r0_record_sha256": r0_sha,
        "stage_receipt_sha256": stage["stage_receipt_sha256"],
        "status": "R0_AND_PRIVATE_STAGE_BYTES_VERIFIED_REVIEW_ONLY",
        "gpu_execution_allowed": False,
        "official_training_authorized": False, "b_test_access": "NONE",
    }
