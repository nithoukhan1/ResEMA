"""Strict, repeatable MV00/MV01 native constructor binding (review-only).

Called only AFTER the frozen C1/R0 source and asset verifier and the full
private-source-stage auditor. This repeats critical per-path SHA checks to
bind the exact files to the native trainer constructor. It does not import
Ultralytics, deserialize a checkpoint, start CUDA, authorize training, or
grant permission to set a trainer readiness flag.

A native worker must use THIS resulting mapping unchanged; any user-provided
override, inferred path, B-TEST reference, or arbitrary checkpoint is rejected.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from .ddp_verified_admission import (
    ADMISSION_SCHEMA, ASSIGNMENT, PRETRAINED, RECIPE, audit_existing_stage,
    sha_object,
)
from .ddp_g4_recovery import validate_identity

SCHEMA = "VGRA_DDP_B16_EXACT_NATIVE_CONSTRUCTOR_REVIEW_V1"
YAML_SHA = "f2cc6e31f5f1be152f4dd7603a397e142963923f1d35de613eb7c61b168bf0c0"
ARCHIVE_SHA = "4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396"
VISIBILITY_SHA = "9de034fa7df668d17effa843674c097341af9dd6461963f9e2c454c5f358b289"
CLASS = {"MV00": "EarlyPairedTrainer", "MV01": "VGRADetectionTrainer"}
SOURCE_MOD = {
    "MV00": "ultralytics.models.yolo.detect.early_paired_trainer",
    "MV01": "ultralytics.models.yolo.detect.vgra_trainer",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DDPBindingError(RuntimeError):
    pass


def need(value: bool, reason: str) -> None:
    if not value:
        raise DDPBindingError("DDP_B16_BINDING_" + reason)


def _has_sealed_component(path: str) -> bool:
    return any(part.casefold() in {"test", "b-test", "b_test"} for
               part in re.split(r"[/\\]", path))


def verified_file(raw: str, digest: str) -> str:
    """Require a regular single-linked file, no symlink ancestors, pinned hash."""
    need(type(raw) is str and os.path.isabs(raw) and ".." not in Path(raw).parts
         and not _has_sealed_component(raw), "ASSET_PATH")
    p = Path(raw)
    need(not any(x.is_symlink() for x in (p, *p.parents)), "ASSET_SYMLINK")
    need(p.is_file(), "ASSET_MISSING")
    meta = p.stat()
    need(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1, "ASSET_FILE_TYPE")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    need(h.hexdigest() == digest, "ASSET_SHA256_MISMATCH")
    return str(p.resolve(strict=True))


def fixed_native_overrides(*, arm: str, run_id: str, yaml_path: str,
                           project: str) -> dict[str, Any]:
    """No arbitrary CLI training options: explicit pinned 100-epoch config only."""
    need(arm in CLASS and type(run_id) is str and
         re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is not None,
         "ARM_OR_RUN")
    need(type(yaml_path) is str and os.path.isabs(yaml_path)
         and yaml_path.lower().endswith((".yaml", ".yml")),
         "DATA_YAML_REQUIRED")
    need(type(project) is str and project.startswith("/kaggle/working/")
         and not _has_sealed_component(project), "PRIVATE_PROJECT")
    raw = PurePosixPath(project)
    need(raw.as_posix() == project and ".." not in raw.parts
         and "\\" not in project, "PROJECT_CANONICAL")
    # Project must be a NEW run namespace; the native trainer will own its
    # epoch-specific child. Do not change the parent C1 or source-stage tree.
    return {
        "data": yaml_path,
        "task": "detect",
        "imgsz": 1024,
        "batch": 16,
        "device": "0,1",
        "seed": 42,
        "deterministic": True,
        "epochs": 100,
        "patience": 100,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "nbs": 64,
        "cos_lr": True,
        "amp": True,
        "workers": 0,
        "cache": False,
        "compile": False,
        "rect": False,
        "multi_scale": 0.0,
        "fraction": 1.0,
        "single_cls": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "plots": False,
        "save": True,
        "val": True,
        "resume": False,
        "exist_ok": False,
        "project": project,
        "name": run_id + "_" + arm + "_DDP_B16_NONOFFICIAL",
    }


def make_native_binding(plan: Any, admission: Any, config: Any, *,
                        project: str) -> dict[str, Any]:
    """Reverify C1-derived stage and exact R0 asset bytes; return review spec.

    No mutation of the plan, source, checkpoint, config or dataset.
    """
    need(type(plan) is dict and type(admission) is dict and
         type(config) is dict, "INPUT_TYPES")
    need(plan.get("mode") == "FRESH_FROM_EARLY" and
         plan.get("b_test_access") == "NONE" and
         plan.get("arm") in CLASS, "FRESH_ONLY_AND_TEST_SEALED")
    need(admission.get("schema") == ADMISSION_SCHEMA and
         admission.get("status") ==
         "R0_AND_PRIVATE_STAGE_BYTES_VERIFIED_REVIEW_ONLY" and
         admission.get("gpu_execution_allowed") is False and
         admission.get("official_training_authorized") is False and
         admission.get("b_test_access") == "NONE", "ADMISSION_REVIEW_STATUS")
    identity = admission.get("identity")
    validate_identity(identity)
    need(identity["arm"] == plan["arm"] and
         identity["run_id"] == plan.get("run_id") and
         identity["ddp_release_sha"] == plan.get("expected_git_head") and
         identity["ddp_recipe_sha256"] == sha_object(RECIPE) and
         identity["assignment_sha256"] == ASSIGNMENT and
         identity["pretrained_sha256"] == PRETRAINED,
         "DDP_IDENTITY_DRIFT")
    stage = audit_existing_stage(plan["stage_root"], plan["checkout_root"])
    need(identity["stage_source_manifest_sha256"] ==
         stage["source_manifest_sha256"] and
         admission.get("stage_receipt_sha256") == stage["stage_receipt_sha256"],
         "STAGE_NOT_BOUND_TO_IDENTITY")
    need(set(config) == {"assets", "overlays", "r21_code",
                         "native_original", "runtime_yaml"}
         and type(config.get("assets")) is dict, "R0_CONFIG_KEYS")
    assets = config["assets"]
    need({"assignment_csv", "early_best_pt", "early_archive_zip",
          "visibility_counts_json"} <= set(assets), "MISSING_R0_ASSETS")
    dataset_yaml = verified_file(config["runtime_yaml"], YAML_SHA)
    assignments = verified_file(assets["assignment_csv"], ASSIGNMENT)
    visibility = verified_file(assets["visibility_counts_json"], VISIBILITY_SHA)
    archive = verified_file(assets["early_archive_zip"], ARCHIVE_SHA)
    pretrained = verified_file(assets["early_best_pt"], PRETRAINED)

    source = Path(plan["stage_root"]) / "source"
    modules = Path(plan["stage_root"]) / "modules"
    need(source.is_dir() and modules.is_dir() and
         not source.is_symlink() and not modules.is_symlink(),
         "PRIVATE_SOURCE_ROOTS")
    need(type(project) is str and not Path(project).exists()
         and not any(x.is_symlink() for x in (Path(project), *Path(project).parents)),
         "OUTPUT_ROOT_MUST_BE_NEW")
    checkout = Path(plan["checkout_root"]).resolve(strict=True)
    destination = Path(project).resolve(strict=False)
    need(not destination.is_relative_to(checkout) and
         not destination.is_relative_to(Path(plan["stage_root"]).resolve(strict=True)),
         "OUTPUT_MUST_NOT_MODIFY_CODE")
    overrides = fixed_native_overrides(
        arm=plan["arm"], run_id=plan["run_id"],
        yaml_path=dataset_yaml, project=project,
    )
    constructor = {
        "train_assignment_manifest": assignments,
        "val_assignment_manifest": assignments,
        "train_visibility_evidence": visibility,
        "pretrained_archive": archive,
        "pretrained_best": pretrained,
    }
    body = {
        "arm": plan["arm"],
        "identity": identity,
        "source_root": str(source),
        "modules_root": str(modules),
        # Authenticated checkout is needed to independently recreate all five
        # source patches immediately before importing the staged native code.
        "checkout_root": str(checkout),
        "trainer_class": CLASS[plan["arm"]],
        "trainer_module": SOURCE_MOD[plan["arm"]],
        "constructor": constructor,
        "overrides": overrides,
        "r0_record_sha256": admission["r0_record_sha256"],
        "stage_receipt_sha256": stage["stage_receipt_sha256"],
    }
    return {
        "schema": SCHEMA,
        **body,
        "binding_sha256": sha_object(body),
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
        "status": "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY",
    }


def safe_binding_summary(binding: dict) -> dict:
    """No raw local dataset paths in the rank-review output."""
    return {
        "schema": binding["schema"],
        "arm": binding["arm"],
        "binding_sha256": binding["binding_sha256"],
        "source_manifest_sha256":
            binding["identity"]["stage_source_manifest_sha256"],
        "ddp_recipe_sha256": binding["identity"]["ddp_recipe_sha256"],
        "status": binding["status"],
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }
