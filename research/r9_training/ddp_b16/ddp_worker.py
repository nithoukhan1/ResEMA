"""Explicit DDP-B16 per-arm worker preflight (NO TRAINING AUTHORITY).

Each MV00/MV01 notebook eventually invokes its own authenticated 2-rank
torchrun child. This entrypoint deliberately offers only review/staging:
the original C1 G4 recovery enforces batch=8 and historical source IDs, so
DDP-B16 must never execute by simply reusing that recovery implementation.

Inputs are not inferred or downloaded. Historical R0 authenticates the
original dataset/checkpoint/visibility assets and their frozen hashes.
C1 source is verified in full before private, immutable-source staging.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from research.r9_training.ddp_b16.ddp_worker_stage import (
    C1_COMMIT, create_private_stage, verify_original_source,
)

SCHEMA = "VGRA_DDP_B16_WORKER_PLAN_REVIEW_V1"
PLAN_KEYS = {
    "schema", "arm", "run_id", "mode", "checkout_root",
    "expected_git_head", "r0_input_config", "stage_root", "b_test_access",
}
SHA40 = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class DDPWorkerError(RuntimeError):
    pass


def _kaggle_location(raw: str, prefix: str) -> str:
    from pathlib import PurePosixPath
    if type(raw) is not str or "\\" in raw or not raw.startswith(prefix + "/"):
        raise DDPWorkerError("DDP_WORKER_BAD_KAGGLE_LOCATION")
    value = PurePosixPath(raw)
    if (".." in value.parts or value.as_posix() != raw or
            any(c.lower() in {"test", "b-test", "b_test"} for c in value.parts)):
        raise DDPWorkerError("DDP_WORKER_NONCANONICAL_OR_TEST_PATH")
    return raw


def validate_plan(plan: Any) -> dict[str, Any]:
    """Only fixed non-executable FRESH model identities, no arbitrary args."""
    if type(plan) is not dict or set(plan) != PLAN_KEYS or plan.get("schema") != SCHEMA:
        raise DDPWorkerError("DDP_WORKER_PLAN_KEYS_OR_SCHEMA")
    if plan["arm"] not in ("MV00", "MV01"):
        raise DDPWorkerError("DDP_WORKER_ARM")
    if type(plan["run_id"]) is not str or not RUN_ID.fullmatch(plan["run_id"]):
        raise DDPWorkerError("DDP_WORKER_RUN_ID")
    if plan["mode"] != "FRESH_FROM_EARLY":
        raise DDPWorkerError("DDP_WORKER_RESUME_NOT_YET_IMPLEMENTED")
    if (type(plan["expected_git_head"]) is not str or
            not SHA40.fullmatch(plan["expected_git_head"])):
        raise DDPWorkerError("DDP_WORKER_RELEASE_SHA_REQUIRED")
    if plan["b_test_access"] != "NONE":
        raise DDPWorkerError("DDP_WORKER_BTEST_MUST_BE_SEALED")
    _kaggle_location(plan["checkout_root"], "/kaggle/working")
    _kaggle_location(plan["r0_input_config"], "/kaggle/input")
    _kaggle_location(plan["stage_root"], "/kaggle/working")
    if plan["stage_root"].startswith(plan["checkout_root"].rstrip("/") + "/"):
        raise DDPWorkerError("DDP_WORKER_STAGE_INSIDE_CHECKOUT")
    return {
        "schema": SCHEMA,
        "arm": plan["arm"],
        "run_id": plan["run_id"],
        "mode": plan["mode"],
        "model": "SCConv-Early" if plan["arm"] == "MV00" else "SCConv-Early+VGRA-V1",
        "world_size": 2,
        "global_batch": 16,
        "batch_per_rank": 8,
        "epochs": 100,
        "imgsz": 1024,
        "seed": 42,
        "b_test_access": "NONE",
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
    }


def _unique_json(p: Path) -> dict:
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise DDPWorkerError("DDP_WORKER_DUPLICATE_JSON_KEY")
            out[k] = v
        return out
    if not p.is_file() or p.is_symlink():
        raise DDPWorkerError("DDP_WORKER_PLAN_NOT_REGULAR")
    return json.loads(p.read_text("utf-8"), object_pairs_hook=unique)


def _read_only_r0(checkout: Path):
    f = (checkout / "research/r9_training/frozen_source/ultralytics/models"
         "/yolo/detect/d4e_r3_read_only_identity_preflight.py")
    if not f.is_file() or f.is_symlink():
        raise DDPWorkerError("DDP_WORKER_R0_MISSING")
    spec = importlib.util.spec_from_file_location("_ddp_b16_r0", f)
    if spec is None or spec.loader is None:
        raise DDPWorkerError("DDP_WORKER_R0_IMPORT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_worker_inputs(plan: dict[str, Any], *, stage_source: bool = False) -> dict:
    """CPU full-byte review. Does not deserialize model or execute train."""
    summary = validate_plan(plan)
    checkout = Path(plan["checkout_root"]).resolve(strict=True)
    proc = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if proc.stdout.strip() != plan["expected_git_head"]:
        raise DDPWorkerError("DDP_WORKER_CHECKOUT_HEAD_DRIFT")
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    )
    if status.stdout.strip():
        raise DDPWorkerError("DDP_WORKER_TRACKED_WORKTREE_NOT_CLEAN")
    verify_original_source(checkout)
    base = checkout / "research/r9_training"
    # Prior input identity, without claiming the historical batch-8 recipe
    # authorizes the DDP-B16 optimizer or recovery.
    r0 = _read_only_r0(checkout)
    record = r0.verify_inputs(
        _unique_json(Path(plan["r0_input_config"])),
        plan["arm"], plan["run_id"],
        "f2cc6e31f5f1be152f4dd7603a397e142963923f1d35de613eb7c61b168bf0c0",
        str(base / "R9_RELEASE_RECIPE_CANDIDATE_R1.json"),
    )
    if record.get("status") != "PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE":
        raise DDPWorkerError("DDP_WORKER_INPUT_PROVENANCE_FAILED")
    if record["identity"]["arm"] != plan["arm"]:
        raise DDPWorkerError("DDP_WORKER_ARM_IDENTITY_DRIFT")
    summary["historical_input_identity_verified"] = True
    summary["historical_r0_record_sha256"] = record["all_input_hashes_sha256"]
    summary["historical_pretrained_sha256"] = record["identity"]["pretrained_sha256"]
    summary["historical_assignment_sha256"] = record["identity"]["assignment_sha256"]
    summary["historical_scientific_source_commit"] = C1_COMMIT
    summary["historical_input_recipe_is_ddp_training_authority"] = False
    if stage_source:
        receipt = create_private_stage(checkout, plan["stage_root"])
        summary["private_source_stage_status"] = receipt["status"]
        summary["private_stage_adapted_sha256"] = receipt["ddp_adapted_files"]
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review-only", action="store_true")
    mode.add_argument("--review-and-stage", action="store_true")
    mode.add_argument("--attest-staged", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    # Unlike the future released worker, this stage has no train path.
    # Fail BEFORE loading any user plan or importing torch.
    if args.execute:
        raise DDPWorkerError("DDP_B16_EXECUTION_LOCKED_PENDING_DDP_G4_AND_CUDA_ACCEPTANCE")
    plan = _unique_json(Path(args.plan))
    report = review_worker_inputs(plan, stage_source=args.review_and_stage)
    if args.attest_staged:
        from research.r9_training.ddp_b16.ddp_verified_admission import verify_staged_admission
        report["ddp_candidate_admission"] = verify_staged_admission(plan, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
