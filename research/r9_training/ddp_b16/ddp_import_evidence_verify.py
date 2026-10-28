"""Strict offline integrity review of Kaggle DDP-B16 IMPORT-ONLY evidence.

This tool does not access GPUs, decrypt assets, import Ultralytics, access
B-TEST, deserialize model weights, or assert genuine Kaggle execution.
Receipt and log can be forged together: passing means INTERNAL consistency
only. Genuine physical T4 acceptance requires inspecting externally obtained
Kaggle run metadata alongside the original artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

from .ddp_kaggle_import_only import (
    SCHEMA as EVIDENCE_SCHEMA, DDPKaggleImportError,
    _check_probe_report, _parse_probe_log, sha_file,
)
from .ddp_native_binding import SCHEMA as BINDING_SCHEMA
from .ddp_verified_admission import RECIPE, sha_object

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ImportEvidenceError(RuntimeError):
    pass


def need(value: bool, reason: str) -> None:
    if not value:
        raise ImportEvidenceError("DDP_IMPORT_EVIDENCE_" + reason)


def strict_regular_file(raw: str | Path) -> Path:
    path = Path(raw)
    need(path.is_absolute() and ".." not in path.parts and
         not any(item.is_symlink() for item in (path, *path.parents)) and
         path.is_file(), "FILE_PATH")
    mode = path.stat()
    need(stat.S_ISREG(mode.st_mode) and mode.st_nlink == 1,
         "REGULAR_SINGLE_LINK")
    return path


def strict_json(path: Path) -> dict:
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, "DUPLICATE_JSON_KEY")
            value[key] = item
        return value
    result = json.loads(path.read_text("utf-8"), object_pairs_hook=reject_duplicates)
    need(type(result) is dict, "JSON_NOT_DICT")
    return result


def verify_evidence(
    evidence_path: str | Path, log_path: str | Path, *, expected_head: str,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify only INTERNAL provenance consistency and output firewall."""
    need(type(expected_head) is str and HEX40.fullmatch(expected_head) is not None,
         "EXTERNAL_HEAD_REQUIRED")
    receipt_file = strict_regular_file(evidence_path)
    log_file = strict_regular_file(log_path)
    evidence = strict_json(receipt_file)
    required = {
        "schema", "timestamp_utc", "status", "plan_sha256",
        "release_git_head", "arm", "run_id", "historical_r0_record_sha256",
        "stage_receipt_sha256", "binding", "source_stage_sha256", "probe",
        "probe_log_sha256", "error_code", "native_model_constructed",
        "pretrained_model_deserialized", "native_training_invoked",
        "optimizer_steps", "official_training_authorized",
        "gpu_training_authorized", "offsession_resume_verified",
        "b_test_access",
    }
    need(set(evidence) == required, "EVIDENCE_SCHEMA_KEYS")
    need(evidence["schema"] == EVIDENCE_SCHEMA and
         evidence["status"] == "LIVE_T4X2_IMPORT_ONLY_PASS_NOT_TRAINING",
         "UNACCEPTED_STATUS")
    need(evidence["release_git_head"] == expected_head,
         "GIT_HEAD_MISMATCH")
    need(evidence["arm"] in ("MV00", "MV01") and
         type(evidence["run_id"]) is str and
         re.fullmatch(r"[A-Za-z0-9_-]{1,128}", evidence["run_id"]) is not None,
         "ARM_OR_RUN")
    need(evidence["error_code"] is None and
         evidence["native_model_constructed"] is False and
         evidence["pretrained_model_deserialized"] is False and
         evidence["native_training_invoked"] is False and
         evidence["optimizer_steps"] == 0 and
         type(evidence["optimizer_steps"]) is int and
         evidence["official_training_authorized"] is False and
         evidence["gpu_training_authorized"] is False and
         evidence["offsession_resume_verified"] is False and
         evidence["b_test_access"] == "NONE", "TRAINING_OR_TEST_FIREWALL")
    for name in ("plan_sha256", "historical_r0_record_sha256",
                 "stage_receipt_sha256", "source_stage_sha256",
                 "probe_log_sha256"):
        value = evidence[name]
        need(type(value) is str and HEX64.fullmatch(value) is not None,
             "BAD_DIGEST_" + name)
    need(evidence["probe_log_sha256"] == sha_file(log_file),
         "RAW_LOG_DIGEST_MISMATCH")
    binding = evidence["binding"]
    need(type(binding) is dict and
         binding.get("schema") == BINDING_SCHEMA and
         binding.get("status") == "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY" and
         binding.get("arm") == evidence["arm"] and
         binding.get("source_manifest_sha256") ==
            evidence["source_stage_sha256"] and
         binding.get("ddp_recipe_sha256") == sha_object(RECIPE) and
         binding.get("gpu_execution_allowed") is False and
         binding.get("official_training_authorized") is False and
         binding.get("b_test_access") == "NONE", "CONSTRUCTOR_BINDING_DRIFT")
    sha = binding.get("binding_sha256")
    need(type(sha) is str and HEX64.fullmatch(sha) is not None,
         "CONSTRUCTOR_BINDING_DIGEST")
    skeleton = {
        "arm": evidence["arm"],
        "binding_sha256": sha,
        "identity": {"stage_source_manifest_sha256":
                     evidence["source_stage_sha256"]},
    }
    try:
        child = _check_probe_report(_parse_probe_log(log_file), skeleton)
    except DDPKaggleImportError as exc:
        raise ImportEvidenceError("DDP_IMPORT_EVIDENCE_BAD_CHILD_PROBE") from exc
    need(child == evidence["probe"], "CHILD_RECEIPT_DISAGREEMENT")
    plan_checked = False
    if plan_path is not None:
        plan_file = strict_regular_file(plan_path)
        plan = strict_json(plan_file)
        need(sha_file(plan_file) == evidence["plan_sha256"],
             "PLAN_CONTENT_HASH_MISMATCH")
        need(plan.get("expected_git_head") == expected_head and
             plan.get("arm") == evidence["arm"] and
             plan.get("run_id") == evidence["run_id"] and
             plan.get("b_test_access") == "NONE" and
             plan.get("mode") == "FRESH_FROM_EARLY",
             "PLAN_IDENTITY_OR_FIREWALL_DRIFT")
        plan_checked = True
    return {
        "schema": "DDP_B16_IMPORT_ONLY_OFFLINE_SELF_CONSISTENCY_V1",
        "status": "INTERNAL_CONSISTENCY_VERIFIED_NOT_GPU_ATTESTATION",
        "release_git_head": expected_head,
        "arm": evidence["arm"],
        "binding_sha256": sha,
        "evidence_sha256": sha_file(receipt_file),
        "raw_log_sha256": sha_file(log_file),
        "original_plan_bytes_checked": plan_checked,
        "actual_kaggle_gpu_execution_independently_attested": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--plan")
    args = p.parse_args(argv)
    verified = verify_evidence(args.evidence, args.log,
                               expected_head=args.expected_head,
                               plan_path=args.plan)
    print(json.dumps(verified, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
