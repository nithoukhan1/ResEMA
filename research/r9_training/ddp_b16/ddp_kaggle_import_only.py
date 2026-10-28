"""Kaggle T4x2 IMPORT-ONLY admission launcher. NOT a trainer.

The explicit --run-import-probe path stages independently authenticated
frozen C1-derived code and launches exactly one two-rank Gloo native CLASS
import probe. It NEVER calls trainer.train, constructs a trainer, loads
checkpoint tensors, creates an optimizer, evaluates B-VAL or opens B-TEST.
A stable user-supplied Git SHA is required by the underlying R0 verifier.
No CI or local synthetic test can be reported as live Kaggle acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .ddp_worker import _unique_json, review_worker_inputs, validate_plan
from .ddp_verified_admission import verify_staged_admission
from .ddp_native_binding import make_native_binding, safe_binding_summary, CLASS

SCHEMA = "DDP_B16_T4X2_IMPORT_ONLY_EVIDENCE_V1"
PROBE_SCHEMA = "DDP_B16_IMPORT_ONLY_TWO_RANK_REVIEW_V1"
PROBE_PREFIX = "DDP_B16_NATIVE_IMPORT_PROBE_ONLY="
HEX = re.compile(r"^[0-9a-f]{64}$")


class DDPKaggleImportError(RuntimeError):
    pass


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise DDPKaggleImportError("DDP_KAGGLE_IMPORT_" + reason)


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_private_output(raw: str, plan: dict, *, label: str) -> Path:
    """Reject existing, aliased, nested, sealed or source-containing outputs."""
    _need(type(raw) is str and raw.startswith("/kaggle/working/"),
          label + "_KAGGLE_ROOT")
    p = PurePosixPath(raw)
    _need(p.as_posix() == raw and ".." not in p.parts and
          "\\" not in raw and not any(
              part.casefold() in ("test", "b_test", "b-test") for part in p.parts
          ), label + "_CANONICAL")
    path = Path(raw)
    _need(not path.exists() and not path.is_symlink() and
          path.parent.is_dir() and
          not any(x.is_symlink() for x in (path, *path.parents)),
          label + "_NEW_REAL_PARENT")
    resolved = path.resolve(strict=False)
    for key in ("checkout_root", "stage_root"):
        forbidden = Path(plan[key]).resolve(strict=False)
        _need(not resolved.is_relative_to(forbidden) and
              not forbidden.is_relative_to(resolved),
              label + "_OVERLAPS_CODE")
    return path


def _check_probe_report(report: Any, binding: dict) -> dict:
    """Never accept an invented PASS phrase or a CPU-only mock as GPU proof."""
    _need(type(report) is dict and report.get("schema") == PROBE_SCHEMA and
          report.get("status") ==
          "BOTH_NATIVE_CLASSES_IMPORTED_NO_MODEL_CONSTRUCTION",
          "CHILD_REPORT_SCHEMA_OR_STATUS")
    _need(report.get("arm") == binding["arm"] and
          type(report.get("world_size")) is int and report["world_size"] == 2 and
          report.get("binding_sha256") == binding["binding_sha256"] and
          report.get("native_class") == CLASS[binding["arm"]] and
          report.get("source_manifest_sha256") ==
          binding["identity"]["stage_source_manifest_sha256"],
          "CHILD_IDENTITY_OR_ARM")
    names = report.get("visible_gpu_names")
    _need(type(names) is list and len(names) == 2 and
          all(type(name) is str and "T4" in name.upper() for name in names),
          "PHYSICAL_TWO_T4_EVIDENCE_REQUIRED")
    _need(report.get("process_group_backend") == "gloo" and
          report.get("gpu_topology_only_checked") is True and
          report.get("native_train_invoked") is False and
          report.get("model_deserialization_performed") is False and
          report.get("gpu_training_authorized") is False and
          report.get("official_training_authorized") is False and
          report.get("b_test_access") == "NONE", "NO_EXECUTION_FIREWALL")
    torch_version = report.get("torch_version")
    cuda_version = report.get("cuda_runtime_version")
    _need(type(torch_version) is str and bool(torch_version) and
          (cuda_version is None or type(cuda_version) is str),
          "RUNTIME_VERSION_EVIDENCE")
    return {
        "arm": report["arm"],
        "binding_sha256": report["binding_sha256"],
        "native_class": report["native_class"],
        "visible_gpu_names": list(names),
        "torch_version": torch_version,
        "cuda_runtime_version": cuda_version,
        "process_group_backend": "gloo",
    }


def _parse_probe_log(path: Path) -> dict:
    lines = path.read_text("utf-8", errors="replace").splitlines()
    records = [line[len(PROBE_PREFIX):] for line in lines
               if line.startswith(PROBE_PREFIX)]
    _need(len(records) == 1, "ONE_RANK0_PROBE_RECEIPT_REQUIRED")
    try:
        report = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise DDPKaggleImportError("DDP_KAGGLE_IMPORT_BAD_PROBE_JSON") from exc
    return report


def _make_receipt(*, plan: dict, plan_path: Path, source_record: dict,
                  binding: dict, mode: str, child: dict | None = None,
                  log_sha256: str | None = None, reason: str | None = None) -> dict:
    _need(mode in ("PREPARED", "IMPORT_ONLY_PASSED", "IMPORT_ONLY_FAILED"),
          "INTERNAL_RECEIPT_STATUS")
    result = {
        "schema": SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": {
            "PREPARED": "PINNED_SOURCE_AND_NATIVE_BINDING_PREPARED_ONLY",
            "IMPORT_ONLY_PASSED": "LIVE_T4X2_IMPORT_ONLY_PASS_NOT_TRAINING",
            "IMPORT_ONLY_FAILED": "T4X2_IMPORT_ONLY_NOT_ACCEPTED",
        }[mode],
        "plan_sha256": sha_file(plan_path),
        "release_git_head": plan["expected_git_head"],
        "arm": plan["arm"],
        "run_id": plan["run_id"],
        "historical_r0_record_sha256": source_record["historical_r0_record_sha256"],
        "stage_receipt_sha256": binding["stage_receipt_sha256"],
        "binding": safe_binding_summary(binding),
        "source_stage_sha256": binding["identity"]["stage_source_manifest_sha256"],
        "probe": child,
        "probe_log_sha256": log_sha256,
        "error_code": reason,
        "native_model_constructed": False,
        "pretrained_model_deserialized": False,
        "native_training_invoked": False,
        "optimizer_steps": 0,
        "official_training_authorized": False,
        "gpu_training_authorized": False,
        "offsession_resume_verified": False,
        "b_test_access": "NONE",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--evidence-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--prepare-only", action="store_true")
    selection.add_argument("--run-import-probe", action="store_true")
    args = parser.parse_args(argv)
    # Reject inherited torchrun rank context and wrong CUDA mapping before
    # expensive source hashing. Preparation alone does not inspect CUDA.
    if args.run_import_probe:
        _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1",
              "EXPLICIT_CUDA_0_1_REQUIRED")
        _need(not any(key in os.environ for key in
                      ("RANK", "WORLD_SIZE", "LOCAL_RANK")),
              "LAUNCHER_MUST_BE_OUTSIDE_TORCHRUN")
    plan_path = Path(args.plan)
    _need(plan_path.is_absolute() and ".." not in plan_path.parts and
          not any(p.is_symlink() for p in (plan_path, *plan_path.parents)) and
          plan_path.is_file(), "EXACT_ABSOLUTE_PLAN_FILE_REQUIRED")
    plan = _unique_json(plan_path)
    validate_plan(plan)
    project = _new_private_output(args.project, plan, label="PROJECT")
    evidence = _new_private_output(args.evidence_root, plan,
                                   label="EVIDENCE")
    _need(not project.is_relative_to(evidence) and
          not evidence.is_relative_to(project), "PROJECT_EVIDENCE_OVERLAP")

    # Original R0 independently authenticates all named assets, checkpoint,
    # source and development-only dataset YAML before any private stage write.
    source_record = review_worker_inputs(plan, stage_source=True)
    admission = verify_staged_admission(plan, source_record)
    config = _unique_json(Path(plan["r0_input_config"]))
    binding = make_native_binding(plan, admission, config, project=str(project))
    evidence.mkdir(mode=0o700)
    report_file = evidence / "IMPORT_ONLY_EVIDENCE.json"
    log_file = evidence / "TWO_RANK_IMPORT_ONLY.log"

    if args.prepare_only:
        report = _make_receipt(plan=plan, plan_path=plan_path,
                               source_record=source_record,
                               binding=binding, mode="PREPARED")
        with report_file.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2, sort_keys=True)
            out.write("\n")
        print("DDP_B16_PREPARED_ONLY_NOT_GPU_ACCEPTANCE")
        print("DDP_B16_EVIDENCE_SHA256=" + sha_file(report_file))
        return 0

    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc_per_node=2",
        "--max_restarts=0", "--module",
        "research.r9_training.ddp_b16.ddp_child_review",
        "--plan", str(plan_path), "--native-import-probe",
        "--project", str(project),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    checkout = str(Path(plan["checkout_root"]).resolve(strict=True))
    env["PYTHONPATH"] = checkout + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        with log_file.open("x", encoding="utf-8") as stream:
            child = subprocess.run(
                command, cwd=checkout, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=1800, check=False,
            )
        _need(child.returncode == 0, "TWO_RANK_PROCESS_FAILED")
        verified = _check_probe_report(_parse_probe_log(log_file), binding)
        report = _make_receipt(
            plan=plan, plan_path=plan_path, source_record=source_record,
            binding=binding, mode="IMPORT_ONLY_PASSED",
            child=verified, log_sha256=sha_file(log_file),
        )
        with report_file.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2, sort_keys=True)
            out.write("\n")
        print("DDP_B16_LIVE_T4X2_NATIVE_IMPORT_ONLY_PASS_NOT_TRAINING")
        print("DDP_B16_EVIDENCE_SHA256=" + sha_file(report_file))
        print("DDP_B16_LOG_SHA256=" + sha_file(log_file))
        return 0
    except (DDPKaggleImportError, subprocess.TimeoutExpired, OSError) as exc:
        report = _make_receipt(
            plan=plan, plan_path=plan_path, source_record=source_record,
            binding=binding, mode="IMPORT_ONLY_FAILED",
            log_sha256=sha_file(log_file) if log_file.is_file() else None,
            reason=type(exc).__name__ + ":" + str(exc)[:120],
        )
        with report_file.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2, sort_keys=True)
            out.write("\n")
        print("DDP_B16_IMPORT_ONLY_REFUSED_NO_TRAINING", file=sys.stderr)
        print("DDP_B16_EVIDENCE_SHA256=" + sha_file(report_file),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
