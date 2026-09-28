#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture(*args: str) -> str:
    return subprocess.check_output(
        list(args),
        cwd=ROOT,
        text=True,
    ).strip()


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_external/arch_corr_01/local_verification"))
    args = parser.parse_args()

    head = capture("git", "rev-parse", "HEAD")
    branch = capture("git", "branch", "--show-current")
    dirty = capture("git", "status", "--porcelain=v1", "--untracked-files=all")
    if head != args.expected_head:
        raise RuntimeError(f"HEAD mismatch: {head} != {args.expected_head}")
    if branch != "research/arch-corr-01b":
        raise RuntimeError(f"Wrong branch: {branch}")
    if dirty:
        raise RuntimeError("Architecture worktree must be clean before verification.\n" + dirty)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    expected_checkpoint_sha = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
    observed_checkpoint_sha = sha256_file(checkpoint)
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: {observed_checkpoint_sha} != {expected_checkpoint_sha}"
        )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tpsc = out / "ARCH_CORR_01B_TPSC_AUDIT.json"
    ema = out / "ARCH_CORR_01C_EMA_AUDIT.json"

    run([
        sys.executable, "-m", "pytest", "-o", "addopts=",
        "research/tests/test_arch_corr_01b_tpsc.py",
        "research/tests/test_arch_corr_01c_ema.py",
        "research/tests/test_arch_corr_01d_dysample.py",
        "-q",
    ])
    run([
        sys.executable,
        "research/tools/arch_corr_01b_transfer_audit.py",
        "--checkpoint", str(checkpoint),
        "--output", str(tpsc),
    ])
    run([
        sys.executable,
        "research/tools/arch_corr_01c_ema_audit.py",
        "--checkpoint", str(checkpoint),
        "--output", str(ema),
    ])

    import torch
    import ultralytics

    ultralytics_source = Path(ultralytics.__file__).resolve()
    if ROOT not in ultralytics_source.parents:
        raise RuntimeError(
            f"Ultralytics import is not from architecture worktree: {ultralytics_source}"
        )

    report = {
        "schema_version": "ARCH-CORR-01-local-verification-v1.0",
        "status": "PASS",
        "git": {"branch": branch, "head": head, "clean": True},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "ultralytics_source": str(ultralytics_source),
            "platform": platform.platform(),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": observed_checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
        },
        "artifacts": {
            "tpsc_audit": {"path": str(tpsc), "sha256": sha256_file(tpsc), "bytes": tpsc.stat().st_size},
            "ema_audit": {"path": str(ema), "sha256": sha256_file(ema), "bytes": ema.stat().st_size},
        },
        "training_started": False,
        "dataset_access": "NONE",
        "test_access": "NONE",
    }
    master = out / "ARCH_CORR_01_LOCAL_VERIFICATION.json"
    master.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("=" * 100)
    print("ARCH-CORR-01 LOCAL VERIFICATION")
    print("=" * 100)
    print(f"HEAD={head}")
    print(f"CHECKPOINT_SHA256={observed_checkpoint_sha}")
    print(f"TPSC_AUDIT_SHA256={report['artifacts']['tpsc_audit']['sha256']}")
    print(f"EMA_AUDIT_SHA256={report['artifacts']['ema_audit']['sha256']}")
    print(f"MASTER={master}")
    print(f"MASTER_SHA256={sha256_file(master)}")
    print("TRAINING_STARTED=FALSE")
    print("DATASET_ACCESS=NONE")
    print("TEST_ACCESS=NONE")
    print("ARCH_CORR_01_LOCAL_VERIFICATION=PASS")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
