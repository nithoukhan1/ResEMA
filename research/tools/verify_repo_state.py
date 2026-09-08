#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

BASE_COMMIT = "5a511729bc21c1b0998b7e6f4a84108fb4e309bf"
EXPECTED_REPO_FRAGMENT = "nithoukhan1/ResEMA"
REQUIRED = [
    "research/README.md",
    "research/00_governance/PROJECT_CHARTER.md",
    "research/00_governance/MASTER_EXECUTION_PLAN.md",
    "research/00_governance/FINAL_RESEARCH_PROTOCOL_V8.md",
    "research/00_governance/EVALUATION_STATISTICS_GENERALIZATION_V8.md",
    "research/00_governance/SOTA_COMPARISON_PLAN_V8.md",
    "research/01_provenance/CODE_BASELINE.md",
    "research/01_provenance/DATASET_PROVENANCE.md",
    "research/01_provenance/KAGGLE_RUNTIME_LOCK.md",
    "research/02_history/PROJECT_TRACKER.csv",
    "research/02_history/EXPERIMENT_REGISTRY.csv",
    "research/02_history/ARTIFACT_REGISTRY.csv",
    "research/02_history/RESULTS_MASTER.csv",
    "research/02_history/DECISION_LOG.md",
    "research/03_literature/Detection-Grazpedwri_RAW.csv",
    "research/06_diagnostics/D00_PLAN.md",
    "research/07_method/METHOD_BLUEPRINT_V7.md",
]

FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".tflite"}
MAX_RESEARCH_FILE_BYTES = 25 * 1024 * 1024

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()

def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def ok(msg: str) -> None:
    print(f"[OK]   {msg}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    try:
        root = Path(git("rev-parse", "--show-toplevel"))
    except Exception as exc:
        fail(f"Not inside a Git repository: {exc}")

    if Path.cwd().resolve() != root.resolve():
        print(f"[INFO] Git root: {root}")

    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok(f"Frozen base commit {BASE_COMMIT[:12]} is an ancestor of HEAD")
    except subprocess.CalledProcessError:
        fail("Current history does not descend from the frozen base commit")

    remote = git("remote", "get-url", "origin")
    if EXPECTED_REPO_FRAGMENT.lower() not in remote.lower():
        fail(f"Unexpected origin remote: {remote}")
    ok(f"Origin points to {EXPECTED_REPO_FRAGMENT}")

    status = git("status", "--porcelain")
    if status and not args.allow_dirty:
        fail("Working tree is dirty")
    ok("Working-tree policy satisfied")

    for rel in REQUIRED:
        if not (root / rel).is_file():
            fail(f"Missing required file: {rel}")
    ok(f"All {len(REQUIRED)} required research files are present")

    bad = []
    oversized = []
    research_root = root / "research"
    for p in research_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(str(p.relative_to(root)))
        if p.stat().st_size > MAX_RESEARCH_FILE_BYTES:
            oversized.append((str(p.relative_to(root)), p.stat().st_size))

    if bad:
        fail("Forbidden model artifacts inside Git research tree: " + ", ".join(bad))
    ok("No forbidden model artifacts found in research/")

    if oversized:
        fail("Oversized research files: " + ", ".join(f"{p}={n}" for p, n in oversized))
    ok("No research file exceeds 25 MiB")

    print("\nResearch repository verification PASSED.")

if __name__ == "__main__":
    main()
