#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

EXPECTED_BRANCH = "research/combination-screen-01"
SOURCE_FREEZE_COMMIT = "8d3cfee47bb4de37b98eb78964f58a953ba4fd36"
SOURCE_FREEZE_CLOSURE_COMMIT = "ca221f0f45db014802fb7d1768826254aae4c857"
SOURCE_FREEZE_RECORD = "research/06_diagnostics/A12_D3_SOURCE_FREEZE.json"
SOURCE_FREEZE_RECORD_SHA256 = "24b34050764342d0ac2c2398ffdb80aedc59a4c151206765b203d01361ba4e19"
AUTH_PATH = "research/06_diagnostics/A12_D3_EXECUTION_AUTHORIZATION.json"
AUTH_CLOSURE_PATH = "research/06_diagnostics/A12_D3_EXECUTION_AUTHORIZATION_CLOSURE.json"
RUNNER_PATH = "research/06_diagnostics/A12_D3_OFFLINE_DIAGNOSTIC_RUNNER.py"
D2_ARCHIVE_SHA256 = "419ac71b1e42b791168a5ea24cf87d71b8a3d1eba9333d183a73009733056b93"
D2_ARCHIVE_BYTES = 43856067

SOURCE_HASHES = {
    "research/06_diagnostics/A12_D3_MATCHING_TAXONOMY_CONTRACT.md":
        "ebd8c059692f54842250e37e8a85f1e629ef0e35bc00a90cf6810977f008d9a1",
    "research/06_diagnostics/A12_D3_OFFLINE_DIAGNOSTIC_ENGINE.py":
        "dc8931c630f3d072af6b03b5f497833efc6f24a97ca68348e410114b2c1bb2f4",
    "research/06_diagnostics/test_A12_D3_OFFLINE_DIAGNOSTIC_ENGINE.py":
        "2807eac959506e8eb0af4789d254c277942a840edc7dbf818449040b7302ba6d",
    "research/06_diagnostics/A12_D3_OUTPUT_SCHEMA_AND_UNCERTAINTY_CONTRACT.md":
        "6e8e9c4d494921d51234e5f84451dd61039a7bfff86a3b8f6ef55d7071b2fd0a",
    "research/06_diagnostics/A12_D3_OFFLINE_DIAGNOSTIC_RUNNER.py":
        "7edae668a571f31274ca02be8f73d97ef49a1c0b4266c5f2a7a4464ece4116c4",
    "research/06_diagnostics/test_A12_D3_OFFLINE_DIAGNOSTIC_RUNNER.py":
        "3c5447a2438c9fc6fabca15806a3aaa092e90b411ed5350b6bc4f223a5fe63ec",
}


def die(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(
        cmd, cwd=cwd, text=True, capture_output=True,
        encoding="utf-8", errors="replace"
    )
    if check and p.returncode != 0:
        sys.stdout.write(p.stdout or "")
        sys.stderr.write(p.stderr or "")
        die(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run(["git", *args], repo, check=check).stdout.strip()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    p = run(["git", "merge-base", "--is-ancestor", ancestor, descendant], repo, check=False)
    return p.returncode == 0


def require_clean_remote(repo: Path) -> tuple[str, str]:
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        die(f"unexpected branch: {branch}")
    if git(repo, "status", "--porcelain"):
        die("worktree is not clean")

    run(
        [
            "git", "fetch", "origin",
            f"refs/heads/{EXPECTED_BRANCH}:refs/remotes/origin/{EXPECTED_BRANCH}"
        ],
        repo,
    )
    remote = git(repo, "rev-parse", f"origin/{EXPECTED_BRANCH}")
    if remote != head:
        die(f"remote/local HEAD mismatch: local={head} remote={remote}")
    return branch, head


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"cannot parse JSON {path}: {exc}")


def verify_authorization_state(repo: Path) -> tuple[dict, dict, str]:
    branch, head = require_clean_remote(repo)

    freeze_path = repo / SOURCE_FREEZE_RECORD
    auth_path = repo / AUTH_PATH
    closure_path = repo / AUTH_CLOSURE_PATH
    gate_path = Path(__file__).resolve()

    for p in (freeze_path, auth_path, closure_path):
        if not p.is_file():
            die(f"required governance file missing: {p}")

    if sha256_file(freeze_path) != SOURCE_FREEZE_RECORD_SHA256:
        die("source-freeze record SHA drift")

    freeze = load_json(freeze_path)
    auth = load_json(auth_path)
    closure = load_json(closure_path)

    if freeze.get("source_freeze_commit") != SOURCE_FREEZE_COMMIT:
        die("source-freeze commit binding drift")
    if freeze.get("status") != "SOURCE_FROZEN_PENDING_EXECUTION_AUTHORIZATION":
        die("unexpected source-freeze record status")

    if auth.get("status") != "AUTHORIZED_FOR_ONE_OFFLINE_DIAGNOSTIC_EXECUTION":
        die("D3 execution authorization is not active")
    if auth.get("source_freeze_commit") != SOURCE_FREEZE_COMMIT:
        die("authorization source-freeze binding drift")
    if auth.get("source_freeze_closure_commit") != SOURCE_FREEZE_CLOSURE_COMMIT:
        die("authorization source-freeze-closure binding drift")
    if auth.get("runner_path") != RUNNER_PATH:
        die("authorization runner-path drift")
    if auth.get("runner_sha256") != SOURCE_HASHES[RUNNER_PATH]:
        die("authorization runner SHA drift")
    if auth.get("d2_preservation_archive_sha256") != D2_ARCHIVE_SHA256:
        die("authorization D2 archive SHA drift")
    if auth.get("d2_preservation_archive_bytes") != D2_ARCHIVE_BYTES:
        die("authorization D2 archive byte-size drift")
    if auth.get("split_b_test_access") != "NONE":
        die("authorization test firewall drift")
    if auth.get("training_authorized") is not False:
        die("authorization unexpectedly permits training")
    if auth.get("validation_rerun_authorized") is not False:
        die("authorization unexpectedly permits validation rerun")

    gate_sha = sha256_file(gate_path)
    auth_sha = sha256_file(auth_path)

    if auth.get("execution_gate_sha256") != gate_sha:
        die("authorization gate SHA drift")
    if closure.get("authorization_json_sha256") != auth_sha:
        die("authorization-closure JSON SHA drift")
    if closure.get("execution_gate_sha256") != gate_sha:
        die("authorization-closure gate SHA drift")

    authorization_commit = str(closure.get("authorization_commit", ""))
    if not authorization_commit:
        die("authorization closure missing authorization_commit")
    if closure.get("status") != "AUTHORIZED_PENDING_ONE_OFFLINE_EXECUTION":
        die("authorization closure status is not executable")

    if not is_ancestor(repo, SOURCE_FREEZE_COMMIT, authorization_commit):
        die("source-freeze commit is not ancestor of authorization commit")
    if not is_ancestor(repo, authorization_commit, head):
        die("authorization commit is not ancestor of execution HEAD")

    # The first governed execution must happen from the authorization-closure
    # commit itself, before unrelated project changes.
    closure_commit = git(repo, "log", "-1", "--format=%H", "--", AUTH_CLOSURE_PATH)
    if closure_commit != head:
        die(
            "execution HEAD is not the authorization-closure commit; "
            f"closure_commit={closure_commit} head={head}"
        )
    parent = git(repo, "rev-parse", f"{head}^")
    if parent != authorization_commit:
        die(
            "authorization-closure parent is not the authorization commit; "
            f"parent={parent} authorization_commit={authorization_commit}"
        )

    for rel, expected_sha in SOURCE_HASHES.items():
        p = repo / rel
        if not p.is_file():
            die(f"frozen scientific source missing: {rel}")
        observed = sha256_file(p)
        if observed != expected_sha:
            die(f"frozen scientific source SHA drift: {rel}")

    return auth, closure, head


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A12-D3 source-bound authorized offline execution gate"
    )
    parser.add_argument(
        "--authorization-preflight-only",
        action="store_true",
        help="Verify Git/source/authorization only; do not touch the D2 archive.",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]

    print("=" * 100)
    print("A12-D3 AUTHORIZED EXECUTION GATE")
    print("SOURCE-BOUND OFFLINE DIAGNOSTICS ONLY - TRAINING NONE - TEST NONE")
    print("=" * 100)

    auth, closure, head = verify_authorization_state(repo)

    print(f"BRANCH={EXPECTED_BRANCH}")
    print(f"EXECUTION_HEAD={head}")
    print(f"SOURCE_FREEZE_COMMIT={SOURCE_FREEZE_COMMIT}")
    print(f"AUTHORIZATION_COMMIT={closure['authorization_commit']}")
    print(f"RUNNER_SHA256={SOURCE_HASHES[RUNNER_PATH]}")
    print("GOVERNANCE_GATE=PASS")

    if args.authorization_preflight_only:
        if args.archive is not None or args.output_dir is not None:
            die("preflight-only mode must not receive --archive or --output-dir")
        print("D2_ARCHIVE_READ=NONE")
        print("OFFLINE_DIAGNOSTIC_EXECUTION=NONE")
        print("CHECKPOINT_LOADING=NONE")
        print("VALIDATION_RERUN=NONE")
        print("TRAINING=NONE")
        print("TEST_ACCESS=NONE")
        print("READY_FOR_ONE_AUTHORIZED_OFFLINE_EXECUTION=TRUE")
        return 0

    if args.archive is None or args.output_dir is None:
        die("execution mode requires both --archive and --output-dir")

    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()

    if not archive.is_file():
        die(f"D2 archive not found: {archive}")
    if archive.stat().st_size != D2_ARCHIVE_BYTES:
        die("D2 archive byte-size drift")
    archive_sha = sha256_file(archive)
    if archive_sha != D2_ARCHIVE_SHA256:
        die("D2 archive SHA256 drift")

    # Keep generated outputs outside the scientific source worktree.
    try:
        output_dir.relative_to(repo)
        die("output directory must be outside the scientific Git worktree")
    except ValueError:
        pass

    if output_dir.exists() and any(output_dir.iterdir()):
        die(f"output directory must be absent or empty: {output_dir}")

    runner = repo / RUNNER_PATH
    cmd = [
        sys.executable,
        str(runner),
        "--archive", str(archive),
        "--output-dir", str(output_dir),
    ]

    print(f"D2_ARCHIVE_BYTES={archive.stat().st_size}")
    print(f"D2_ARCHIVE_SHA256={archive_sha}")
    print(f"OUTPUT_DIR={output_dir}")
    print("ARCHIVE_AUTHORITY_GATE=PASS")
    print("EXECUTING_FROZEN_OFFLINE_RUNNER=TRUE")

    p = subprocess.run(cmd, cwd=runner.parent)
    if p.returncode != 0:
        die(f"frozen D3 runner failed with exit code {p.returncode}")

    print("AUTHORIZED_OFFLINE_EXECUTION_GATE=PASS")
    print("CHECKPOINT_LOADING=NONE")
    print("VALIDATION_RERUN=NONE")
    print("TRAINING=NONE")
    print("TEST_ACCESS=NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
