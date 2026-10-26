"""Build and verify the Chat28 GitHub research migration ZIP.

This packages every GIT-TRACKED working-tree file from the exact DDP
branch, source handoff, historical checkpoint registry and Git metadata
as TEXT. The earlier Chat1-26 30.5MB standalone archive, local Kaggle
checkpoints, and raw medical dataset are not in Git and are not included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(output_dir: Path) -> dict:
    repo = Path(__file__).resolve().parents[3]
    output_dir = output_dir.resolve()
    if output_dir == repo or repo in output_dir.parents:
        raise RuntimeError("Migration outputs must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    branch = git(repo, "branch", "--show-current").decode().strip()
    if branch != "research/vgra-ddp-b16":
        raise RuntimeError(f"Wrong branch: {branch}")
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("Tracked checkout is dirty")
    files = [s.decode("utf-8") for s in git(repo, "ls-files", "-z").split(b"\x00") if s]
    if len(files) < 300:
        raise RuntimeError(f"Unexpectedly small tracked source population: {len(files)}")
    members = {}
    for relative in files:
        src = repo / relative
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f"Missing or symlink tracked file: {relative}")
        if ".." in Path(relative).parts or Path(relative).is_absolute():
            raise RuntimeError(f"Unsafe path: {relative}")
        members["REPOSITORY_SNAPSHOT/" + relative] = src.read_bytes()
    for name, cmd in {
        "git_log_all.txt": ("log", "--all", "--date=iso-strict", "--format=%H|%ad|%an|%s"),
        "git_branches.txt": ("branch", "-avv"),
        "git_refs.txt": ("show-ref",),
        "git_status.txt": ("status", "--short"),
    }.items():
        members["GIT_PROVENANCE/" + name] = git(repo, *cmd)
    members["GIT_PROVENANCE/current_head.txt"] = (head + "\n").encode()
    members["GIT_PROVENANCE/source_c1_head.txt"] = (
        "fa6a7675daae385a04c7b77e69a9de29f9e1b2f3\n"
    ).encode()
    package_readme = (
        "Detection Project Chats1-27 for Chat28 — 2026-10-26\n"
        "This verified archive contains every tracked GitHub repository file "
        "from DDP branch and the Chat1-27 master handoff.\n"
        "Read REPOSITORY_SNAPSHOT/research/migrations/chat28_2026-10-26/"
        "00_READ_THIS_FIRST.md first.\n"
        "IMPORTANT LIMITATION: The earlier independent 30,515,326-byte "
        "historical Chat1-26 portable ZIP is NOT contained in this repository "
        "snapshot. It remains in Detection Project sources. Original Kaggle "
        "checkpoint binaries and raw medical data are also not included.\n"
        "Official MV00/MV01 100 epochs: 0/100 each; B-TEST SEALED.\n"
    )
    members["PACKAGE_README.txt"] = package_readme.encode()
    manifest = {
        "schema": "DETECTION_CHAT28_GITHUB_SNAPSHOT_PACKAGE_V1",
        "created_utc_date": "2026-10-26",
        "git_branch": branch,
        "git_head": head,
        "frozen_science_release": "fa6a7675daae385a04c7b77e69a9de29f9e1b2f3",
        "tested_pre_migration_ddp_code_sha": "1b1696d3d77cf327452bcd40cb236c05c91098a2",
        "upstream_historical_zip_included": False,
        "upstream_historical_zip_bytes": 30515326,
        "checkpoint_binaries_included": False,
        "dataset_images_included": False,
        "official_experiments_completed": {"MV00": 0, "MV01": 0},
        "b_test_access": "NONE",
        "member_sha256": {n: sha(d) for n, d in sorted(members.items())},
    }
    members["PACKAGE_MEMBER_MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    ).encode()
    basename = "DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT27_FOR_CHAT28_2026-10-26_PORTABLE"
    out = output_dir / (basename + ".zip")
    if out.exists():
        raise RuntimeError("Output ZIP already exists")
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as z:
        for filename in sorted(members):
            z.writestr(filename, members[filename])
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        names = set(z.namelist())
        if names != set(members):
            raise RuntimeError("ZIP member inventory mismatch")
        for filename, digest in manifest["member_sha256"].items():
            if sha(z.read(filename)) != digest:
                raise RuntimeError("ZIP member SHA256 mismatch: " + filename)
        if json.loads(z.read("PACKAGE_MEMBER_MANIFEST.json")) != manifest:
            raise RuntimeError("Embedded manifest mismatch")
    outer = sha(out.read_bytes())
    sha_path = output_dir / (basename + ".zip.sha256")
    sha_path.write_text(outer + "  " + out.name + "\n", encoding="utf8")
    proof = {
        "schema": "DETECTION_CHAT28_ZIP_VERIFIED_V1",
        "zip_filename": out.name,
        "zip_size_bytes": out.stat().st_size,
        "zip_sha256": outer,
        "verified_members": len(members),
        "source_git_head": head,
        "source_git_branch": branch,
        "testzip": "PASS",
        "member_sha256": "PASS",
        "complete_historical_nested_zip_included": False,
    }
    (output_dir / "ZIP_VERIFICATION_RESULT.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(p.parse_args().output_dir), indent=2))
