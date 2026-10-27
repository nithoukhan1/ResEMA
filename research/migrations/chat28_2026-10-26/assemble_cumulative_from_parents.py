"""Assemble VERIFIED cumulative Chat1–27 archives for Chat28 from real parents.

This is an OFFLINE packager. It cannot invent the original Chat1–26
historical archives and NEVER fetches or modifies dataset/checkpoints.
Uses the archived source's recorded SHA256 pins, verifies each nested ZIP's
CRC before nesting it, and writes separately verified FULL and PORTABLE ZIPs.

Workflow: download the previous full and portable parent ZIPs from your
Detection Project sources; download/extract the pinned Chat28 GitHub Action
artifact; run this script with the three actual ZIP paths.

Outputs are created only on the user's machine. Merely checking this
script into GitHub DOES NOT create the full cumulative archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

HISTORICAL_FULL_SHA256 = "942983c166536e78c02a4fe1c4e5d67b626a9809c78fddec3cb0c10c6ad1af3f"
HISTORICAL_PORTABLE_SHA256 = "ffebe21e221f4b548369ca9e5a90cdfc1e160f51b70bb22796d4bf5a8d9504ce"
NEW_RESEARCH_CODE_ZIP_SHA256 = "dc141f9ab37bc3221b5f8139a9b01e7edd7f3f51b484c75ff4839de7b410a238"
SOURCE_C1_SHA = "fa6a7675daae385a04c7b77e69a9de29f9e1b2f3"
DDP_CPU_TEST_SHA = "1b1696d3d77cf327452bcd40cb236c05c91098a2"
GITHUB_ARCHIVE_BUILD_SHA = "e3c2110c039b09aa813ad42fdf1deb8ea270a982"
DATE = "2026-10-27"
BASE = f"DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT27_FOR_CHAT28_{DATE}"

HANDOFF_SOURCE = "REPOSITORY_SNAPSHOT/research/migrations/chat28_2026-10-26/"
EXPORT_DOCUMENTS = {
    "01_MASTER_HANDOFF_CHAT1_TO_CHAT27.md": "MASTER_HANDOFF_CHAT1_TO_CHAT27_2026-10-27.md",
    "02_DDP_B16_IMPLEMENTATION_AND_EVIDENCE.md": "DDP_B16_IMPLEMENTATION_AND_EVIDENCE.md",
    "03_NEXT_STEPS_AND_GATES.md": "OVERALL_PROJECT_PROGRESS_TRACKER_2026-10-27.md",
    "04_SOURCE_INVENTORY_AND_LIMITATIONS.md": "SOURCE_INVENTORY_AND_LIMITATIONS.md",
    "CHAT28_FIRST_PROMPT_2026-10-26.txt": "CHAT28_FIRST_PROMPT_2026-10-27.txt",
    "CHAT26_CHECKPOINT_REGISTRY_SOURCE_2026-10-22.json": "CHECKPOINT_SHA256_REGISTRY_CHAT26_SOURCE.json",
    "HISTORICAL_MASTER_HANDOFF_CHAT1_TO_CHAT26_2026-10-22.md": "PREVIOUS_MASTER_HANDOFF_CHAT1_TO_CHAT26.md",
}


def sha256_file(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def require_archive(file: Path, expected: str, name: str) -> dict:
    if not file.is_file() or file.is_symlink() or file.suffix.lower() != ".zip":
        raise RuntimeError(f"{name}: missing/unsafe .zip source")
    found = sha256_file(file)
    if found != expected:
        raise RuntimeError(f"{name}: SHA256 mismatch: expected {expected}, found {found}")
    with zipfile.ZipFile(file) as arc:
        corrupt_member = arc.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"{name}: ZIP CRC FAILED at {corrupt_member}")
        names = arc.namelist()
        if not names or len(names) != len(set(names)):
            raise RuntimeError(f"{name}: empty/duplicate member names")
        if any(
            Path(n).is_absolute() or ".." in n.replace("\\", "/").split("/")
            for n in names
        ):
            raise RuntimeError(f"{name}: unsafe path")
    return {"filename": file.name, "bytes": file.stat().st_size, "sha256": found,
            "zip_crc": "PASS", "members": len(names)}


def addon_exports(file: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(file) as arc:
        manifest = json.loads(arc.read("PACKAGE_MEMBER_MANIFEST.json"))
        if (manifest.get("schema") != "DETECTION_CHAT28_GITHUB_SNAPSHOT_PACKAGE_V1" or
                manifest.get("git_head") != GITHUB_ARCHIVE_BUILD_SHA or
                manifest.get("frozen_science_release") != SOURCE_C1_SHA or
                manifest.get("upstream_historical_zip_included") is not False):
            raise RuntimeError("Chat28 GitHub archive has wrong provenance/parent claim")
        return {export: arc.read(HANDOFF_SOURCE + internal)
                for internal, export in EXPORT_DOCUMENTS.items()}


def _verify_written(path: Path, members: dict[str, dict]) -> dict:
    with zipfile.ZipFile(path) as z:
        error = z.testzip()
        if error is not None:
            raise RuntimeError("New ZIP failed CRC verification: " + error)
        if set(z.namelist()) != set(members) | {"PACKAGE_MEMBER_SHA256.json"}:
            raise RuntimeError("New ZIP membership mismatch")
        for name, record in members.items():
            h = hashlib.sha256()
            size = 0
            with z.open(name) as stream:
                for b in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(b)
                    size += len(b)
            if h.hexdigest() != record["sha256"] or size != record["bytes"]:
                raise RuntimeError("Nested source member failed SHA/size check: " + name)
        embedded = json.loads(z.read("PACKAGE_MEMBER_SHA256.json"))
        if embedded["members"] != members:
            raise RuntimeError("Package manifest mismatch")
    result = {"filename": path.name, "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "zip_crc": "PASS",
              "member_sha256": "PASS", "members_verified": len(members) + 1}
    path.with_suffix(path.suffix + ".sha256").write_text(
        result["sha256"] + "  " + path.name + "\n", "utf8")
    return result


def assemble(*, parent_full: Path, parent_portable: Path,
             addon: Path, output_dir: Path) -> dict:
    inputs = [parent_full.resolve(strict=True), parent_portable.resolve(strict=True),
              addon.resolve(strict=True)]
    if len(set(inputs)) != 3:
        raise RuntimeError("Each parent must be a DISTINCT source ZIP")
    expected = [HISTORICAL_FULL_SHA256, HISTORICAL_PORTABLE_SHA256,
                NEW_RESEARCH_CODE_ZIP_SHA256]
    if any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in expected):
        raise RuntimeError("Malformed source hash registry")

    attested = [require_archive(p, hash_, label) for p, hash_, label in zip(
        inputs, expected, ("Chat1-26 full parent", "Chat1-26 portable parent",
                           "Chat27 Github code add-on"))]
    docs = addon_exports(inputs[2])
    if not docs:
        raise RuntimeError("Missing authoritative handoff documents")

    output_dir = output_dir.resolve()
    if output_dir in inputs or any(p.parent == output_dir and p.name.startswith(BASE)
                                    for p in inputs):
        raise RuntimeError("Refuse to overwrite parent ZIPs")
    output_dir.mkdir(parents=True, exist_ok=True)
    required = sum(x["bytes"] for x in attested) * 2 + 64 * 1024 * 1024
    if shutil.disk_usage(output_dir).free < required:
        raise RuntimeError("Insufficient disk space for both verified cumulative archives")

    results = {}
    for variant, parent, parent_proof in (
        ("FINAL", inputs[0], attested[0]),
        ("PORTABLE", inputs[1], attested[1]),
    ):
        out = output_dir / (BASE + "_" + variant + ".zip")
        if out.exists():
            raise RuntimeError("Refuse to overwrite existing migration output: " + str(out))
        records = {
            f"historical_parent/{parent.name}":
                {"sha256": parent_proof["sha256"], "bytes": parent_proof["bytes"]},
            f"chat27_repo_and_handoff/{inputs[2].name}":
                {"sha256": attested[2]["sha256"], "bytes": attested[2]["bytes"]},
        }
        for name, data in docs.items():
            records["CURRENT_HANDOFF/" + name] = {
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        readme = (
            "DETECTION PROJECT — VERIFIED CUMULATIVE CHAT 1–27 FOR CHAT 28\n"
            f"Variant: {variant}; packaging date {DATE}\n"
            "Includes the EXACT old Chat1–26 predecessor ZIP plus the SHA-pinned "
            "Chat27 Github research/code ZIP. ZIPs are NESTED without changing bytes.\n"
            "Start with CURRENT_HANDOFF/MASTER_HANDOFF_CHAT1_TO_CHAT27_2026-10-27.md "
            "and CURRENT_HANDOFF/CHAT28_FIRST_PROMPT_2026-10-27.txt.\n"
            "Source code DDP CPU checkpoint: " + DDP_CPU_TEST_SHA + "\n"
            "Official MV00/MV01 completed: 0/100; B-TEST SEALED.\n"
            "Raw unprovided Kaggle checkpoints and patient X-ray files NOT present.\n"
            "See PACKAGE_MEMBER_SHA256.json for exact parent/addon member pins.\n"
        ).encode()
        records["00_READ_THIS_FIRST.txt"] = {
            "sha256": hashlib.sha256(readme).hexdigest(), "bytes": len(readme)}
        manifest = {
            "schema": "DETECTION_CHAT28_FULL_CHAIN_ASSEMBLY_V1",
            "variant": variant, "date": DATE,
            "source_c1_release": SOURCE_C1_SHA,
            "ddp_cpu_verified_sha": DDP_CPU_TEST_SHA,
            "chat27_github_package_git_head": GITHUB_ARCHIVE_BUILD_SHA,
            "original_chat1_to_chat26_parent_preserved_byte_for_byte": True,
            "original_unprovided_checkpoint_binaries_included": False,
            "official_epochs": {"MV00": 0, "MV01": 0},
            "b_test_access": "NONE",
            "members": records,
        }
        try:
            with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_STORED,
                                 allowZip64=True) as z:
                z.write(parent, arcname="historical_parent/" + parent.name)
                z.write(inputs[2], arcname="chat27_repo_and_handoff/" + inputs[2].name)
                for name, data in sorted(docs.items()):
                    z.writestr("CURRENT_HANDOFF/" + name, data)
                z.writestr("00_READ_THIS_FIRST.txt", readme)
                z.writestr("PACKAGE_MEMBER_SHA256.json",
                           json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
            results[variant] = _verify_written(out, records)
        except Exception:
            if out.exists():
                out.unlink()
            raise

    summary = {
        "schema": "DETECTION_CHAT28_FULL_AND_PORTABLE_VERIFIED_V1",
        "historical_full_parent": attested[0],
        "historical_portable_parent": attested[1],
        "verified_git_code_addon": attested[2],
        "outputs": results,
        "status": "FULL_AND_PORTABLE_ZIP_CRC_AND_MEMBER_SHA256_PASS",
    }
    (output_dir / "MIGRATION_VERIFICATION_REPORT.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", "utf8")
    (output_dir / "MIGRATION_SHA256_CHECKSUMS.txt").write_text(
        "".join(results[k]["sha256"] + "  " + results[k]["filename"] + "\n"
                for k in ("FINAL", "PORTABLE")), "utf8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-parent", type=Path, required=True,
                   help="Original Chat1–26 FINAL.zip downloaded from Project")
    p.add_argument("--portable-parent", type=Path, required=True,
                   help="Original Chat1–26 PORTABLE.zip downloaded from Project")
    p.add_argument("--addon", type=Path, required=True,
                   help="Chat28 Github Action zip inside extracted artifact")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(assemble(parent_full=args.full_parent,
                              parent_portable=args.portable_parent,
                              addon=args.addon, output_dir=args.output_dir),
                     indent=2))


if __name__ == "__main__":
    main()
