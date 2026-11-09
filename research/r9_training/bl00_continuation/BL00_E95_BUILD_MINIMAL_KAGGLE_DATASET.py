#!/usr/bin/env python3
r"""Create a SHA-authenticated minimal BL00 Epoch95 Kaggle Dataset from extracted failed output.

Example:
python BL00_E95_BUILD_MINIMAL_KAGGLE_DATASET.py --source "C:\Users\ik_pa\Downloads\results (15)bl00\BL00_CONTINUOUS_B16_OUTPUT\g4"

No dependencies beyond Python stdlib. The original failed output is never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

PIN = "05a87470d32ec7506f9d4212ce7ba5881ec703299883c2c445505bf2bc6e0c5d"
COMMIT = "f440d329d4544ec08ad63c1972bcf9a8b8a7eed8"
NAME = "BL00_EPOCH095_two_rank_manifest.json"
ROOT = "BL00_CONTINUOUS_B16_OUTPUT"

def require(ok, message):
    if not ok:
        raise RuntimeError("BL00_E95_DATASET_" + message)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def checkfile(p):
    require(p.is_file() and not p.is_symlink(), "FILE_MISSING_OR_SYMLINK_" + str(p))

def readjson(p):
    checkfile(p)
    return json.loads(p.read_text(encoding="utf-8"))

def relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "UNSAFE_MEMBER_" + repr(name))
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "." not in p.parts,
            "UNSAFE_MEMBER_" + name)
    return Path(*p.parts)

def verified_copy(source, target, expected):
    checkfile(source)
    require(sha(source) == expected, "SOURCE_SHA_MISMATCH_" + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "TARGET_ALREADY_EXISTS_" + str(target))
    shutil.copyfile(source, target)
    require(sha(target) == expected, "COPY_SHA_MISMATCH_" + str(target))
    print("SHA256 PASS", target.name, format(target.stat().st_size, ",") + " bytes", flush=True)
    return (target, expected)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Extracted BL00 output root OR g4 subfolder")
    p.add_argument("--out", help="Destination (must not exist)")
    a = p.parse_args()
    original = Path(a.source).expanduser().resolve()
    if original.name.lower() == "g4":
        original = original.parent
    require(original.is_dir() and original.name == ROOT, "WRONG_OUTPUT_ROOT_" + str(original))
    g4, boot = original / "g4", original / "bootstrap"
    manifest_path = g4 / NAME
    pointer_path = g4 / "LATEST_MANIFEST.json"
    registry_path = boot / "SOURCE_SHA256.json"
    for f in (manifest_path, pointer_path, registry_path):
        checkfile(f)
    require(sha(manifest_path) == PIN, "EPOCH95_MANIFEST_PIN_DRIFT")
    manifest, pointer, registry = map(readjson, (manifest_path, pointer_path, registry_path))
    expected_meta = {"arm":"BL00","complete":True,"completed_epochs":95,
                     "next_epoch_index":95,"optimizer_updates":21767,
                     "world_size":2,"b_test_access":"NONE","frozen_commit":COMMIT}
    for key, expected in expected_meta.items():
        require(manifest.get(key) == expected, "MANIFEST_METADATA_" + key)
    members = manifest.get("files", {})
    require(isinstance(members, dict) and set(members) ==
            {"BL00_EPOCH095_rank0.pt","BL00_EPOCH095_rank1.pt","BL00_EPOCH095_best.pt"},
            "G4_MEMBER_SET")
    require(manifest.get("best_filename") == "BL00_EPOCH095_best.pt" and
            manifest.get("selected_best_sha256") == members["BL00_EPOCH095_best.pt"],
            "BEST_SHA_IDENTITY")
    for rank in (0,1):
        name = "BL00_EPOCH095_rank" + str(rank) + ".pt"
        row = manifest.get("ranks", {}).get(str(rank), {})
        require(row.get("filename") == name and row.get("sha256") == members[name],
                "RANK_SHA_IDENTITY_" + str(rank))
    require(pointer.get("manifest") == NAME and
            pointer.get("manifest_sha256") == PIN and
            pointer.get("completed_epochs") == 95, "LATEST_POINTER_IDENTITY")
    require(isinstance(registry, dict) and len(registry) == 13, "EXPECTED_13_BOOTSTRAP_MEMBERS")
    require("bl00_epoch1_runner.py" in registry and
            "bl00_epoch1_validator.py" in registry, "ORIGINAL_BOOTSTRAP_MISSING_RUNNER")
    output = (Path(a.out).expanduser().resolve() if a.out else
              original.parent / "BL00_E95_RECOVERY_MINIMAL_KAGGLE_DATASET")
    require(not output.exists(), "DESTINATION_EXISTS_NO_OVERWRITE_" + str(output))
    require(original not in output.parents, "DESTINATION_INSIDE_SOURCE")
    base = output / ROOT
    copied = []
    print("SOURCE:", original, "\nOUTPUT:", output, "\nE95 SHA256:", PIN, "\nB-TEST: SEALED")
    try:
        for name, digest in sorted(members.items()):
            rel = relative(name)
            require(len(rel.parts) == 1, "G4_MEMBER_NOT_BASENAME")
            copied.append(verified_copy(g4 / rel, base / "g4" / rel, digest))
        copied.append(verified_copy(manifest_path, base / "g4" / NAME, PIN))
        copied.append(verified_copy(pointer_path, base / "g4" / "LATEST_MANIFEST.json",
                                    sha(pointer_path)))
        for name, digest in sorted(registry.items()):
            rel = relative(name)
            copied.append(verified_copy(boot / rel, base / "bootstrap" / rel, digest))
        copied.append(verified_copy(registry_path, base / "bootstrap" / "SOURCE_SHA256.json",
                                    sha(registry_path)))
        require(len(copied) == 19, "INCOMPLETE_19_FILE_DATASET")
        for f, digest in copied:
            require(sha(f) == digest, "FINAL_DATASET_AUDIT_" + f.name)
        receipt = {"schema":"BL00_E95_MANUAL_TRANSFER_DATASET_V1",
                   "arm":"BL00","completed_epochs":95,"optimizer_updates":21767,
                   "g4_manifest_sha256":PIN,"b_test_access":"NONE",
                   "file_count":len(copied),
                   "total_bytes":sum(f.stat().st_size for f,_ in copied),
                   "files":{f.relative_to(output).as_posix(): d for f,d in copied}}
        (output / "BL00_E95_DATASET_TRANSFER_RECEIPT.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print("\nBL00 E95 MINIMAL DATASET: PASS")
        print("INCLUDED FILES:", len(copied), "(plus receipt)")
        print("SIZE GiB:", round(receipt["total_bytes"] / (1024**3),3))
        print("UPLOAD THIS FOLDER:", output)
        print("B-TEST: SEALED")
    except Exception:
        print("BUILD FAILED. SOURCE UNMODIFIED. Partial output may exist:", output, file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("BL00_E95_DATASET_ERROR:", str(e), file=sys.stderr)
        sys.exit(1)
