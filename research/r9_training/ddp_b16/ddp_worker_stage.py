"""Authenticated, private DDP-B16 source staging (CPU only, no training).

Verifies the existing C1 336-file science source, 7 integration modules,
original source manifest and historical candidate recipe before copying.
Applies only the reviewed 3-file native DDP adapter inside a NEW, disposable
stage. Emits an exact SHA256 inventory, explicitly marked NOT AUTHORIZED.

Never modifies frozen_source/ or the original C1 Git checkout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from research.r9_training.ddp_b16.native_source_adapter import (
    BASE_BLOBS, create_candidate_sources,
)

C1_COMMIT = "fa6a7675daae385a04c7b77e69a9de29f9e1b2f3"
SOURCE_FILES = 336
INTEGRATION_FILES = 7


class DDPStageError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _r9_authenticator(checkout: Path) -> Any:
    path = checkout / "research/r9_training/code/R9B_R3_NATIVE_WORKER_INTEGRATION.py"
    if not path.is_file() or path.is_symlink():
        raise DDPStageError("DDP_R9_AUTHENTICATOR_NOT_REGULAR")
    spec = importlib.util.spec_from_file_location("_ddp_b16_r9_authenticator", path)
    if spec is None or spec.loader is None:
        raise DDPStageError("DDP_R9_AUTHENTICATOR_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_original_source(checkout: str | Path) -> dict[str, Any]:
    """Use the frozen R9 file-byte verifier, not a new mutable manifest."""
    root = Path(checkout).resolve(strict=True)
    base = root / "research/r9_training"
    r9 = _r9_authenticator(root)
    output = r9.verify_science_source(
        base / "frozen_source",
        base / "integration_modules",
        base / "code/R9_CANDIDATE_MANIFEST_R1.json",
    )
    recipe = base / "R9_RELEASE_RECIPE_CANDIDATE_R1.json"
    if not recipe.is_file() or recipe.is_symlink():
        raise DDPStageError("DDP_C1_RECIPE_NOT_REGULAR")
    if file_sha256(recipe) != r9.SOURCE_RECIPE_SHA:
        raise DDPStageError("DDP_C1_RECIPE_SHA_DRIFT")
    if output != {"source_files_verified": SOURCE_FILES,
                   "integration_modules_verified": INTEGRATION_FILES}:
        raise DDPStageError("DDP_C1_SOURCE_COUNT_DRIFT")
    return output


def _copy_checked_tree(src: Path, dest: Path) -> None:
    for path in src.rglob("*"):
        if path.is_symlink():
            raise DDPStageError("DDP_STAGE_SOURCE_SYMLINK_FORBIDDEN")
    shutil.copytree(
        src, dest, symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def create_private_stage(checkout: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create a NEW verified stage; return not-authorized receipt.

    Caller is responsible for choosing the intended private /kaggle/working
    location; this CPU function also supports disposable CI test directories.
    Any failed newly-created stage is removed, never the original C1 source.
    """
    root = Path(checkout).resolve(strict=True)
    dest = Path(destination)
    if not dest.is_absolute() or ".." in dest.parts or dest.exists() or dest.is_symlink():
        raise DDPStageError("DDP_STAGE_REQUIRES_NEW_ABSOLUTE_ROOT")
    if not dest.parent.is_dir() or dest.parent.is_symlink():
        raise DDPStageError("DDP_STAGE_PARENT_MUST_EXIST")
    resolved_dest = dest.resolve(strict=False)
    if resolved_dest == root or resolved_dest.is_relative_to(root):
        raise DDPStageError("DDP_STAGE_REFUSE_WRITE_INSIDE_CHECKOUT")
    verify_original_source(root)
    base = root / "research/r9_training"
    dest.mkdir(mode=0o700)
    try:
        source = dest / "source"
        modules = dest / "modules"
        _copy_checked_tree(base / "frozen_source", source)
        _copy_checked_tree(base / "integration_modules", modules)
        verify_original_source(root)  # reject source drift between initial checks and staging
        # Verify staged ORIGINAL tree by the same independently frozen manifest.
        r9 = _r9_authenticator(root)
        r9.verify_science_source(
            source, modules, base / "code/R9_CANDIDATE_MANIFEST_R1.json",
        )
        originals = {
            rel: (source / rel).read_bytes() for rel in BASE_BLOBS
        }
        modified = create_candidate_sources(originals)
        for relative, payload in modified.items():
            target = source / relative
            if not target.is_file() or target.is_symlink():
                raise DDPStageError("DDP_STAGE_PATCH_TARGET_NOT_REGULAR")
            target.write_bytes(payload)
            if target.read_bytes() != payload:
                raise DDPStageError("DDP_STAGE_PATCH_WRITE_DRIFT")
        orig_manifest = json.loads(
            (base / "code/R9_CANDIDATE_MANIFEST_R1.json").read_text("utf-8")
        )
        expected = dict(orig_manifest["candidate_source_sha256"])
        expected.update({p: file_sha256(source / p) for p in modified})
        actual = {
            p.relative_to(source).as_posix(): file_sha256(p)
            for p in source.rglob("*") if p.is_file()
        }
        if len(actual) != SOURCE_FILES or actual != expected:
            raise DDPStageError("DDP_STAGE_PATCHED_SOURCE_INVENTORY_DRIFT")
        module_actual = {
            p.relative_to(modules).as_posix(): file_sha256(p)
            for p in modules.rglob("*") if p.is_file()
        }
        if (len(module_actual) != INTEGRATION_FILES or
                module_actual != orig_manifest["unchanged_integration_modules_sha256"]):
            raise DDPStageError("DDP_STAGE_MODULE_DRIFT")
        receipt = {
            "schema": "VGRA_DDP_B16_PRIVATE_SOURCE_STAGE_CANDIDATE_V1",
            "status": "EXACT_SOURCE_STAGED_CPU_ONLY_NOT_GPU_AUTHORITY",
            "base_c1_commit": C1_COMMIT,
            "science_files_verified": SOURCE_FILES,
            "integration_modules_verified": INTEGRATION_FILES,
            "ddp_adapted_files": {p: file_sha256(source / p) for p in sorted(modified)},
            "stage_source_sha256": expected,
            "stage_modules_sha256": module_actual,
            "official_training_authorized": False,
            "gpu_execution_allowed": False,
            "gpu_used": False,
            "b_test_access": "NONE",
        }
        (dest / "STAGE_RECEIPT.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", "utf-8"
        )
        return receipt
    except BaseException:
        shutil.rmtree(dest)
        raise
