"""Native two-rank class assembly and identity binding, CPU-tested candidate.

THIS MODULE IS NOT AN EXECUTABLE RUNNER. It cannot authorize training, download
datasets, choose checkpoint bytes, or run trainer.train(). The real candidate
worker must first perform both-rank full R0/private-source-file verification
and exact constructor consensus; physical two-T4 CUDA/NCCL acceptance is a
separate future gate. Frozen C1 original source is never modified.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from .ddp_g4_recovery import validate_identity
from .ddp_native_binding import CLASS, SCHEMA as BINDING_SCHEMA, SOURCE_MOD
from .ddp_native_state import SCHEMA as STATE_SCHEMA
from .ddp_verified_admission import RECIPE, sha_object, audit_existing_stage


class DDPNativeBootstrapError(RuntimeError):
    pass


def need(ok: bool, error: str) -> None:
    if not ok:
        raise DDPNativeBootstrapError("DDP_B16_BOOTSTRAP_" + error)


def require_native_binding_identity(binding: dict) -> None:
    need(type(binding) is dict and binding.get("schema") == BINDING_SCHEMA and
         binding.get("status") == "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY",
         "CANDIDATE_SCHEMA")
    need(binding.get("gpu_execution_allowed") is False and
         binding.get("official_training_authorized") is False and
         binding.get("b_test_access") == "NONE", "NONEXECUTABLE_POLICY")
    identity = binding.get("identity")
    validate_identity(identity)
    arm = identity["arm"]
    need(arm in CLASS and arm == binding.get("arm") and
         binding.get("trainer_class") == CLASS[arm] and
         binding.get("trainer_module") == SOURCE_MOD[arm],
         "TRAINER_ARM_OR_MODULE")
    need(identity["ddp_recipe_sha256"] == sha_object(RECIPE),
         "RECIPE_SHA")
    body = {k: v for k, v in binding.items()
            if k not in ("schema", "binding_sha256", "status",
                         "gpu_execution_allowed", "official_training_authorized",
                         "b_test_access")}
    need(binding.get("binding_sha256") == sha_object(body), "BINDING_SHA")
    opts = binding.get("overrides")
    need(type(opts) is dict and
         opts.get("device") == "0,1" and
         opts.get("batch") == 16 and
         opts.get("epochs") == 100 and
         opts.get("resume") is False and
         opts.get("amp") is True and
         opts.get("workers") == 0, "FIXED_OVERRIDES")
    ctor = binding.get("constructor")
    need(type(ctor) is dict and set(ctor) == {
        "train_assignment_manifest", "val_assignment_manifest",
        "train_visibility_evidence", "pretrained_archive", "pretrained_best",
    } and ctor["train_assignment_manifest"] == ctor["val_assignment_manifest"],
         "EXACT_CONSTRUCTOR_PARAMETERS")


def verify_two_t4_runtime(cuda: Any) -> None:
    """Physical topology must be checked before importing native Ultralytics."""
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1",
         "PHYSICAL_CUDA_MAPPING")
    try:
        world, rank, local_rank = [int(os.environ[key])
                                   for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK")]
    except (KeyError, ValueError) as exc:
        raise DDPNativeBootstrapError("DDP_B16_BOOTSTRAP_TORCHRUN_ENV") from exc
    need((world, rank, local_rank) in ((2, 0, 0), (2, 1, 1)),
         "TORCHRUN_RANK_MAPPING")
    need(cuda.is_available() and cuda.device_count() == 2,
         "TWO_VISIBLE_CUDA_DEVICES_REQUIRED")
    names = [str(cuda.get_device_name(i)) for i in (0, 1)]
    need(all("T4" in name.upper() for name in names),
         "PHYSICAL_GPU_MUST_BE_T4")
    return names


def assemble_amp_ready_native_class(native: type, expected_name: str) -> type:
    """Preserve exact native class name for frozen arm and DDP guard checks."""
    need(isinstance(native, type) and native.__name__ == expected_name and
         expected_name in CLASS.values(), "NATIVE_TRAINER_CLASS")
    class BeforeNativeAMP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._r7c_amp_gate_ready = True
            super().__init__(*args, **kwargs)
    result = type(expected_name, (BeforeNativeAMP, native), {
        "__module__": native.__module__,
    })
    need(result.__name__ == expected_name and
         result.mro()[1] is BeforeNativeAMP and result.mro()[2] is native,
         "AMP_CLASS_MRO")
    return result


def reauthenticate_private_stage_for_import(binding: dict) -> dict:
    """Rebuild expected source hashes from C1 immediately before native import.

    The earlier two-rank admission record alone is not sufficient: a staged
    file or receipt could change after the first review. This runs the
    independent pinned-C1 source derivation and rehashes all staged files.
    A fresh verification is mandatory; no fallback to a prior receipt.
    """
    require_native_binding_identity(binding)
    need("ultralytics" not in sys.modules,
         "ULTRALYTICS_ALREADY_IMPORTED")
    for key in ("checkout_root", "source_root", "modules_root"):
        need(type(binding.get(key)) is str and
             Path(binding[key]).is_absolute() and
             ".." not in Path(binding[key]).parts,
             "PRIVATE_SOURCE_PATH_" + key)
    source = Path(binding["source_root"]).resolve(strict=True)
    modules = Path(binding["modules_root"]).resolve(strict=True)
    checkout = Path(binding["checkout_root"]).resolve(strict=True)
    need(source.is_dir() and modules.is_dir() and checkout.is_dir() and
         not source.is_symlink() and not modules.is_symlink() and
         not checkout.is_symlink(), "PRIVATE_SOURCE_MISSING")
    need(source.name == "source" and modules.name == "modules" and
         source.parent == modules.parent and
         not source.parent.is_relative_to(checkout),
         "PRIVATE_STAGE_LAYOUT")
    stage = audit_existing_stage(source.parent, checkout)
    need(stage["source_manifest_sha256"] ==
         binding["identity"]["stage_source_manifest_sha256"] and
         stage["stage_receipt_sha256"] ==
         binding["stage_receipt_sha256"],
         "STAGED_BYTES_CHANGED_AFTER_REVIEW")
    return {
        "status": "C1_DERIVED_SOURCE_REVERIFIED_BEFORE_NATIVE_IMPORT",
        "stage_receipt_sha256": stage["stage_receipt_sha256"],
        "source_manifest_sha256": stage["source_manifest_sha256"],
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
    }


def load_pinned_native_class(binding: dict) -> type:
    """Import only newly reverified staged source, never pip Ultralytics."""
    reauthenticate_private_stage_for_import(binding)
    source = Path(binding["source_root"]).resolve(strict=True)
    modules = Path(binding["modules_root"]).resolve(strict=True)
    sys.path.insert(0, str(modules))
    sys.path.insert(0, str(source))
    os.chdir(source)  # approved relative model YAML is under this exact source
    pkg = importlib.import_module("ultralytics")
    package_file = Path(pkg.__file__).resolve(strict=True)
    need(package_file.is_relative_to(source), "PIP_ULTRALYTICS_FORBIDDEN")
    native_module = importlib.import_module(binding["trainer_module"])
    expected_file = source / (
        binding["trainer_module"].replace(".", "/") + ".py"
    )
    need(Path(native_module.__file__).resolve(strict=True) ==
         expected_file.resolve(strict=True), "TRAINER_MODULE_OUTSIDE_STAGE")
    native = getattr(native_module, binding["trainer_class"], None)
    need(isinstance(native, type) and
         native.__module__ == binding["trainer_module"],
         "TRAINER_CLASS_IDENTITY")
    return assemble_amp_ready_native_class(native, binding["trainer_class"])


def bind_candidate_fresh_state(trainer: Any, binding: dict,
                               checkpoint_root: str | Path) -> None:
    """Attach a verified fresh-candidate identity; NEVER grant run authority.

    This does not open the locked --execute entrypoints. It must only be
    invoked by a future worker after independent rank-consensus verification
    of exactly this binding and a fresh, private checkpoint-root directory.
    """
    require_native_binding_identity(binding)
    need(type(trainer).__name__ == binding["trainer_class"],
         "CONSTRUCTED_TRAINER_CLASS")
    root = Path(checkpoint_root)
    need(root.is_absolute() and ".." not in root.parts and
         root.is_dir() and not root.is_symlink() and
         not any(part.lower() in {"test", "b-test", "b_test"} for part in root.parts),
         "CHECKPOINT_ROOT")
    need(getattr(trainer, "world_size", None) == 2 and
         getattr(trainer, "ddp", None) is False,
         "NATIVE_TORCHRUN_CONTEXT")
    args = getattr(trainer, "args", None)
    need(args is not None and
         getattr(args, "batch", None) == 16 and
         getattr(args, "device", None) == "0,1" and
         getattr(args, "resume", None) is False,
         "TRAINER_OPTIONS_DRIFT")
    identity = dict(binding["identity"])
    trainer._ddp_b16_g4_identity = identity
    trainer._ddp_b16_g4_root = str(root.resolve(strict=True))
    trainer._ddp_b16_verified_worker_ready = True
    trainer._ddp_b16_g4_native_bound = True
    trainer._ddp_b16_native_state_request = {
        "schema": STATE_SCHEMA,
        "mode": "FRESH_FROM_EARLY",
        "identity": identity,
        "checkpoint_root": str(root.resolve(strict=True)),
        "cursor": None,
        "externally_pinned_manifest_sha256": None,
    }
    # Terminal processing remains a separate verified acceptance gate.
    trainer._ddp_b16_terminal_candidate_ready = False
