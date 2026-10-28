"""Two-rank native class IMPORT-ONLY acceptance probe, never a training run.

Each process must first independently pass the full R0/staged-source review,
constructor binding and two-rank identity consensus. Only then does this
Gloo collective verify the actual 2xT4 CUDA topology, rehash 343 staged
files from immutable C1 and import the exact private native class.

No trainer is instantiated, no model is deserialized, no optimizer is created,
no epoch is entered, and no B-TEST path is opened. GPU acceptance of training,
NCCL, FP16, DDP loss, and G4 off-session replay remain separate work.
"""
from __future__ import annotations

from typing import Any

from .ddp_native_binding import CLASS, SOURCE_MOD
from .ddp_native_bootstrap_candidate import (
    require_native_binding_identity, verify_two_t4_runtime,
    load_pinned_native_class,
)


class DDPNativeImportProbeError(RuntimeError):
    pass


def collective_native_import_probe(binding: Any) -> dict:
    """Collectively report native import success/failure without running training."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized() and
            dist.get_world_size() == 2 and dist.get_rank() in (0, 1)):
        raise DDPNativeImportProbeError("DDP_NATIVE_IMPORT_TWO_RANK_GLOO_REQUIRED")
    if dist.get_backend() != "gloo":
        raise DDPNativeImportProbeError("DDP_NATIVE_IMPORT_PROBE_GLOO_ONLY")
    rank = dist.get_rank()
    error = None
    signature = None
    try:
        require_native_binding_identity(binding)
        verify_two_t4_runtime(torch.cuda)
        # The loader itself regenerates C1-derived source and checks all stage
        # SHA256 values AGAIN immediately before the Python module import.
        native_class = load_pinned_native_class(binding)
        if (native_class.__name__ != CLASS[binding["arm"]] or
                native_class.__module__ != SOURCE_MOD[binding["arm"]]):
            raise ValueError("NATIVE_CLASS_ORIGIN_DRIFT")
        signature = {
            "arm": binding["arm"],
            "binding_sha256": binding["binding_sha256"],
            "trainer_class": native_class.__name__,
            "trainer_module": native_class.__module__,
            "source_sha256": binding["identity"]["stage_source_manifest_sha256"],
            "stage_receipt_sha256": binding["stage_receipt_sha256"],
        }
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)[:180]
    wire = [None, None]
    dist.all_gather_object(wire, {
        "rank": rank, "error": error,
        "signature": signature if error is None else None,
    })
    if [entry["rank"] for entry in wire] != [0, 1]:
        raise DDPNativeImportProbeError("DDP_NATIVE_IMPORT_RANK_ORDER")
    if any(entry["error"] is not None for entry in wire):
        raise DDPNativeImportProbeError("DDP_NATIVE_IMPORT_PEER_REJECTED")
    if wire[0]["signature"] != wire[1]["signature"]:
        raise DDPNativeImportProbeError("DDP_NATIVE_IMPORT_RANK_DRIFT")
    return {
        "schema": "DDP_B16_IMPORT_ONLY_TWO_RANK_REVIEW_V1",
        "status": "BOTH_NATIVE_CLASSES_IMPORTED_NO_MODEL_CONSTRUCTION",
        "arm": binding["arm"],
        "world_size": 2,
        "binding_sha256": binding["binding_sha256"],
        "native_class": wire[0]["signature"]["trainer_class"],
        "source_manifest_sha256": wire[0]["signature"]["source_sha256"],
        "gpu_topology_only_checked": True,
        "native_train_invoked": False,
        "model_deserialization_performed": False,
        "official_training_authorized": False,
        "gpu_training_authorized": False,
        "b_test_access": "NONE",
    }
