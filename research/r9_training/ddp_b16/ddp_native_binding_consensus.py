"""Two-rank consensus over exact native constructor bindings (review-only).

Each rank MUST independently run full frozen R0 + stage provenance and construct
this binding from the actual pinned asset files. Matching declared objects
alone does NOT authenticate a run, and cannot authorize a GPU experiment.
"""
from __future__ import annotations

import re
from typing import Any

from .ddp_native_binding import CLASS, SCHEMA, safe_binding_summary
from .ddp_verified_admission import sha_object
from .ddp_g4_recovery import validate_identity


class DDPNativeBindingConsensusError(RuntimeError):
    pass


def require_native_binding_consensus(binding: Any) -> dict:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized() and
            dist.get_world_size() == 2 and dist.get_rank() in (0, 1)):
        raise DDPNativeBindingConsensusError("DDP_BIND_CONSENSUS_TWO_RANK_GROUP")
    rank = dist.get_rank()
    error = None
    record = None
    try:
        if type(binding) is not dict or binding.get("schema") != SCHEMA:
            raise ValueError("SCHEMA")
        if (binding.get("status") != "EXACT_NATIVE_CONSTRUCTOR_BOUND_REVIEW_ONLY"
                or binding.get("gpu_execution_allowed") is not False
                or binding.get("official_training_authorized") is not False
                or binding.get("b_test_access") != "NONE"):
            raise ValueError("NONEXECUTABLE_POLICY")
        identity = binding.get("identity")
        validate_identity(identity)
        if identity["arm"] not in CLASS or binding.get("arm") != identity["arm"]:
            raise ValueError("ARM")
        if binding.get("trainer_class") != CLASS[identity["arm"]]:
            raise ValueError("TRAINER_CLASS")
        expected = {
            key: value for key, value in binding.items()
            if key not in {"schema", "binding_sha256",
                           "gpu_execution_allowed", "official_training_authorized",
                           "b_test_access", "status"}
        }
        digest = sha_object(expected)
        if binding.get("binding_sha256") != digest:
            raise ValueError("FULL_CONSTRUCTOR_DIGEST")
        for key in ("r0_record_sha256", "stage_receipt_sha256"):
            if (type(binding.get(key)) is not str or
                    re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None):
                raise ValueError("INPUT_RECEIPT_" + key)
        options = binding.get("overrides")
        if (type(options) is not dict or options.get("batch") != 16 or
                options.get("device") != "0,1" or options.get("epochs") != 100 or
                options.get("resume") is not False or options.get("amp") is not True):
            raise ValueError("NATIVE_RECIPE")
        record = {
            "binding_sha256": digest,
            "arm": identity["arm"],
            "release_sha": identity["ddp_release_sha"],
            "stage_sha": identity["stage_source_manifest_sha256"],
            "r0_record_sha256": binding["r0_record_sha256"],
            "stage_receipt_sha256": binding["stage_receipt_sha256"],
        }
    except Exception as exc:
        error = type(exc).__name__ + ":" + str(exc)[:140]
    peers = [None, None]
    dist.all_gather_object(peers, {
        "rank": rank, "error": error, "record": record,
    })
    if [p["rank"] for p in peers] != [0, 1]:
        raise DDPNativeBindingConsensusError("DDP_BIND_CONSENSUS_RANK_ORDER")
    if any(p["error"] is not None for p in peers):
        raise DDPNativeBindingConsensusError("DDP_BIND_CONSENSUS_RANK_REJECTION")
    if peers[0]["record"] != peers[1]["record"]:
        raise DDPNativeBindingConsensusError("DDP_BIND_CONSENSUS_CONSTRUCTOR_DRIFT")
    return {
        "status": "TWO_RANK_NATIVE_CONSTRUCTOR_REVIEW_ONLY",
        "binding_sha256": peers[0]["record"]["binding_sha256"],
        "arm": peers[0]["record"]["arm"],
        "world_size": 2,
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }
