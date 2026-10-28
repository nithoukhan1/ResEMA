"""Two-rank review-only DDP admission consensus. No training authority.

The calling torchrun child must independently verify local R0 assets and all
stage files before this collective. Comparing two caller-made objects alone is
NOT proof of their provenance.
"""
from __future__ import annotations

import re
from typing import Any

from .ddp_verified_admission import (
    ADMISSION_SCHEMA, ASSIGNMENT, PRETRAINED, RECIPE, sha_object,
)
from .ddp_g4_recovery import validate_identity


class DDPRankAdmissionError(RuntimeError):
    pass


def require_review_consensus(admission: Any, *, local_error: str | None = None) -> dict:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized() and
            dist.get_world_size() == 2 and dist.get_rank() in (0, 1)):
        raise DDPRankAdmissionError("DDP_RANK_REVIEW_EXPECT_TWO_PROCESS_GROUP")
    rank = dist.get_rank()
    rejection = local_error
    if rejection is None:
        try:
            if not isinstance(admission, dict) or admission.get("schema") != ADMISSION_SCHEMA:
                raise ValueError("ADMISSION_SCHEMA")
            if (admission.get("status") != "R0_AND_PRIVATE_STAGE_BYTES_VERIFIED_REVIEW_ONLY"
                    or admission.get("gpu_execution_allowed") is not False
                    or admission.get("official_training_authorized") is not False
                    or admission.get("b_test_access") != "NONE"):
                raise ValueError("REVIEW_ONLY_POLICY")
            identity = admission.get("identity")
            validate_identity(identity)
            if (identity["assignment_sha256"] != ASSIGNMENT or
                    identity["pretrained_sha256"] != PRETRAINED or
                    identity["ddp_recipe_sha256"] != sha_object(RECIPE)):
                raise ValueError("FIXED_SCIENTIFIC_PINS")
            for k in ("r0_record_sha256", "stage_receipt_sha256"):
                if not isinstance(admission.get(k), str) or not re.fullmatch(
                        r"[0-9a-f]{64}", admission[k]):
                    raise ValueError("DIGEST_" + k)
        except Exception as e:
            rejection = type(e).__name__ + ":" + str(e)[:120]
    submission = {
        "rank": rank, "error": rejection,
        "record": ({
            "identity": admission["identity"],
            "r0_record_sha256": admission["r0_record_sha256"],
            "stage_receipt_sha256": admission["stage_receipt_sha256"],
        } if rejection is None else None),
    }
    members = [None, None]
    dist.all_gather_object(members, submission)
    if [x["rank"] for x in members] != [0, 1]:
        raise DDPRankAdmissionError("DDP_RANK_REVIEW_BAD_RANK_ORDER")
    if any(x["error"] is not None for x in members):
        raise DDPRankAdmissionError("DDP_RANK_REVIEW_LOCAL_ERROR")
    if members[0]["record"] != members[1]["record"]:
        raise DDPRankAdmissionError("DDP_RANK_REVIEW_IDENTITY_DISAGREEMENT")
    return {
        "status": "TWO_RANK_R0_STAGE_CONSENSUS_REVIEW_ONLY",
        "rank": rank,
        "world_size": 2,
        "identity": dict(members[0]["record"]["identity"]),
        "gpu_execution_allowed": False,
        "official_training_authorized": False,
        "b_test_access": "NONE",
    }
