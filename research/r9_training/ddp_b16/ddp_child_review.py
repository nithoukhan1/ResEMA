"""Explicit two-rank torchrun CPU admission review — NO training path.

Example (only with verified existing Kaggle staged inputs):
torchrun --standalone --nproc_per_node=2 -m \\
  research.r9_training.ddp_b16.ddp_child_review --plan /kaggle/input/.../plan.json --review-only

Every rank independently verifies pinned R0 assets, C1 source, and the staged
private DDP source. Only AFTER local verification does it initialize a Gloo
process group and compare identities. A local error is transmitted to both.
No CUDA, model, optimizer, AMP, resume, checkpoint save or B-TEST access.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

from .ddp_worker import _unique_json, review_worker_inputs
from .ddp_verified_admission import verify_staged_admission
from .ddp_rank_admission import require_review_consensus
from .ddp_native_binding import make_native_binding
from .ddp_native_binding_consensus import require_native_binding_consensus


class DDPChildReviewError(RuntimeError):
    pass


def parse_rank_env() -> tuple[int, int]:
    try:
        world, rank, local = (
            int(os.environ.get(key, "")) for key in
            ("WORLD_SIZE", "RANK", "LOCAL_RANK")
        )
    except (TypeError, ValueError) as e:
        raise DDPChildReviewError("DDP_CHILD_EXPLICIT_TORCHRUN_REQUIRED") from e
    if (world, rank, local) not in ((2, 0, 0), (2, 1, 1)):
        raise DDPChildReviewError("DDP_CHILD_TWO_RANK_MAPPING_REQUIRED")
    return rank, world


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review-only", action="store_true")
    mode.add_argument("--native-binding-review", action="store_true")
    mode.add_argument("--native-import-probe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--project", default=None)
    args = parser.parse_args(argv)
    # Reject execution BEFORE opening a plan or importing torch.
    if args.execute:
        raise DDPChildReviewError("DDP_CHILD_EXECUTION_NOT_IMPLEMENTED_OR_AUTHORIZED")
    if (args.native_binding_review or args.native_import_probe) and not args.project:
        raise DDPChildReviewError("DDP_CHILD_PRIVATE_PROJECT_REQUIRED")
    if args.review_only and args.project is not None:
        raise DDPChildReviewError("DDP_CHILD_PROJECT_FOR_NATIVE_REVIEW_ONLY")
    rank, world = parse_rank_env()
    record = None
    binding = None
    local_error = None
    try:
        plan = _unique_json(__import__("pathlib").Path(args.plan))
        historical = review_worker_inputs(plan, stage_source=False)
        record = verify_staged_admission(plan, historical)
        if args.native_binding_review or args.native_import_probe:
            config = _unique_json(__import__("pathlib").Path(plan["r0_input_config"]))
            binding = make_native_binding(
                plan, record, config, project=args.project)
    except Exception as exc:
        local_error = type(exc).__name__ + ": " + str(exc)[:160]

    # Import/init torch only for the independent two-rank CPU/Gloo handshake.
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method="env://",
                            world_size=world, rank=rank,
                            timeout=timedelta(seconds=120))
    try:
        approved = require_review_consensus(record, local_error=local_error)
        if args.native_binding_review or args.native_import_probe:
            approved = require_native_binding_consensus(binding)
        if args.native_import_probe:
            # This probe checks physical topology and native CLASS IMPORT only.
            # It does not instantiate a trainer, deserialize weights or train.
            from .ddp_native_import_probe import collective_native_import_probe
            approved = collective_native_import_probe(binding)
        if rank == 0:
            label = ("DDP_B16_NATIVE_IMPORT_PROBE_ONLY=" if args.native_import_probe
                     else "DDP_B16_NATIVE_BINDING_REVIEW_ONLY="
                     if args.native_binding_review
                     else "DDP_B16_TWO_RANK_REVIEW_ONLY=")
            print(label + json.dumps(approved, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
