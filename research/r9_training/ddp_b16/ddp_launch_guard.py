"""Fail-closed native launch boundary for candidate DDP-B16 trainers.

The stock Ultralytics auto-DDP subprocess generator drops VGRA's required
non-overrides constructor inputs (assignment manifest, visibility evidence,
verified pretrained archive and checkpoint). Never use trainer.train() from
a non-torchrun parent for DDP-B16.

This standalone guard does not authorize training. A separate, independently
reviewed worker must authenticate *all* frozen inputs and distributed G4
recovery and explicitly arm the trainer before this guard can permit training.
"""
from __future__ import annotations

import os


class DDPB16LaunchError(RuntimeError):
    pass


def require_authenticated_ddp_child(trainer) -> None:
    args = getattr(trainer, "args", None)
    if args is None:
        raise DDPB16LaunchError("DDP_B16_ARGS_MISSING")
    if type(getattr(trainer, "world_size", None)) is not int or trainer.world_size != 2:
        raise DDPB16LaunchError("DDP_B16_WORLD_SIZE")
    if type(getattr(args, "batch", None)) is not int or args.batch != 16:
        raise DDPB16LaunchError("DDP_B16_GLOBAL_BATCH")
    if str(getattr(args, "device", None)) != "0,1":
        raise DDPB16LaunchError("DDP_B16_GPU_MAPPING")
    if getattr(trainer, "ddp", None) is not False:
        raise DDPB16LaunchError("DDP_B16_REFUSE_ULTRALYTICS_AUTO_LAUNCH")

    try:
        world = int(os.environ.get("WORLD_SIZE", ""))
        rank = int(os.environ.get("RANK", ""))
        local_rank = int(os.environ.get("LOCAL_RANK", ""))
    except (TypeError, ValueError) as exc:
        raise DDPB16LaunchError("DDP_B16_REQUIRES_EXPLICIT_TORCHRUN") from exc
    if (world, rank, local_rank) not in ((2, 0, 0), (2, 1, 1)):
        raise DDPB16LaunchError("DDP_B16_INVALID_TORCHRUN_RANK_MAPPING")

    # NEVER accept a bare environment flag or an approval phrase.
    # The future per-arm worker must bind a separately reviewed, verified
    # recovery/safety object before setting this instance-local sentinel.
    if getattr(trainer, "_ddp_b16_verified_worker_ready", None) is not True:
        raise DDPB16LaunchError("DDP_B16_VERIFIED_WORKER_NOT_IMPLEMENTED")

    if not getattr(trainer, "_d4e_recovery_configured", False):
        raise DDPB16LaunchError("DDP_B16_GOVERNED_RECOVERY_NOT_BOUND")
