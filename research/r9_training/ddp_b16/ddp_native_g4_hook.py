"""Native DDP-B16 G4 epoch callback candidate — NOT training authority.

Called on BOTH ranks after rank-0 save_model and on_fit_epoch_end, before
the original native stop broadcast. Captures a resumable 2-rank checkpoint
only for NONTERMINAL, fully validated epochs. The original C1 D4E/G4 hooks
must never be called from the DDP source candidate.

Native startup, CUDA AMP, independently pinned off-session transport and
terminal best-model evaluation are separate unfinished gates.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .ddp_g4_recovery import distributed_capture_and_publish, validate_identity


class DDPG4NativeHookError(RuntimeError):
    pass


def _need(value: bool, reason: str) -> None:
    if not value:
        raise DDPG4NativeHookError("DDP_B16_NATIVE_G4_" + reason)


def _sha(path: Any) -> str:
    file = Path(path)
    _need(file.is_file() and not file.is_symlink(), "SELECTED_CHECKPOINT_MISSING")
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ddp_b16_g4_epoch_end(
    trainer: Any, *, epoch: int, nb: int, last_opt_step: int,
    _cpu_test_augment_capture: Any = None,
) -> dict[str, Any]:
    """Synchronize checkpoint identity and capture both rank states.

    _cpu_test_augment_capture is for artificial CI-only tests. The future
    production worker must call this hook without supplying that override.
    """
    import torch.distributed as dist

    _need(dist.is_available() and dist.is_initialized(), "NO_DISTRIBUTED_GROUP")
    rank = dist.get_rank()
    _need(dist.get_world_size() == 2 and rank in (0, 1), "RANKS")
    _need(getattr(trainer, "world_size", None) == 2, "TRAINER_WORLD")
    _need(getattr(trainer, "_d4e_recovery_configured", False) is False,
          "OLD_BATCH8_G4_FORBIDDEN")
    _need(getattr(trainer, "_ddp_b16_g4_native_bound", False) is True,
          "AUTHENTICATED_BINDING_MISSING")
    args = getattr(trainer, "args", None)
    _need(args is not None and
          (getattr(args, "batch", None), getattr(args, "imgsz", None),
           getattr(args, "epochs", None), getattr(args, "workers", None),
           getattr(args, "seed", None), getattr(args, "nbs", None)) ==
          (16, 1024, 100, 0, 42, 64) and bool(getattr(args, "amp", False)),
          "RECIPE_DRIFT")
    _need(type(epoch) is int and 0 <= epoch < 100 and
          epoch == getattr(trainer, "epoch", None), "EPOCH")
    _need(nb == 890 and type(last_opt_step) is int, "BATCH_OR_STEP_COUNT")
    identity = getattr(trainer, "_ddp_b16_g4_identity", None)
    validate_identity(identity)
    root = getattr(trainer, "_ddp_b16_g4_root", None)
    _need(isinstance(root, (str, Path)), "CAPSULE_ROOT_MISSING")
    loader = getattr(trainer, "train_loader", None)
    sampler = getattr(loader, "vgra_pair_batch_sampler", None)
    _need(sampler is not None and getattr(sampler, "rank", None) == rank and
          getattr(sampler, "epoch", None) == epoch + 1 and len(loader) == nb,
          "SAMPLER_CURSOR")
    _need(getattr(trainer, "metrics", None) and
          math.isfinite(float(getattr(trainer, "fitness", float("nan")))),
          "PRE_SAVE_VALIDATION_REQUIRED")
    opt_steps = getattr(trainer, "_d4_optimizer_steps_observed", None)
    _need(type(opt_steps) is int and opt_steps >= 0, "OPTIMIZER_STEP_COUNTER")
    _need(type(getattr(trainer, "accumulate", None)) is int and
          trainer.accumulate >= 1, "ACCUMULATION")

    # Authenticate the rank-0 selected-best checkpoint. Both ranks must
    # follow identical stop/terminal decisions; rank 1 does not read best.pt.
    shared = [None]
    if rank == 0:
        try:
            if not Path(trainer.last).is_file():
                raise DDPG4NativeHookError("DDP_B16_NATIVE_G4_NATIVE_LAST_MISSING")
            meta = {
                "ok": True,
                "best_sha": _sha(trainer.best),
                "last_sha": _sha(trainer.last),
                "stop": bool(trainer.stop) or epoch + 1 >= 100,
                "best_fitness": float(trainer.best_fitness),
                "stopper_best_epoch": int(trainer.stopper.best_epoch),
                "stopper_possible_stop": bool(trainer.stopper.possible_stop),
                "optimizer_updates": opt_steps,
                "last_opt_step": last_opt_step,
                "accumulate": trainer.accumulate,
            }
            _need(math.isfinite(meta["best_fitness"]), "BEST_FITNESS_NONFINITE")
            shared[0] = meta
        except Exception as exc:
            shared[0] = {"ok": False, "reason": str(exc)[:240]}
    dist.broadcast_object_list(shared, src=0)
    meta = shared[0]
    _need(type(meta) is dict and meta.get("ok") is True,
          "RANK0_CHECKPOINT_" + str(meta.get("reason", "invalid")))
    local_matches = (meta["optimizer_updates"] == opt_steps and
                     meta["last_opt_step"] == last_opt_step and
                     meta["accumulate"] == trainer.accumulate)
    # Do not let only one rank enter the G4 checkpoint collective after a
    # local mismatch. Make both ranks fail at the same collective boundary.
    agreements = [None, None]
    dist.all_gather_object(agreements, bool(local_matches))
    _need(agreements == [True, True], "RANK_STEP_DIVERGENCE")

    # A terminal epoch has no next-epoch resume capsule. Terminal best
    # evaluation/export is another, still-locked contract.
    if meta["stop"]:
        dist.barrier()
        return {"rank": rank, "epoch": epoch,
                "status": "TERMINAL_EPOCH_G4_CAPSULE_NOT_WRITTEN",
                "terminal_export_authorized": False}

    cursor = {
        "completed_epoch": epoch, "next_epoch": epoch + 1,
        "steps_per_rank": nb, "global_batch": 16,
        "optimizer_updates": opt_steps, "last_opt_step": last_opt_step,
        "accumulate": trainer.accumulate, "sampler_next_epoch": epoch + 1,
    }
    stopper = {
        "best_fitness": meta["best_fitness"],
        "best_epoch": meta["stopper_best_epoch"],
        "possible_stop": meta["stopper_possible_stop"],
        "selected_best_sha256": meta["best_sha"],
    }
    ema = getattr(trainer, "ema", None)
    _need(ema is not None and getattr(ema, "ema", None) is not None,
          "EMA_UNAVAILABLE")
    if _cpu_test_augment_capture is None:
        from ultralytics.models.yolo.detect.d4e_transform_rng_state import capture_augmentation_rng
        augmentation = capture_augmentation_rng(loader.dataset.transforms)
    else:
        augmentation = _cpu_test_augment_capture(loader.dataset.transforms)
    result = distributed_capture_and_publish(root, rank=rank, capture_kwargs={
        "identity": identity, "cursor": cursor,
        "model": trainer.model, "optimizer": trainer.optimizer,
        "ema_model": ema.ema, "ema_updates": ema.updates,
        "scheduler": trainer.scheduler, "scaler": trainer.scaler,
        "loader_generator": loader.generator,
        "augmentation_rng": augmentation, "sampler": sampler,
        "stopper": stopper,
    })
    result.update({"epoch": epoch, "selected_best_sha256": meta["best_sha"],
                   "native_last_sha256": meta["last_sha"],
                   "offsession_persistence_verified": False})
    return result
