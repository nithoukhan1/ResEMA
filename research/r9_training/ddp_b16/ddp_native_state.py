"""Native DDP-B16 FRESH/G4 RESUME lifecycle candidate (NOT executable authority).

Both ranks call this after native _setup_train(). A future independently
authenticated torchrun worker must bind the request; neither this module nor
a boolean readiness marker authorizes executing a model. This code is CPU
candidate engineering; real NCCL/AMP/off-session replay remains unproven.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from .ddp_batch_loader import begin_distributed_pair_epoch, verify_ddp_training_loader
from .ddp_g4_recovery import read_pinned_rank, restore_rank, validate_cursor, validate_identity

SCHEMA = "DDP_B16_NATIVE_STATE_REQUEST_CANDIDATE_V1"
KEYS = {"schema", "mode", "identity", "checkpoint_root", "cursor", "externally_pinned_manifest_sha256"}
MODES = {"FRESH_FROM_EARLY", "RESUME_FROM_DDP_B16"}


class DDPNativeStateError(RuntimeError):
    pass


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise DDPNativeStateError("DDP_B16_NATIVE_" + code)


def _file_sha(path: Any) -> str:
    p = Path(path)
    _need(p.is_file() and not p.is_symlink(), "SELECTED_BEST_MISSING")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_request(trainer: Any, rank: int) -> dict:
    import torch

    plan = getattr(trainer, "_ddp_b16_native_state_request", None)
    _need(type(plan) is dict and set(plan) == KEYS and
          plan.get("schema") == SCHEMA and plan.get("mode") in MODES,
          "REQUEST_SCHEMA")
    validate_identity(plan["identity"])
    _need(plan["identity"] == getattr(trainer, "_ddp_b16_g4_identity", None),
          "RUNTIME_IDENTITY")
    arm = ("MV00" if type(trainer).__name__ == "EarlyPairedTrainer" else
           "MV01" if type(trainer).__name__ == "VGRADetectionTrainer" else None)
    _need(arm is not None and arm == plan["identity"]["arm"], "ARM")
    _need(getattr(trainer, "world_size", None) == 2 and
          getattr(trainer, "_ddp_b16_verified_worker_ready", False) is True and
          getattr(trainer, "_ddp_b16_g4_native_bound", False) is True,
          "UNVERIFIED_WORKER_OR_G4")
    _need(not getattr(trainer, "_d4e_recovery_configured", False) and
          not getattr(trainer, "_g3b4_ready", False), "LEGACY_BATCH8_G4")
    _need(not getattr(trainer, "_ddp_b16_native_state_ready", False),
          "ALREADY_INITIALIZED")
    root = plan["checkpoint_root"]
    _need(type(root) is str and Path(root).is_absolute() and
          ".." not in Path(root).parts and
          not any(p.is_symlink() for p in (Path(root), *Path(root).parents)),
          "G4_ROOT_PATH")
    target = Path(root).resolve(strict=True)
    _need(target.is_dir() and
          target == Path(getattr(trainer, "_ddp_b16_g4_root", "")).resolve(),
          "G4_ROOT_DRIFT")
    loader = getattr(trainer, "train_loader", None)
    sampler = getattr(loader, "vgra_pair_batch_sampler", None)
    _need(loader is not None and sampler is not None, "LOADER_ABSENT")
    verify_ddp_training_loader(loader, sampler)
    _need(len(loader) == 890 and sampler.rank == rank and
          sampler.seed == 42 and sampler.num_images == 14227 and
          sampler.num_pairs == 6496 and sampler.num_singles == 1235,
          "PAIR_SAMPLER_COUNTS")
    _need(isinstance(getattr(loader, "generator", None), torch.Generator),
          "LOADER_GENERATOR")
    _need(all(getattr(trainer, k, None) is not None for k in
              ("model", "optimizer", "scheduler", "scaler", "ema", "stopper")),
          "INCOMPLETE_TRAINER")
    _need(getattr(trainer.ema, "ema", None) is not None and
          getattr(trainer, "start_epoch", None) == 0 and
          getattr(trainer, "_d4_optimizer_steps_observed", None) == 0,
          "NATIVE_INITIAL_STATE")
    if plan["mode"] == "FRESH_FROM_EARLY":
        _need(plan["cursor"] is None and
              plan["externally_pinned_manifest_sha256"] is None and
              getattr(trainer.ema, "updates", None) == 0, "FRESH_RESUME_DRIFT")
    else:
        validate_cursor(plan["cursor"])
        pin = plan["externally_pinned_manifest_sha256"]
        _need(type(pin) is str and re.fullmatch(r"[0-9a-f]{64}", pin) is not None,
              "EXTERNAL_PIN_REQUIRED")
    return plan


def _two_rank_agreement(rank: int, error: str | None, record: Any, stage: str) -> dict:
    import torch.distributed as dist

    peers = [None, None]
    dist.all_gather_object(peers, {"rank": rank, "error": error, "record": record})
    _need([p["rank"] for p in peers] == [0, 1], stage + "_RANK_MAPPING")
    if any(p["error"] is not None for p in peers):
        raise DDPNativeStateError("DDP_B16_NATIVE_" + stage + "_PEER_FAILED")
    _need(peers[0]["record"] == peers[1]["record"], stage + "_PEER_DRIFT")
    return peers[0]["record"]


def ddp_b16_native_state_start(trainer: Any) -> int:
    """Authenticate two-rank state, restore once, and return optimizer cursor.

    The current review-only worker DOES NOT bind the request or call trainer.train().
    """
    import torch.distributed as dist

    _need(dist.is_available() and dist.is_initialized() and
          dist.get_world_size() == 2 and dist.get_rank() in (0, 1),
          "TWO_RANK_PROCESS_GROUP")
    rank = dist.get_rank()
    plan = payload = record = None
    error = None
    try:
        plan = _check_request(trainer, rank)
        if plan["mode"] == "RESUME_FROM_DDP_B16":
            payload = read_pinned_rank(
                plan["checkpoint_root"], rank=rank, identity=plan["identity"],
                cursor=plan["cursor"],
                externally_pinned_manifest_sha256=plan["externally_pinned_manifest_sha256"],
            )
            state = payload["early_stopper_state"]
            _need(type(state) is dict and
                  math.isfinite(float(state["best_fitness"])) and
                  type(state["best_epoch"]) is int and
                  type(state["possible_stop"]) is bool, "STOPPER_STATE")
            if rank == 0:
                _need(_file_sha(trainer.best) == state["selected_best_sha256"],
                      "SELECTED_BEST_HASH")
            record = {"mode": plan["mode"], "identity": plan["identity"],
                      "cursor": plan["cursor"],
                      "selected_best": state["selected_best_sha256"]}
        else:
            record = {"mode": plan["mode"], "identity": plan["identity"],
                      "cursor": None, "selected_best": None}
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)[:200]
    _two_rank_agreement(rank, error, record, "PREFLIGHT")

    last_opt_step = -1
    error = None
    ready = None
    try:
        if plan["mode"] == "FRESH_FROM_EARLY":
            trainer._ddp_b16_native_restored = False
            trainer._ddp_b16_pinned_resume_verified = False
            trainer._ddp_b16_skip_first_epoch_reset = False
        else:
            loader = trainer.train_loader
            sampler = loader.vgra_pair_batch_sampler
            epoch = plan["cursor"]["next_epoch"]
            sampler.set_epoch(epoch)
            loader.reset()  # must occur BEFORE restore_rank restores global RNG last
            _need(sampler.epoch == epoch, "RESET_CONSUMED_SAMPLER")
            from ultralytics.models.yolo.detect.d4e_transform_rng_state import restore_augmentation_rng
            restored = restore_rank(
                payload, rank=rank, model=trainer.model,
                optimizer=trainer.optimizer, ema_model=trainer.ema.ema,
                scheduler=trainer.scheduler, scaler=trainer.scaler,
                loader_generator=loader.generator, sampler=sampler,
                restore_augmentation=lambda state: restore_augmentation_rng(
                    loader.dataset.transforms, state),
            )
            trainer.start_epoch = restored["next_epoch"]
            trainer.ema.updates = restored["ema_updates"]
            trainer.best_fitness = float(restored["early_stopper_state"]["best_fitness"])
            trainer.stopper.best_fitness = trainer.best_fitness
            trainer.stopper.best_epoch = restored["early_stopper_state"]["best_epoch"]
            trainer.stopper.possible_stop = restored["early_stopper_state"]["possible_stop"]
            trainer.accumulate = restored["accumulate"]
            trainer._d4_optimizer_steps_observed = restored["optimizer_updates"]
            trainer._ddp_b16_native_restored = True
            trainer._ddp_b16_pinned_resume_verified = True
            trainer._ddp_b16_skip_first_epoch_reset = True
            trainer._ddp_b16_first_resume_epoch = trainer.start_epoch
            last_opt_step = restored["last_opt_step"]
        ready = {
            "mode": plan["mode"], "identity": plan["identity"],
            "next_epoch": trainer.start_epoch,
            "last_opt_step": last_opt_step,
            "optimizer_updates": trainer._d4_optimizer_steps_observed,
            "accumulate": int(getattr(trainer, "accumulate", 1)),
        }
        _need(ready["accumulate"] >= 1, "ACCUMULATION")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)[:200]
    _two_rank_agreement(rank, error, ready, "RESTORE")
    trainer._ddp_b16_native_mode = plan["mode"]
    trainer._ddp_b16_native_last_opt_step = last_opt_step
    trainer._ddp_b16_native_state_ready = True
    return last_opt_step


def ddp_b16_begin_epoch(trainer: Any, epoch: int) -> None:
    """Avoid resetting the already-prepared first restored epoch iterator."""
    _need(type(epoch) is int and epoch >= 0 and
          getattr(trainer, "_ddp_b16_native_state_ready", False) is True,
          "EPOCH_OR_STATE")
    loader = trainer.train_loader
    sampler = loader.vgra_pair_batch_sampler
    verify_ddp_training_loader(loader, sampler)
    if getattr(trainer, "_ddp_b16_skip_first_epoch_reset", False):
        _need(trainer._ddp_b16_native_restored is True and
              epoch == trainer._ddp_b16_first_resume_epoch and
              sampler.epoch == epoch, "FIRST_RESUME_EPOCH_DRIFT")
        trainer._ddp_b16_skip_first_epoch_reset = False
        return
    begin_distributed_pair_epoch(loader, epoch)
