"""Fail-closed DDP-B16 native pre-loop state contract (CPU review only).

This is deliberately NOT a recovery implementation or permission to train.
The real torchrun worker must independently authenticate the staged C1 bytes,
approved assets, DDP recipe and BOTH rank state capsules before setting any
runtime binding. A hand-set sentinel is never an acceptable evidence chain.
Original batch-eight D4E/G3B4 configuration is always rejected for DDP-B16.
"""
from __future__ import annotations

from pathlib import Path

from .ddp_g4_recovery import validate_identity


class DDPB16StartupError(RuntimeError):
    pass


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise DDPB16StartupError("DDP_B16_PRELOOP_" + code)


def require_ddp_b16_native_preloop(trainer) -> None:
    """Reject training before a verified DDP-specific fresh/resume state exists.

    Called by the private native engine AFTER _setup_train, before the first
    optimizer step, and on both ranks. It cannot create readiness or restore
    state. The future worker must provide an independently tested binding path.
    """
    args = getattr(trainer, "args", None)
    _need(type(getattr(trainer, "world_size", None)) is int and
          trainer.world_size == 2, "WORLD_SIZE")
    _need(not getattr(trainer, "_d4e_recovery_configured", False), "LEGACY_D4E_FORBIDDEN")
    _need(not getattr(trainer, "_g3b4_ready", False), "LEGACY_G3B4_FORBIDDEN")
    _need(not getattr(trainer, "_d4e_recovery_restored", False), "LEGACY_RESUME_FORBIDDEN")
    _need(getattr(trainer, "_ddp_b16_verified_worker_ready", None) is True,
          "AUTHENTICATED_WORKER_MISSING")
    _need(getattr(trainer, "_ddp_b16_g4_native_bound", None) is True,
          "TWO_RANK_G4_UNBOUND")
    identity = getattr(trainer, "_ddp_b16_g4_identity", None)
    try:
        validate_identity(identity)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise DDPB16StartupError("DDP_B16_PRELOOP_IDENTITY_INVALID") from exc
    arm = ("MV00" if type(trainer).__name__ == "EarlyPairedTrainer" else
           "MV01" if type(trainer).__name__ == "VGRADetectionTrainer" else None)
    _need(arm is not None and identity["arm"] == arm, "ARM")
    root = getattr(trainer, "_ddp_b16_g4_root", None)
    _need(isinstance(root, (str, Path)), "G4_ROOT")
    checkpoint_root = Path(root)
    _need(checkpoint_root.is_dir() and not checkpoint_root.is_symlink(),
          "G4_ROOT_NOT_REGULAR")
    _need(args is not None and
          (getattr(args, "batch", None), getattr(args, "imgsz", None),
           getattr(args, "epochs", None), getattr(args, "seed", None),
           getattr(args, "workers", None), getattr(args, "nbs", None)) ==
          (16, 1024, 100, 42, 0, 64), "RECIPE")
    _need(str(getattr(args, "optimizer", "")).upper() == "SGD" and
          getattr(args, "amp", False) is True and
          getattr(args, "resume", False) is False and
          getattr(args, "val", False) is True and
          getattr(args, "save", False) is True, "GOVERNED_OPTIONS")
    loader = getattr(trainer, "train_loader", None)
    sampler = getattr(loader, "vgra_pair_batch_sampler", None)
    _need(loader is not None and len(loader) == 890 and sampler is not None,
          "PAIR_LOADER")
    _need(getattr(sampler, "rank", None) in (0, 1) and
          getattr(sampler, "num_images", None) == 14227 and
          getattr(sampler, "num_pairs", None) == 6496 and
          getattr(sampler, "num_singles", None) == 1235, "PAIR_POPULATION")
    # This marker is intentionally NOT set by the existing review-only worker.
    # Supplying an arbitrary True without future source-and-G4 evidence is
    # NOT a release mechanism; no executable child exists at this milestone.
    _need(getattr(trainer, "_ddp_b16_native_state_ready", None) is True,
          "FRESH_OR_RESUME_STATE_NOT_IMPLEMENTED")
    mode = getattr(trainer, "_ddp_b16_native_mode", None)
    _need(mode in ("FRESH_FROM_EARLY", "RESUME_FROM_DDP_B16"), "MODE")
    if mode == "RESUME_FROM_DDP_B16":
        _need(getattr(trainer, "_ddp_b16_pinned_resume_verified", None) is True,
              "OFFSESSION_RESUME_NOT_VERIFIED")
