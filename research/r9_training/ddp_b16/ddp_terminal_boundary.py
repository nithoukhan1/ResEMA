"""Two-rank terminal selected-BEST B-VAL candidate; NOT a training release.

Call collectively on BOTH DDP ranks after the native terminal epoch. Only
rank 0 may invoke the governed paired validator via the original arm-specific
final_eval(). Rank 1 waits and receives the finite metrics and checkpoint
digests. This never runs B-TEST and never performs an export or authorizes
a GPU experiment. Real Kaggle NCCL and selected-BEST B-VAL remain untested.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


class DDPB16TerminalError(RuntimeError):
    pass


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise DDPB16TerminalError("DDP_B16_TERMINAL_" + code)


def _sha(path: Any) -> str:
    p = Path(path)
    _need(p.is_file() and not p.is_symlink(), "SELECTED_NATIVE_CHECKPOINT_MISSING")
    h = hashlib.sha256()
    with p.open("rb") as reader:
        for piece in iter(lambda: reader.read(1 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def rank0_selected_best_terminal(trainer: Any) -> dict[str, Any]:
    """No rank-1 validation; explicit error transport across two ranks."""
    import torch.distributed as dist

    _need(dist.is_available() and dist.is_initialized(), "PROCESS_GROUP")
    rank, world = dist.get_rank(), dist.get_world_size()
    _need(world == 2 and rank in (0, 1), "TWO_RANKS")
    local_error = None
    try:
        _need(getattr(trainer, "world_size", None) == 2, "TRAINER_WORLD_SIZE")
        _need(getattr(trainer, "_ddp_b16_terminal_candidate_ready", None) is True,
              "TERMINAL_NOT_AUTHORIZED")
        _need(getattr(trainer, "_ddp_b16_g4_native_bound", None) is True,
              "DDP_G4_NOT_BOUND")
        _need(not getattr(trainer, "_d4e_recovery_configured", False),
              "LEGACY_RECOVERY_FORBIDDEN")
        data = getattr(trainer, "data", None)
        _need(type(data) is dict and bool(data.get("val")) and
              not data.get("test"), "BVAL_ONLY_TEST_SEALED")
        validator = getattr(trainer, "validator", None)
        _need(validator is not None and
              len(validator.dataloader.dataset) == 3049, "BVAL_POPULATION")
        _need(getattr(trainer, "_ddp_b16_rank0_validation_ready", False) is False,
              "REENTRANT_VALIDATION")
    except Exception as exc:
        local_error = type(exc).__name__ + ": " + str(exc)[:200]
    preflight = [None, None]
    dist.all_gather_object(preflight, {"rank": rank, "error": local_error})
    _need([record["rank"] for record in preflight] == [0, 1],
          "PREFLIGHT_RANK_ORDER")
    _need(all(record["error"] is None for record in preflight),
          "PREFLIGHT_" + str([x["error"] for x in preflight if x["error"]][:1]))

    wire = [None]
    if rank == 0:
        trainer._ddp_b16_rank0_validation_ready = True
        try:
            best_sha_before = _sha(trainer.best)
            last_sha = _sha(trainer.last)
            trainer.final_eval()  # existing MV00/MV01 BEST-aware paired B-VAL
            validator = trainer.validator
            _need(getattr(validator, "seen", None) == 3049 and
                  getattr(validator, "_d4_fixed_seen", None) == 3049,
                  "FINAL_BVAL_CARDINALITY")
            _need(_sha(trainer.best) == best_sha_before,
                  "BEST_CHECKPOINT_CHANGED_DURING_EVALUATION")
            raw = getattr(trainer, "metrics", None)
            _need(type(raw) is dict and bool(raw), "FINAL_METRICS")
            metrics = {}
            for key, value in raw.items():
                _need(type(key) is str and bool(key), "METRIC_KEY")
                number = float(value)
                _need(math.isfinite(number), "METRIC_NONFINITE")
                metrics[key] = number
            wire[0] = {
                "ok": True, "val_images": 3049, "metrics": metrics,
                "selected_best_sha256": best_sha_before,
                "native_last_sha256": last_sha,
            }
        except Exception as exc:
            wire[0] = {"ok": False, "reason": type(exc).__name__ + ": " + str(exc)[:240]}
        finally:
            trainer._ddp_b16_rank0_validation_ready = False

    dist.broadcast_object_list(wire, src=0)
    record = wire[0]
    _need(type(record) is dict and record.get("ok") is True,
          "RANK0_FAILURE_" + str(record.get("reason", "unknown")))
    _need(record["val_images"] == 3049 and
          bool(record["selected_best_sha256"]) and
          bool(record["native_last_sha256"]), "BROADCAST")
    trainer.metrics = dict(record["metrics"])
    dist.barrier()
    return {
        "rank": rank, "selected_best_sha256": record["selected_best_sha256"],
        "native_last_sha256": record["native_last_sha256"],
        "metrics": dict(trainer.metrics), "operational_val_images": 3049,
        "evaluated_selected_best_on_this_rank": rank == 0,
        "terminal_export_authorized": False,
        "official_training_authorized": False,
    }
