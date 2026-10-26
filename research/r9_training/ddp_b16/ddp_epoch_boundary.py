"""One full B-VAL pass on rank 0; broadcast verified metrics to rank 1.

DDP-B16 execution candidate only. This module must be called on BOTH ranks
at a matching epoch boundary, after optimizer synchronization. It does not
read data itself or authorize any GPU run. The private DDP validators enforce
real 3,049-image accounting and paired-forward behavior.
"""
from __future__ import annotations

import math
from typing import Any


class DDPValidationBoundaryError(RuntimeError):
    pass


def rank0_validate_and_sync(trainer: Any, *, epoch: int) -> dict[str, Any]:
    import torch.distributed as dist

    def need(ok: bool, code: str) -> None:
        if not ok:
            raise DDPValidationBoundaryError("DDP_B16_VAL_" + code)

    need(dist.is_available() and dist.is_initialized(), "NO_PROCESS_GROUP")
    rank, world = dist.get_rank(), dist.get_world_size()
    need(world == 2 and rank in (0, 1), "WORLD_SIZE_OR_RANK")
    need(getattr(trainer, "world_size", None) == 2, "TRAINER_WORLD_SIZE")
    need(type(epoch) is int and 0 <= epoch < 100, "EPOCH")
    need(getattr(trainer, "epoch", None) == epoch, "TRAINER_EPOCH")
    need(getattr(trainer, "validator", None) is not None, "MISSING_VALIDATOR")
    need(not bool(getattr(trainer, "_ddp_b16_rank0_validation_ready", False)),
         "REENTRANT_VALIDATION")

    # No stock BaseTrainer.validate(): it broadcasts EMA buffers inside
    # the rank-0-only call and would deadlock. Rank 0 uses the governed
    # pair-aware validator DIRECTLY, never the native image-only fallback.
    dist.barrier()
    wire = [None]
    if rank == 0:
        trainer._ddp_b16_rank0_validation_ready = True
        try:
            validator = trainer.validator
            seen_loader = len(validator.dataloader.dataset)
            need(seen_loader == 3049, "LOADER_POPULATION")
            raw = validator(trainer=trainer)
            need(type(raw) is dict and bool(raw), "METRICS_MISSING")
            need(getattr(validator, "seen", None) == 3049 and
                 getattr(validator, "_d4_fixed_seen", None) == 3049,
                 "EVALUATED_CARDINALITY")
            need("fitness" in raw, "FITNESS_MISSING")
            fitness = float(raw["fitness"])
            need(math.isfinite(fitness), "FITNESS_NONFINITE")
            metrics = {}
            for k, v in raw.items():
                need(type(k) is str and bool(k), "METRIC_KEY")
                val = float(v)
                need(math.isfinite(val), "METRIC_NONFINITE")
                if k != "fitness":
                    metrics[k] = val
            wire[0] = {
                "ok": True, "epoch": epoch, "seen": 3049,
                "fitness": fitness, "metrics": metrics,
            }
        except Exception as exc:
            # Avoid rank-1 waiting forever on a broadcast when rank-0
            # validator throws an ordinary Python exception.
            wire[0] = {"ok": False, "epoch": epoch,
                       "error": type(exc).__name__ + ": " + str(exc)[:240]}
        finally:
            trainer._ddp_b16_rank0_validation_ready = False

    dist.broadcast_object_list(wire, src=0)
    record = wire[0]
    need(type(record) is dict and record.get("epoch") == epoch, "BROADCAST")
    need(record.get("ok") is True, "RANK0_FAILED_" + str(record.get("error", "unknown")))
    need(record.get("seen") == 3049, "GLOBAL_CARDINALITY")

    trainer.metrics = dict(record["metrics"])
    trainer.fitness = record["fitness"]
    best = getattr(trainer, "best_fitness", None)
    if best is None or float(best) < trainer.fitness:
        trainer.best_fitness = trainer.fitness
    dist.barrier()
    return {
        "epoch": epoch, "rank": rank, "operational_val_images": 3049,
        "fitness": trainer.fitness, "metrics": dict(trainer.metrics),
        "validation_was_executed_here": rank == 0,
        "official_training_authorized": False,
    }
