"""Pinned-source MV01 DDP-B16 multi-epoch Kaggle continuation (nonofficial).

The scientific source stays at f440d329d4544ec08ad63c1972bcf9a8b8a7eed8.
Real torchrun ranks separately authenticate R0, source stage and external G4;
the fixed native trainer handles all optimizer, augmentation and B-VAL math.
Only the verified epoch-G4 boundary may stop a session. B-TEST is sealed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/kaggle/working/MV01_B16_CONTINUOUS")
REPO = ROOT / "repo"
G4 = ROOT / "g4"
OUT = Path("/kaggle/working/MV01_B16_CONTINUOUS_OUTPUT")
PLAN = ROOT / "plan.json"
EXTERNAL_PIN = os.environ["DDP_B16_PREVIOUS_MANIFEST_SHA256"]
BUDGET_HOURS = float(os.environ.get("DDP_B16_SESSION_HOURS", "9.0"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

from research.r9_training.ddp_b16.ddp_worker import review_worker_inputs, _unique_json
from research.r9_training.ddp_b16.ddp_verified_admission import verify_staged_admission
from research.r9_training.ddp_b16.ddp_rank_admission import require_review_consensus
from research.r9_training.ddp_b16.ddp_native_binding import make_native_binding
from research.r9_training.ddp_b16.ddp_native_binding_consensus import require_native_binding_consensus
from research.r9_training.ddp_b16.ddp_native_bootstrap_candidate import verify_two_t4_runtime, load_pinned_native_class
from research.r9_training.ddp_b16.ddp_native_state import SCHEMA as STATE_SCHEMA
from research.r9_training.ddp_b16 import ddp_native_g4_hook as hook


class EpochBoundaryStop(Exception):
    pass


def require(value, code):
    if not value:
        raise RuntimeError("MV01_MULTI_EPOCH_" + code)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_exclusive_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def persist_epoch(trainer, result, identity, epoch, previous_sha):
    """Rank 0: SHA-verify all six members and rotate one resumable output."""
    step = epoch + 1
    filename = f'{identity["run_id"]}_epoch{step:03d}_two_rank_manifest.json'
    manifest_path = G4 / filename
    require(manifest_path.is_file() and
            sha256(manifest_path) == result["manifest_sha256"], "MANIFEST_HASH")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    require(manifest.get("complete") is True and
            manifest.get("identity") == identity and
            manifest.get("cursor", {}).get("next_epoch") == step and
            manifest["cursor"]["completed_epoch"] == epoch and
            manifest.get("b_test_access") == "NONE" and
            set(manifest["ranks"]) == {"0", "1"}, "MANIFEST_CONTENT")
    files = {filename: result["manifest_sha256"],
             "best.pt": result["selected_best_sha256"]}
    for rank in (0, 1):
        member = manifest["ranks"][str(rank)]
        files[member["filename"]] = member["sha256"]
        files[member["receipt_filename"]] = member["receipt_sha256"]
    require(len(files) == 6, "SIX_G4_MEMBERS")
    temporary = OUT / f".next_{step:03d}"
    require(not temporary.exists(), "TEMP_EXISTS")
    temporary.mkdir()
    for name, digest in files.items():
        require(Path(name).name == name, "PATH_TRAVERSAL")
        origin = Path(trainer.best) if name == "best.pt" else G4 / name
        require(origin.is_file() and not origin.is_symlink() and
                sha256(origin) == digest, "G4_SOURCE_SHA")
        shutil.copyfile(origin, temporary / name)
        require(sha256(temporary / name) == digest, "G4_EXPORT_SHA")
    receipt = {
        "schema": "MV01_DDP_B16_CONTINUOUS_OUTPUT_V1",
        "status": "MV01_RESUMABLE_EPOCH_G4_CANDIDATE",
        "identity": identity, "run_id": identity["run_id"],
        "source_release_sha": identity["ddp_release_sha"],
        "completed_epoch": step, "next_epoch": step,
        "optimizer_updates": manifest["cursor"]["optimizer_updates"],
        "manifest_sha256": result["manifest_sha256"],
        "previous_manifest_sha256": previous_sha,
        "selected_best_sha256": result["selected_best_sha256"],
        "files_sha256": files, "b_test_access": "NONE",
        "official_training_authorized": False,
        "offsession_restoration_verified": False,
    }
    write_exclusive_json(temporary / "ACCEPTANCE.json", receipt)
    latest, previous = OUT / "g4", OUT / ".previous_g4"
    require(not previous.exists(), "PREVIOUS_ROTATION_UNFINISHED")
    if latest.exists():
        latest.rename(previous)
    temporary.rename(latest)
    if previous.exists():
        shutil.rmtree(previous)
    # Prune only generated work-state copies; the newest exported G4 is intact.
    for path in G4.iterdir():
        if path.is_file() and path.name not in files and path.name != "best.pt":
            path.unlink()
    return receipt


def persist_terminal(trainer, identity):
    """Executed only after native selected-BEST final B-VAL succeeds."""
    require(trainer.epoch == 99 and
            trainer.validator.seen == 3049 and
            trainer.validator._d4_fixed_seen == 3049, "TERMINAL_BVAL")
    metrics = {str(k): float(v) for k, v in trainer.metrics.items()}
    require(bool(metrics) and all(math.isfinite(x) for x in metrics.values()),
            "TERMINAL_METRICS")
    terminal = OUT / "terminal"
    require(not terminal.exists(), "TERMINAL_OUTPUT_EXISTS")
    terminal.mkdir()
    files = {}
    for name, path in (("best.pt", Path(trainer.best)),
                       ("last.pt", Path(trainer.last))):
        require(path.is_file() and not path.is_symlink(),
                "TERMINAL_CHECKPOINT")
        files[name] = sha256(path)
        shutil.copyfile(path, terminal / name)
        require(sha256(terminal / name) == files[name], "TERMINAL_SHA")
    receipt = {
        "schema": "MV01_DDP_B16_TERMINAL_V1",
        "status": "EPOCH100_FINISHED_SELECTED_BEST_BVAL_CANDIDATE",
        "identity": identity, "completed_epochs": 100,
        "metrics": metrics, "operational_val_images": 3049,
        "files_sha256": files, "b_test_access": "NONE",
        "official_training_authorized": False,
    }
    write_exclusive_json(terminal / "TERMINAL.json", receipt)
    return receipt


def main():
    rank, local, world = (int(os.environ[k])
                          for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1" and
            (world, rank, local) in ((2, 0, 0), (2, 1, 1)), "RANKS")
    require(0.5 <= BUDGET_HOURS <= 10.5, "SESSION_BUDGET")
    plan = _unique_json(PLAN)
    manifests = list(G4.glob("*_two_rank_manifest.json"))
    require(len(manifests) == 1 and
            sha256(manifests[0]) == EXTERNAL_PIN, "EXTERNAL_MANIFEST_PIN")
    manifest = json.loads(manifests[0].read_text("utf-8"))
    identity, cursor = manifest["identity"], manifest["cursor"]
    require(identity["arm"] == "MV01" and
            identity["run_id"] == plan["run_id"] and
            identity["ddp_release_sha"] == plan["expected_git_head"] and
            1 <= cursor["next_epoch"] < 100, "RESUME_IDENTITY")
    require(G4.is_dir() and OUT.is_dir() and not (OUT / "g4").exists(),
            "NEW_OUTPUT_REQUIRED")

    dist.init_process_group("gloo", init_method="env://",
                            rank=rank, world_size=2,
                            timeout=timedelta(seconds=600))
    try:
        review = review_worker_inputs(plan, stage_source=False)
        admission = verify_staged_admission(plan, review)
        require_review_consensus(admission)
        config = _unique_json(Path(plan["r0_input_config"]))
        binding = make_native_binding(
            plan, admission, config, project=str(ROOT / "native_project"))
        require_native_binding_consensus(binding)
        require(binding["identity"] == identity and
                binding["overrides"]["epochs"] == 100 and
                binding["overrides"]["batch"] == 16 and
                binding["overrides"]["resume"] is False, "FROZEN_RECIPE")
        verify_two_t4_runtime(torch.cuda)
        native = load_pinned_native_class(binding)
        require(native.__name__ == "VGRADetectionTrainer", "NATIVE_ARM")
        trainer = native(**binding["constructor"],
                         overrides=binding["overrides"])
        require(trainer.world_size == 2 and trainer.ddp is False,
                "TWO_RANK_NATIVE")
        if rank == 0:
            target = Path(trainer.best)
            require(not target.exists(), "NATIVE_BEST_ALREADY_EXISTS")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(G4 / "best.pt", target)
            require(sha256(target) == sha256(G4 / "best.pt"),
                    "BEST_COPY_SHA")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    # Historical FRESH plan above authenticates the R0 input only.
    # The actual optimizer/model/EMA/RNG state is always native RESUME.
    trainer._ddp_b16_g4_identity = dict(identity)
    trainer._ddp_b16_g4_root = str(G4.resolve(strict=True))
    trainer._ddp_b16_native_state_request = {
        "schema": STATE_SCHEMA, "mode": "RESUME_FROM_DDP_B16",
        "identity": identity, "cursor": cursor,
        "checkpoint_root": str(G4.resolve(strict=True)),
        "externally_pinned_manifest_sha256": EXTERNAL_PIN,
    }
    trainer._ddp_b16_verified_worker_ready = True
    trainer._ddp_b16_g4_native_bound = True
    trainer._ddp_b16_terminal_candidate_ready = True

    # Validate the REAL native post-G4 restore, before any training batch.
    # The nominal launcher epoch alone is NOT sufficient evidence.
    from research.r9_training.ddp_b16 import ddp_native_state as native_state
    _real_native_start = native_state.ddp_b16_native_state_start
    _expected_cursor = int(os.environ["DDP_B16_EXPECTED_NEXT_EPOCH_INDEX"])
    _expected_updates = int(os.environ["DDP_B16_EXPECTED_OPTIMIZER_UPDATES"])

    def _checked_native_state_start(bound_trainer):
        last_step = _real_native_start(bound_trainer)
        require(getattr(bound_trainer, "_ddp_b16_native_restored", False) is True,
                "NATIVE_G4_RESTORATION_NOT_CONFIRMED")
        require(bound_trainer.start_epoch == _expected_cursor,
                "REAL_NATIVE_RESTORE_WRONG_EPOCH")
        require(getattr(bound_trainer, "_d4_optimizer_steps_observed", None)
                == _expected_updates, "REAL_NATIVE_OPTIMIZER_CURSOR_DRIFT")
        require(dist.is_initialized() and dist.get_backend() == "nccl",
                "REAL_NATIVE_NCCL_BACKEND")
        if rank == 0:
            print("STRICT_NATIVE_RESUME_ACCEPTED=" +
                  json.dumps({"starting_epoch": _expected_cursor + 1,
                              "completed_epochs": _expected_cursor,
                              "optimizer_updates": _expected_updates,
                              "manifest_sha256": EXTERNAL_PIN},sort_keys=True),
                  flush=True)
        return last_step

    native_state.ddp_b16_native_state_start = _checked_native_state_start

    start = time.monotonic()
    last_epoch_end = start
    state = {"manifest": EXTERNAL_PIN, "next_epoch": cursor["next_epoch"]}
    old_hook = hook.ddp_b16_g4_epoch_end

    def session_hook(obj, *, epoch, nb, last_opt_step):
        nonlocal last_epoch_end
        result = old_hook(obj, epoch=epoch, nb=nb, last_opt_step=last_opt_step)
        if epoch == 99:
            require(result.get("status") ==
                    "TERMINAL_EPOCH_G4_CAPSULE_NOT_WRITTEN",
                    "TERMINAL_STATUS")
            return result
        require(result.get("status") ==
                "LOCAL_SYNCHRONIZED_DDP_CAPSULE_EXTERNAL_AUTHENTICATION_PENDING",
                "EPOCH_G4")
        require(epoch == state["next_epoch"], "EPOCH_SEQUENCE")
        consensus = [None, None]
        dist.all_gather_object(consensus, {
            "rank": rank,
            "manifest": result["manifest_sha256"],
            "best": result["selected_best_sha256"],
        })
        require([x["rank"] for x in consensus] == [0, 1] and
                len({x["manifest"] for x in consensus}) == 1 and
                len({x["best"] for x in consensus}) == 1,
                "G4_RANK_CONSENSUS")
        message = [None]
        if rank == 0:
            try:
                saved = persist_epoch(obj, result, identity, epoch,
                                      state["manifest"])
                now = time.monotonic()
                elapsed = now - start
                this_epoch = now - last_epoch_end
                stop = (elapsed >= BUDGET_HOURS * 3600 or
                        elapsed + max(5400.0, 1.5 * this_epoch)
                        >= 11.0 * 3600)
                message[0] = {
                    "ok": True, "stop": stop,
                    "manifest_sha256": saved["manifest_sha256"],
                }
            except Exception as exc:
                message[0] = {"ok": False, "error": str(exc)[:300]}
        dist.broadcast_object_list(message, src=0)
        require(message[0]["ok"],
                "EXPORT_" + str(message[0].get("error", "failed")))
        require(message[0]["manifest_sha256"] == result["manifest_sha256"],
                "EXPORT_HASH_RANK")
        state["manifest"] = result["manifest_sha256"]
        state["next_epoch"] = epoch + 1
        last_epoch_end = time.monotonic()
        dist.barrier()
        if message[0]["stop"]:
            raise EpochBoundaryStop("SESSION_BUDGET_REACHED_AFTER_G4")
        return result

    hook.ddp_b16_g4_epoch_end = session_hook
    try:
        safe = [np._core.multiarray.scalar, np.dtype]
        safe.extend(type(np.dtype(n)) for n in (
            "float16", "float32", "float64", "int16", "int32",
            "int64", "uint8", "uint16", "uint32", "uint64", "bool",
        ))
        safe = list(dict.fromkeys(safe))
        try:
            with torch.serialization.safe_globals(safe):
                trainer.train()
            require(state["next_epoch"] == 99 and trainer.epoch == 99,
                    "UNEXPECTED_NATIVE_STOP")
            status = "EPOCH100_COMPLETED"
        except EpochBoundaryStop:
            status = "PAUSED_AT_G4"

        done = [None]
        if rank == 0:
            try:
                terminal = persist_terminal(trainer, identity) if (
                    status == "EPOCH100_COMPLETED") else None
                done[0] = {
                    "ok": True, "status": status,
                    "next_epoch": 100 if terminal else state["next_epoch"],
                    "manifest_sha256": state["manifest"],
                    "terminal": terminal,
                    "training_executed": True,
                    "b_test_access": "NONE",
                }
            except Exception as exc:
                done[0] = {"ok": False, "error": str(exc)[:300]}
        dist.broadcast_object_list(done, src=0)
        require(done[0]["ok"],
                "TERMINAL_" + str(done[0].get("error", "unknown")))
        if rank == 0:
            print("MV01_CONTINUOUS_RESULT=" +
                  json.dumps(done[0], sort_keys=True), flush=True)
        dist.barrier()
    finally:
        hook.ddp_b16_g4_epoch_end = old_hook
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
