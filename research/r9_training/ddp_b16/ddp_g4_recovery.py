"""VGRA DDP-B16 two-rank epoch-boundary recovery candidate (NOT RUN AUTHORITY).

Each rank writes a complete local state independently. A separate manifest is
published ONLY after both immutable rank files exist and match the same frozen
identity/cursor. Resume requires a SHA256 pin obtained OFF-SESSION, prior to
loading weights_only Torch bytes. Actual native loop integration, G4 publishing,
cross-session CUDA replay, and Kaggle approval remain BLOCKED.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import random
import re
from pathlib import Path
from typing import Any


SCHEMA = "VGRA_DDP_B16_G4_EPOCH_PAIR_CANDIDATE_V1"
SHA = re.compile(r"^[0-9a-f]{64}$")
OID = re.compile(r"^[0-9a-f]{40}$")
RUN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ID_KEYS = {
    "arm", "run_id", "ddp_release_sha", "ddp_recipe_sha256",
    "assignment_sha256", "pretrained_sha256", "stage_source_manifest_sha256",
}
CURSOR_KEYS = {
    "completed_epoch", "next_epoch", "steps_per_rank", "global_batch",
    "optimizer_updates", "last_opt_step", "accumulate", "sampler_next_epoch",
}
STOPPER_KEYS = {"best_fitness", "best_epoch", "possible_stop", "selected_best_sha256"}


class DDPG4Error(RuntimeError):
    pass


def _need(ok: bool, error: str) -> None:
    if not ok:
        raise DDPG4Error("DDP_G4_" + error)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def validate_identity(identity: Any) -> None:
    _need(type(identity) is dict and set(identity) == ID_KEYS, "IDENTITY_SCHEMA")
    _need(identity["arm"] in ("MV00", "MV01"), "ARM")
    _need(type(identity["run_id"]) is str and bool(RUN.fullmatch(identity["run_id"])), "RUN_ID")
    _need(type(identity["ddp_release_sha"]) is str and bool(OID.fullmatch(identity["ddp_release_sha"])), "RELEASE")
    for field in ID_KEYS - {"arm", "run_id", "ddp_release_sha"}:
        _need(type(identity[field]) is str and bool(SHA.fullmatch(identity[field])), "IDENTITY_SHA_" + field)


def validate_cursor(cursor: Any) -> None:
    _need(type(cursor) is dict and set(cursor) == CURSOR_KEYS, "CURSOR_SCHEMA")
    _need(all(type(v) is int for v in cursor.values()), "CURSOR_TYPES")
    e = cursor["completed_epoch"]
    _need(0 <= e < 99 and cursor["next_epoch"] == e + 1, "CURSOR_EPOCH")
    _need(cursor["sampler_next_epoch"] == e + 1, "SAMPLER_EPOCH")
    _need(cursor["steps_per_rank"] == 890 and cursor["global_batch"] == 16, "BATCH_RECIPE")
    _need(cursor["optimizer_updates"] >= 0 and cursor["accumulate"] >= 1, "OPTIMIZER_CURSOR")
    _need(-1 <= cursor["last_opt_step"] < (e + 1) * 890, "LAST_OPT_STEP")


def _path(root: str | Path, filename: str, *, exists: bool) -> Path:
    r = Path(root).resolve(strict=True)
    _need(r.is_dir() and not r.is_symlink(), "ROOT_NOT_DIRECTORY")
    _need("/" not in filename and "\\" not in filename and filename not in (".", ".."), "FILE_NAME")
    p = r / filename
    _need(not p.is_symlink(), "SYMLINK")
    if exists:
        _need(p.is_file(), "FILE_MISSING")
    else:
        _need(not p.exists(), "FILE_OVERWRITE")
    return p


def _to_cpu(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return copy.deepcopy(value)


def _finite(value: Any) -> None:
    import math
    import torch
    if isinstance(value, torch.Tensor):
        _need(not (value.is_floating_point() or value.is_complex()) or
              bool(torch.isfinite(value).all().item()), "NONFINITE_TENSOR")
    elif isinstance(value, float):
        _need(math.isfinite(value), "NONFINITE_SCALAR")
    elif isinstance(value, dict):
        for v in value.values():
            _finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _finite(v)


def _rng_capture(local_rank: int) -> dict[str, Any]:
    import numpy as np
    import torch
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(state[0]),
            "keys": torch.as_tensor(state[1].copy(), dtype=torch.uint32),
            "pos": int(state[2]), "has_gauss": int(state[3]),
            "cached": float(state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_local": torch.cuda.get_rng_state(local_rank).cpu().clone()
        if torch.cuda.is_available() else None,
    }


def _rng_restore(data: dict[str, Any], rank: int) -> None:
    import numpy as np
    import torch
    # Must run LAST after loader iterator/optimizer/model restoration.
    random.setstate(data["python"])
    n = data["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].cpu().numpy(),
                         int(n["pos"]), int(n["has_gauss"]), float(n["cached"])))
    torch.set_rng_state(data["torch_cpu"].cpu())
    if data["torch_cuda_local"] is not None:
        _need(torch.cuda.is_available() and torch.cuda.device_count() == 2, "CUDA_TOPOLOGY")
        torch.cuda.set_rng_state(data["torch_cuda_local"].cpu(), device=rank)


def capture_rank(
    *, rank: int, identity: dict, cursor: dict, model: Any, optimizer: Any,
    ema_model: Any, ema_updates: int, scheduler: Any, scaler: Any,
    loader_generator: Any, augmentation_rng: dict,
    stopper: dict,
) -> dict[str, Any]:
    """Capture LOCAL state; DDP-wrapped models are unwrapped, never discarded."""
    import torch
    validate_identity(identity)
    validate_cursor(cursor)
    _need(type(rank) is int and rank in (0, 1), "RANK")
    _need(type(ema_updates) is int and ema_updates >= 0, "EMA_UPDATES")
    _need(type(stopper) is dict and set(stopper) == STOPPER_KEYS, "STOPPER_SCHEMA")
    _need(bool(SHA.fullmatch(stopper["selected_best_sha256"])), "BEST_SHA")
    _need(type(stopper["best_epoch"]) is int and
          type(stopper["possible_stop"]) is bool, "STOPPER_TYPES")
    _finite(stopper["best_fitness"])
    _need(type(augmentation_rng) is dict and
          augmentation_rng.get("schema") == "D4E_TRAIN_ALBUMENTATIONS_RNG_R2" and
          isinstance(augmentation_rng.get("entries"), list), "AUGMENT_RNG")
    _need(isinstance(loader_generator, torch.Generator), "LOADER_GENERATOR")
    _need(model is not None and optimizer is not None and
          ema_model is not None and scheduler is not None and scaler is not None,
          "MISSING_TRAINING_COMPONENT")
    native = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    grads = {name: _to_cpu(p.grad) if p.grad is not None else None
             for name, p in native.named_parameters()}
    result = {
        "schema": SCHEMA, "rank": rank,
        "identity": dict(identity), "cursor": dict(cursor),
        "model_state": _to_cpu(native.state_dict()),
        "parameter_grads": grads,
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "ema_state": _to_cpu(ema_model.state_dict()), "ema_updates": ema_updates,
        "scheduler_state": _to_cpu(scheduler.state_dict()),
        "scaler_state": _to_cpu(scaler.state_dict()),
        "loader_generator_state": loader_generator.get_state().cpu().clone(),
        "augmentation_rng": _to_cpu(augmentation_rng),
        "early_stopper_state": _to_cpu(stopper),
        "rng_state": _rng_capture(rank),
    }
    _finite(result)
    return result


def save_rank(payload: dict[str, Any], root: str | Path) -> dict[str, Any]:
    """Write one rank file and receipt exclusively; never claim offsite safety."""
    import torch
    _need(type(payload) is dict and payload.get("schema") == SCHEMA, "PAYLOAD_SCHEMA")
    validate_identity(payload["identity"])
    validate_cursor(payload["cursor"])
    rank = payload.get("rank")
    _need(type(rank) is int and rank in (0, 1), "RANK")
    stem = f'{payload["identity"]["run_id"]}_epoch{payload["cursor"]["next_epoch"]:03d}_rank{rank}'
    path = _path(root, stem + ".pt", exists=False)
    receipt_path = _path(root, stem + ".receipt.json", exists=False)
    buf = io.BytesIO()
    torch.save(payload, buf)
    data = buf.getvalue()
    meta = {
        "schema": SCHEMA, "rank": rank, "filename": path.name,
        "sha256": _sha(data), "identity": payload["identity"],
        "cursor": payload["cursor"],
        "status": "LOCAL_RANK_CHECKPOINT_NOT_OFFSITE_VERIFIED",
    }
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    with receipt_path.open("xb") as stream:
        stream.write(_canonical(meta))
        stream.flush()
        os.fsync(stream.fileno())
    return meta


def publish_pair(root: str | Path, *, identity: dict, cursor: dict) -> dict[str, Any]:
    """Publish manifest only when BOTH rank receipts and checkpoint hashes match."""
    validate_identity(identity)
    validate_cursor(cursor)
    stem = f'{identity["run_id"]}_epoch{cursor["next_epoch"]:03d}'
    manifest_path = _path(root, stem + "_two_rank_manifest.json", exists=False)
    records = {}
    for rank in (0, 1):
        name = f"{stem}_rank{rank}"
        receipt = _path(root, name + ".receipt.json", exists=True)
        data = json.loads(receipt.read_text("utf8"))
        _need(data.get("schema") == SCHEMA and data.get("rank") == rank, "RECEIPT_SCHEMA")
        _need(data.get("identity") == identity and data.get("cursor") == cursor, "CROSS_RANK_IDENTITY")
        checkpoint = _path(root, name + ".pt", exists=True)
        _need(data.get("filename") == checkpoint.name and
              data.get("sha256") == _sha(checkpoint.read_bytes()), "RANK_SHA")
        records[str(rank)] = {
            "filename": checkpoint.name, "sha256": data["sha256"],
            "receipt_filename": receipt.name, "receipt_sha256": _sha(receipt.read_bytes()),
        }
    manifest = {
        "schema": SCHEMA, "identity": identity, "cursor": cursor,
        "ranks": records, "world_size": 2, "complete": True,
        "status": "LOCAL_BOTH_RANKS_PRESENT_EXTERNAL_AUTHENTICATION_PENDING",
        "official_training_authorized": False, "gpu_acceptance": False,
        "b_test_access": "NONE",
    }
    with manifest_path.open("xb") as stream:
        stream.write(_canonical(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "manifest_file": manifest_path.name,
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "status": manifest["status"],
        "offsite_persistence_verified": False,
    }


def read_pinned_rank(
    root: str | Path, *, rank: int, identity: dict, cursor: dict,
    externally_pinned_manifest_sha256: str,
) -> dict[str, Any]:
    """Require independent external manifest SHA before ANY torch.load."""
    import torch
    validate_identity(identity)
    validate_cursor(cursor)
    _need(type(rank) is int and rank in (0, 1), "RANK")
    _need(type(externally_pinned_manifest_sha256) is str and
          bool(SHA.fullmatch(externally_pinned_manifest_sha256)), "EXTERNAL_MANIFEST_PIN")
    stem = f'{identity["run_id"]}_epoch{cursor["next_epoch"]:03d}'
    manifest_file = _path(root, stem + "_two_rank_manifest.json", exists=True)
    manifest_bytes = manifest_file.read_bytes()
    _need(_sha(manifest_bytes) == externally_pinned_manifest_sha256, "EXTERNAL_MANIFEST_SHA")
    manifest = json.loads(manifest_bytes.decode("utf8"))
    _need(manifest.get("schema") == SCHEMA and manifest.get("complete") is True and
          manifest.get("world_size") == 2 and manifest.get("identity") == identity and
          manifest.get("cursor") == cursor, "MANIFEST_IDENTITY")
    _need(set(manifest.get("ranks", {})) == {"0", "1"}, "MISSING_RANK")
    for i in (0, 1):
        record = manifest["ranks"][str(i)]
        name = f"{stem}_rank{i}"
        _need(record["filename"] == name + ".pt" and
              record["receipt_filename"] == name + ".receipt.json", "RANK_FILENAMES")
        receipt = _path(root, record["receipt_filename"], exists=True)
        _need(_sha(receipt.read_bytes()) == record["receipt_sha256"], "RECEIPT_SHA")
        receipt_data = json.loads(receipt.read_text("utf8"))
        _need(receipt_data.get("sha256") == record["sha256"] and
              receipt_data.get("rank") == i and
              receipt_data.get("identity") == identity and
              receipt_data.get("cursor") == cursor, "RECEIPT_CONTENT")
        checkpoint = _path(root, record["filename"], exists=True)
        _need(_sha(checkpoint.read_bytes()) == record["sha256"], "CHECKPOINT_SHA")
    own = manifest["ranks"][str(rank)]
    blob = _path(root, own["filename"], exists=True).read_bytes()
    # weights_only avoids arbitrary Python object constructors.
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    _need(payload.get("schema") == SCHEMA and payload.get("rank") == rank and
          payload.get("identity") == identity and payload.get("cursor") == cursor,
          "PAYLOAD_IDENTITY")
    _finite(payload)
    return payload


def restore_rank(
    payload: dict[str, Any], *, rank: int, model: Any, optimizer: Any,
    ema_model: Any, scheduler: Any, scaler: Any, loader_generator: Any,
    restore_augmentation: Any,
) -> dict[str, Any]:
    """Restore captured state. Caller resets the verified local loader FIRST."""
    import torch
    _need(payload.get("rank") == rank, "RESTORE_RANK")
    native = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    native.load_state_dict(payload["model_state"], strict=True)
    ema_model.load_state_dict(payload["ema_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    names = {name for name, _ in native.named_parameters()}
    _need(set(payload["parameter_grads"]) == names, "GRAD_NAMES")
    for name, param in native.named_parameters():
        value = payload["parameter_grads"][name]
        param.grad = None if value is None else value.to(param.device, dtype=param.dtype).clone()
    _need(isinstance(loader_generator, torch.Generator), "RESTORE_GENERATOR")
    loader_generator.set_state(payload["loader_generator_state"])
    _need(callable(restore_augmentation), "AUGMENT_RESTORER_REQUIRED")
    restore_augmentation(payload["augmentation_rng"])
    _rng_restore(payload["rng_state"], rank)
    return {
        "next_epoch": payload["cursor"]["next_epoch"],
        "last_opt_step": payload["cursor"]["last_opt_step"],
        "accumulate": payload["cursor"]["accumulate"],
        "optimizer_updates": payload["cursor"]["optimizer_updates"],
        "ema_updates": payload["ema_updates"],
        "early_stopper_state": payload["early_stopper_state"],
    }
