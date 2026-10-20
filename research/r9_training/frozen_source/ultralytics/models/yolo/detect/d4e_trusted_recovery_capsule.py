"""D4-E Step14.4B experimental CPU-tested recovery boundary.

Does NOT alter or authorize the frozen VGRA trainer. Epoch-boundary snapshots only.
The independent receipt SHA256 MUST be pinned outside the writable run directory.
Full-precision model, EMA, optimizer, scaler, scheduler, parameter gradients,
training cursor, sampler epoch and RNG state are preserved. A separate governed
integration and real GPU replay are required before production use.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import secrets
from pathlib import Path
from typing import Any

import numpy as np
import torch
from .d4e_transform_rng_state import SCHEMA as AUGMENT_SCHEMA, restore_augmentation_rng

SCHEMA = "D4E_TRUSTED_EPOCH_BOUNDARY_RECOVERY_R2_AUGMENT"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha(s: str) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _checked_file(path: str | Path, root: str | Path, *, must_exist: bool) -> Path:
    root = Path(root).resolve(strict=True)
    p = Path(path).absolute()
    try:
        rel = p.relative_to(root)
    except ValueError as exc:
        raise ValueError("RECOVERY_PATH_OUTSIDE_APPROVED_ROOT") from exc
    if not rel.parts or any(part in ("..", ".") for part in rel.parts):
        raise ValueError("RECOVERY_INVALID_RELATIVE_PATH")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("RECOVERY_SYMLINK_FORBIDDEN")
    if must_exist and (not p.is_file() or p.is_symlink()):
        raise FileNotFoundError("RECOVERY_TRUSTED_FILE_MISSING")
    return p


def _identity_check(identity: dict[str, str]) -> None:
    required = {"arm", "source_commit", "pretrained_sha256", "recipe_sha256",
                "data_yaml_sha256", "assignment_sha256", "overlays_sha256", "run_id"}
    if set(identity) != required:
        raise ValueError("RECOVERY_IDENTITY_KEYS_MISMATCH")
    if identity["arm"] not in {"MV00", "MV01"}:
        raise ValueError("RECOVERY_ARM_MISMATCH")
    oid = identity["source_commit"]
    if not isinstance(oid, str) or len(oid) != 40 or any(c not in "0123456789abcdef" for c in oid):
        raise ValueError("RECOVERY_SOURCE_GIT_OID_INVALID")
    for key in required - {"arm", "run_id", "source_commit"}:
        if not _valid_sha(identity[key]):
            raise ValueError(f"RECOVERY_IDENTITY_SHA_INVALID_{key}")
    if not identity["run_id"] or len(identity["run_id"]) > 128 or not all(
        c.isalnum() or c in "-_" for c in identity["run_id"]
    ):
        raise ValueError("RECOVERY_RUN_ID_INVALID")


def _rng_capture() -> dict[str, Any]:
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": n[0], "keys": torch.from_numpy(n[1].copy()),
                  "pos": int(n[2]), "has_gauss": int(n[3]), "cached_gaussian": float(n[4])},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _rng_restore(data: dict[str, Any]) -> None:
    random.setstate(data["python"])
    n = data["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].numpy().astype(np.uint32),
                         int(n["pos"]), int(n["has_gauss"]), float(n["cached_gaussian"])))
    torch.set_rng_state(data["torch_cpu"].cpu())
    if data["torch_cuda"]:
        if not torch.cuda.is_available() or len(data["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("RECOVERY_CUDA_TOPOLOGY_MISMATCH")
        torch.cuda.set_rng_state_all(data["torch_cuda"])


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True).contiguous()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _finite(obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        if (obj.is_floating_point() or obj.is_complex()) and not bool(torch.isfinite(obj).all().item()):
            raise ValueError("RECOVERY_NONFINITE_TENSOR")
    elif isinstance(obj, (dict, list, tuple)):
        items = obj.values() if isinstance(obj, dict) else obj
        for v in items:
            _finite(v)
    elif isinstance(obj, float) and not np.isfinite(obj):
        raise ValueError("RECOVERY_NONFINITE_NUMBER")


def _cursor_check(cursor: dict[str, Any]) -> None:
    keys = {"completed_epoch", "next_epoch", "nb", "last_opt_step", "optimizer_updates",
            "accumulate", "sampler_next_epoch"}
    if set(cursor) != keys or any(type(v) is not int for v in cursor.values()):
        raise ValueError("RECOVERY_CURSOR_SCHEMA_INVALID")
    e = cursor["completed_epoch"]
    if e < 0 or cursor["next_epoch"] != e + 1 or cursor["sampler_next_epoch"] != e + 1:
        raise ValueError("RECOVERY_EPOCH_CURSOR_INVALID")
    if cursor["nb"] <= 0 or cursor["accumulate"] <= 0 or cursor["optimizer_updates"] < 0:
        raise ValueError("RECOVERY_BATCH_CURSOR_INVALID")
    if not (-1 <= cursor["last_opt_step"] <= (e + 1) * cursor["nb"] - 1):
        raise ValueError("RECOVERY_LAST_OPT_STEP_INVALID")


def snapshot_epoch_boundary(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ema_model: torch.nn.Module, ema_updates: int, scheduler: Any, scaler: Any,
    identity: dict[str, str], cursor: dict[str, int], loader_generator: torch.Generator | None = None,
    early_stopper_state: dict[str, Any] | None = None,
    augmentation_rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture after epoch metrics/validation and before the next epoch begins."""
    _identity_check(identity)
    _cursor_check(cursor)
    if not isinstance(augmentation_rng_state,dict) or augmentation_rng_state.get("schema")!=AUGMENT_SCHEMA:
        raise ValueError("R2_AUGMENT_RNG_REQUIRED_FOR_NEW_CHECKPOINT")
    grads = {name: _to_cpu(p.grad) if p.grad is not None else None
             for name, p in model.named_parameters()}
    payload = {
        "schema": SCHEMA, "identity": dict(identity), "cursor": dict(cursor),
        "model_state": _to_cpu(model.state_dict()), "parameter_grads": grads,
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "ema_state": _to_cpu(ema_model.state_dict()), "ema_updates": int(ema_updates),
        "scheduler_state": _to_cpu(scheduler.state_dict()),
        "scaler_state": _to_cpu(scaler.state_dict()) if scaler is not None else {},
        "early_stopper_state": _to_cpu(early_stopper_state or {}),
        "augmentation_rng_state": augmentation_rng_state,
        "rng_state": _rng_capture(),
        "loader_generator_state": loader_generator.get_state().clone() if loader_generator is not None else None,
    }
    _finite(payload["model_state"])
    for k in ("parameter_grads", "optimizer_state", "ema_state", "scheduler_state", "scaler_state"):
        _finite(payload[k])
    return payload


def save_capsule(
    *, payload: dict[str, Any], checkpoint_path: str | Path, root: str | Path,
) -> dict[str, str]:
    """Write immutable checkpoint and receipt; pin receipt SHA *outside* run directory."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("RECOVERY_CAPSULE_SCHEMA_INVALID")
    _identity_check(payload["identity"])
    _cursor_check(payload["cursor"])
    p = _checked_file(checkpoint_path, root, must_exist=False)
    if p.suffix != ".pt" or p.exists():
        raise FileExistsError("RECOVERY_CAPSULE_MUST_BE_NEW_PT")
    receipt = p.with_suffix(".receipt.json")
    receipt = _checked_file(receipt, root, must_exist=False)
    if receipt.exists():
        raise FileExistsError("RECOVERY_RECEIPT_MUST_BE_NEW")
    p.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    checkpoint_bytes = buffer.getvalue()
    meta = {"schema": SCHEMA, "checkpoint_filename": p.name,
            "checkpoint_sha256": _digest(checkpoint_bytes),
            "identity": payload["identity"], "cursor": payload["cursor"]}
    receipt_bytes = (json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    tmp1 = p.parent / ("." + p.name + "." + secrets.token_hex(8) + ".tmp")
    tmp2 = receipt.parent / ("." + receipt.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        for tmp, contents in [(tmp1, checkpoint_bytes), (tmp2, receipt_bytes)]:
            with tmp.open("xb") as f:
                f.write(contents)
                f.flush()
                os.fsync(f.fileno())
        if p.exists() or receipt.exists():
            raise FileExistsError("RECOVERY_DESTINATION_RACE")
        # link() is fail-if-exists even under concurrent writers (replace() is not).
        os.link(tmp1, p)
        os.link(tmp2, receipt)
    finally:
        tmp1.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)
    return {"checkpoint": str(p), "checkpoint_sha256": _digest(checkpoint_bytes),
            "receipt": str(receipt), "trusted_receipt_sha256": _digest(receipt_bytes)}


def read_authenticated_capsule(
    *, checkpoint_path: str | Path, root: str | Path,
    trusted_receipt_sha256: str, identity: dict[str, str],
) -> dict[str, Any]:
    """Fail before *any* Torch deserialization on mismatch or untrusted SHA."""
    _identity_check(identity)
    if not _valid_sha(trusted_receipt_sha256):
        raise ValueError("RECOVERY_EXTERNAL_RECEIPT_PIN_REQUIRED")
    p = _checked_file(checkpoint_path, root, must_exist=True)
    receipt = _checked_file(p.with_suffix(".receipt.json"), root, must_exist=True)
    raw_receipt = receipt.read_bytes()
    if _digest(raw_receipt) != trusted_receipt_sha256:
        raise ValueError("RECOVERY_RECEIPT_AUTHENTICATION_FAILED")
    meta = json.loads(raw_receipt)
    if meta.get("schema") != SCHEMA or meta.get("checkpoint_filename") != p.name:
        raise ValueError("RECOVERY_RECEIPT_INVALID")
    if meta.get("identity") != identity:
        raise ValueError("RECOVERY_IDENTITY_MISMATCH")
    _cursor_check(meta.get("cursor", {}))
    raw_ckpt = p.read_bytes()
    if _digest(raw_ckpt) != meta.get("checkpoint_sha256"):
        raise ValueError("RECOVERY_CHECKPOINT_HASH_MISMATCH")
    # The only deserialization route is weights_only=True, AFTER external provenance checks.
    # D4F R5: after receipt and checkpoint SHA authentication, allow only the
    # five established NumPy value types for restricted PyTorch deserialization.
    # These permissions are scoped to this call and do not weaken provenance.
    _numpy_safe_types = [
        np._core.multiarray.scalar, np.dtype,
        type(np.dtype('float64')), type(np.dtype('float32')), type(np.dtype('int64')),
    ]
    with torch.serialization.safe_globals(_numpy_safe_types):
        payload = torch.load(io.BytesIO(raw_ckpt), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("RECOVERY_DESERIALIZED_SCHEMA_INVALID")
    if payload.get("identity") != identity or payload.get("cursor") != meta["cursor"]:
        raise ValueError("RECOVERY_PAYLOAD_BINDING_MISMATCH")
    _finite(payload["model_state"])
    for k in ("parameter_grads", "optimizer_state", "ema_state", "scheduler_state", "scaler_state"):
        _finite(payload[k])
    if not isinstance(payload.get("augmentation_rng_state"),dict) or payload["augmentation_rng_state"].get("schema")!=AUGMENT_SCHEMA:
        raise ValueError("R2_AUGMENT_RNG_MISSING_IN_AUTHENTICATED_CAPSULE")
    return payload


def restore_capsule(
    *, payload: dict[str, Any], model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, ema_model: torch.nn.Module,
    scheduler: Any, scaler: Any, loader_generator: torch.Generator | None = None,
    augmentation_transform_tree: Any = None,
) -> dict[str, Any]:
    """Call only AFTER fresh governed source setup; BEFORE iteration/prefetch, restore RNG last."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("RECOVERY_PAYLOAD_UNAPPROVED_SCHEMA")
    _cursor_check(payload["cursor"])
    if augmentation_transform_tree is None or payload.get("augmentation_rng_state",{}).get("schema")!=AUGMENT_SCHEMA:
        raise ValueError("R2_MISSING_VERIFIED_AUGMENT_RNG_FOR_RESTORE")
    current = model.state_dict()
    archived = payload["model_state"]
    if current.keys() != archived.keys() or any(
        current[k].shape != archived[k].shape or current[k].dtype != archived[k].dtype
        for k in current
    ):
        raise ValueError("RECOVERY_MODEL_STATE_SHAPE_OR_DTYPE_MISMATCH")
    named_params = dict(model.named_parameters())
    if set(payload["parameter_grads"]) != named_params.keys():
        raise ValueError("RECOVERY_MODEL_PARAMETER_NAMES_MISMATCH")
    for name, p in named_params.items():
        g = payload["parameter_grads"][name]
        if g is not None and (tuple(g.shape) != tuple(p.shape) or g.dtype != p.dtype):
            raise ValueError("RECOVERY_GRADIENT_SHAPE_OR_DTYPE_MISMATCH")
    current_ema = ema_model.state_dict()
    if current_ema.keys() != payload["ema_state"].keys() or any(
        current_ema[k].shape != payload["ema_state"][k].shape
        for k in current_ema
    ):
        raise ValueError("RECOVERY_EMA_STATE_SHAPE_MISMATCH")
    active_groups = optimizer.state_dict()["param_groups"]
    saved_groups = payload["optimizer_state"]["param_groups"]
    if len(active_groups) != len(saved_groups) or any(
        len(a["params"]) != len(b["params"]) for a, b in zip(active_groups, saved_groups)
    ):
        raise ValueError("RECOVERY_OPTIMIZER_PARAMETER_GROUP_MISMATCH")
    # Check ahead of the first state mutation, including NaN gradients.
    for key in ("model_state", "parameter_grads", "ema_state", "optimizer_state"):
        _finite(payload[key])
    model.load_state_dict(archived, strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    ema_model.load_state_dict(payload["ema_state"], strict=True)
    scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state"])
    for name, p in model.named_parameters():
        g = payload["parameter_grads"][name]
        if g is None:
            p.grad = None
        elif tuple(g.shape) != tuple(p.shape) or g.dtype != p.dtype:
            raise ValueError("RECOVERY_GRADIENT_SHAPE_OR_DTYPE_MISMATCH")
        else:
            p.grad = g.to(p.device, copy=True)
    if loader_generator is not None:
        if payload["loader_generator_state"] is None:
            raise ValueError("RECOVERY_LOADER_GENERATOR_STATE_MISSING")
        loader_generator.set_state(payload["loader_generator_state"])
    _rng_restore(payload["rng_state"])
    restore_augmentation_rng(augmentation_transform_tree,payload["augmentation_rng_state"])
    return {"cursor": dict(payload["cursor"]), "ema_updates": payload["ema_updates"],
            "early_stopper_state": dict(payload["early_stopper_state"])}
