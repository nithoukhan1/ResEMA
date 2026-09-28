#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel, torch_safe_load


BASELINE_YAML = ROOT / "ultralytics/cfg/models/11/yolo11s.yaml"
TPEMA_YAML = ROOT / "ultralytics/cfg/models/11/yolo11s-tpema-head-v1.yaml"
INIT_LOCK = ROOT / "research/01_provenance/INIT01_INITIALIZATION_LOCK.json"

EXPECTED_BASELINE_PARAMS = 9_431_275
EXPECTED_TPEMA_PARAMS = 9_435_423


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(path: Path) -> DetectionModel:
    torch.manual_seed(42)
    return DetectionModel(str(path), ch=3, nc=9, verbose=False)


def flatten_tensors(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        out: list[torch.Tensor] = []
        for key in sorted(obj):
            out.extend(flatten_tensors(obj[key]))
        return out
    if isinstance(obj, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in obj:
            out.extend(flatten_tensors(item))
        return out
    return []


@torch.no_grad()
def identity_contract(source: torch.nn.Module, target: torch.nn.Module) -> dict[str, Any]:
    source.eval()
    target.eval()
    x = torch.randn(1, 3, 256, 256)
    left = flatten_tensors(source(x))
    right = flatten_tensors(target(x))
    if len(left) != len(right):
        raise RuntimeError(f"Output structure differs: {len(left)} != {len(right)}")
    max_abs = 0.0
    exact = True
    for a, b in zip(left, right):
        if a.shape != b.shape:
            raise RuntimeError(f"Output shape differs: {a.shape} != {b.shape}")
        max_abs = max(max_abs, (a - b).abs().max().item())
        exact = exact and torch.equal(a, b)
    if max_abs > 1e-7:
        raise RuntimeError(f"Zero-gate identity failed: max_abs={max_abs}")
    return {"tensor_outputs": len(left), "bitwise_exact": exact, "max_abs_diff": max_abs}


def shared_contract(source: torch.nn.Module, target: torch.nn.Module) -> dict[str, Any]:
    src = source.state_dict()
    dst = target.state_dict()
    missing_native = [k for k, v in src.items() if k not in dst or dst[k].shape != v.shape]
    if missing_native:
        raise RuntimeError(f"Missing native keys: {missing_native[:20]}")
    extra = [k for k in dst if k not in src]
    unexpected = [k for k in extra if ".ema_adapter." not in k]
    if unexpected:
        raise RuntimeError(f"Unexpected non-EMA additions: {unexpected[:20]}")
    result = target.load_state_dict(src, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected load keys: {result.unexpected_keys}")
    if any(".ema_adapter." not in key for key in result.missing_keys):
        raise RuntimeError(f"Non-EMA keys missing after native load: {result.missing_keys}")
    for key, value in src.items():
        if not torch.equal(target.state_dict()[key], value):
            raise RuntimeError(f"Transferred native tensor differs: {key}")
    return {
        "source_state_items": len(src),
        "target_state_items": len(dst),
        "native_items_matched": len(src),
        "new_ema_state_items": len(extra),
        "new_ema_state_keys": extra,
    }


def official_contract(checkpoint: Path, target: torch.nn.Module, init_lock: dict[str, Any]) -> dict[str, Any]:
    observed = sha256_file(checkpoint)
    expected = init_lock["official_checkpoint"]["sha256"]
    if observed != expected:
        raise RuntimeError(f"Checkpoint SHA drift: {observed} != {expected}")

    ckpt, _ = torch_safe_load(str(checkpoint))
    source = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    target_state = target.state_dict()
    transferable = {
        k: v for k, v in source.items()
        if k in target_state and target_state[k].shape == v.shape
    }
    expected_count = init_lock["nine_class_initialization_contract"]["transferable_state_items"]
    if len(transferable) != expected_count:
        raise RuntimeError(f"Transfer count drift: {len(transferable)} != {expected_count}")

    baseline_nontransfer = set(
        init_lock["nine_class_initialization_contract"]["nontransferable_target_keys"]
    )
    nontransferred = set(target_state) - set(transferable)
    unexpected_native = {
        k for k in nontransferred
        if ".ema_adapter." not in k and k not in baseline_nontransfer
    }
    if unexpected_native:
        raise RuntimeError(f"Unexpected native nontransfer keys: {sorted(unexpected_native)}")

    result = target.load_state_dict(transferable, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected official load keys: {result.unexpected_keys}")
    for key, value in transferable.items():
        if not torch.equal(target.state_dict()[key], value):
            raise RuntimeError(f"Official tensor not loaded exactly: {key}")

    return {
        "transferable_source_items": len(transferable),
        "baseline_native_nontransfer_keys": sorted(baseline_nontransfer),
        "new_ema_nontransfer_items": sum(1 for k in nontransferred if ".ema_adapter." in k),
        "unexpected_native_nontransfer_items": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("ARCH_CORR_01C_EMA_AUDIT.json"))
    args = parser.parse_args()

    baseline = build(BASELINE_YAML)
    target = build(TPEMA_YAML)
    counts = {
        "baseline": sum(p.numel() for p in baseline.parameters()),
        "tpema_head": sum(p.numel() for p in target.parameters()),
    }
    expected_counts = {
        "baseline": EXPECTED_BASELINE_PARAMS,
        "tpema_head": EXPECTED_TPEMA_PARAMS,
    }
    if counts != expected_counts:
        raise RuntimeError(f"Parameter-count contract failed: {counts} != {expected_counts}")

    shared = shared_contract(baseline, target)
    identity = identity_contract(baseline, target)
    init_lock = json.loads(INIT_LOCK.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "schema_version": "ARCH-CORR-01C-ema-audit-v1.0",
        "training_started": False,
        "test_access": "NONE",
        "parameter_counts": counts,
        "added_parameters": counts["tpema_head"] - counts["baseline"],
        "shared_state_contract": shared,
        "zero_gate_identity": identity,
        "official_checkpoint": None,
    }
    if args.checkpoint is not None:
        report["official_checkpoint"] = official_contract(
            args.checkpoint.resolve(),
            build(TPEMA_YAML),
            init_lock,
        )

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={args.output.resolve()}")
    print("TRAINING_STARTED=FALSE")
    print("TEST_ACCESS=NONE")
    print("ARCH_CORR_01C_EMA_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
