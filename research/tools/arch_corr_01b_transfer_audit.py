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
EARLY_YAML = ROOT / "ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml"
G4_YAML = ROOT / "ultralytics/cfg/models/11/yolo11s-tpsc-g4-v1.yaml"
INIT_LOCK = ROOT / "research/01_provenance/INIT01_INITIALIZATION_LOCK.json"

EXPECTED_BASELINE_PARAMS = 9_431_275
EXPECTED_EARLY_PARAMS = 9_570_093
EXPECTED_G4_PARAMS = 10_021_679


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(path: Path) -> DetectionModel:
    torch.manual_seed(42)
    return DetectionModel(str(path), ch=3, nc=9, verbose=False)


def shared_state_contract(source: torch.nn.Module, target: torch.nn.Module) -> dict[str, Any]:
    src = source.state_dict()
    dst = target.state_dict()
    missing = [k for k, v in src.items() if k not in dst or dst[k].shape != v.shape]
    if missing:
        raise RuntimeError(f"Native shared-state contract failed: {missing[:20]}")
    extra = [k for k in dst if k not in src]
    non_adapter_extra = [k for k in extra if ".sc_adapters." not in k]
    if non_adapter_extra:
        raise RuntimeError(f"Unexpected non-adapter target state: {non_adapter_extra[:20]}")

    result = target.load_state_dict(src, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_after_load = list(result.missing_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected load keys: {unexpected}")
    if any(".sc_adapters." not in k for k in missing_after_load):
        raise RuntimeError(f"Non-adapter keys missing after load: {missing_after_load}")

    for key, value in src.items():
        if not torch.equal(target.state_dict()[key], value):
            raise RuntimeError(f"Transferred native tensor differs after load: {key}")

    return {
        "source_state_items": len(src),
        "target_state_items": len(dst),
        "native_items_matched": len(src),
        "extra_adapter_state_items": len(extra),
        "extra_adapter_state_keys": extra,
    }


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
    a = flatten_tensors(source(x))
    b = flatten_tensors(target(x))
    if len(a) != len(b):
        raise RuntimeError(f"Output structure differs: {len(a)} != {len(b)}")
    max_abs = 0.0
    exact = True
    for left, right in zip(a, b):
        if left.shape != right.shape:
            raise RuntimeError(f"Output shape differs: {left.shape} != {right.shape}")
        diff = (left - right).abs().max().item()
        max_abs = max(max_abs, diff)
        exact = exact and torch.equal(left, right)
    if max_abs > 1e-7:
        raise RuntimeError(f"Zero-gate identity contract failed: max_abs={max_abs}")
    return {"tensor_outputs": len(a), "bitwise_exact": exact, "max_abs_diff": max_abs}


def official_checkpoint_contract(
    checkpoint: Path,
    targets: dict[str, torch.nn.Module],
    init_lock: dict[str, Any],
) -> dict[str, Any]:
    expected_sha = init_lock["official_checkpoint"]["sha256"]
    observed_sha = sha256_file(checkpoint)
    if observed_sha != expected_sha:
        raise RuntimeError(f"Checkpoint SHA drift: {observed_sha} != {expected_sha}")

    ckpt, _ = torch_safe_load(str(checkpoint))
    source_model = ckpt.get("ema") or ckpt["model"]
    source_state = source_model.float().state_dict()
    expected_transfer_items = init_lock["nine_class_initialization_contract"]["transferable_state_items"]
    baseline_nontransfer = set(
        init_lock["nine_class_initialization_contract"]["nontransferable_target_keys"]
    )

    report: dict[str, Any] = {}
    for name, target in targets.items():
        target_state = target.state_dict()
        transferable = {
            k: v for k, v in source_state.items()
            if k in target_state and target_state[k].shape == v.shape
        }
        if len(transferable) != expected_transfer_items:
            raise RuntimeError(
                f"{name}: transferable source items {len(transferable)} != {expected_transfer_items}"
            )

        nontransferred_target = set(target_state) - set(transferable)
        unexpected_native = {
            k for k in nontransferred_target
            if ".sc_adapters." not in k and k not in baseline_nontransfer
        }
        if unexpected_native:
            raise RuntimeError(f"{name}: unexpected native nontransfer keys: {sorted(unexpected_native)}")

        result = target.load_state_dict(transferable, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"{name}: unexpected checkpoint load keys: {result.unexpected_keys}")

        for key, value in transferable.items():
            if not torch.equal(target.state_dict()[key], value):
                raise RuntimeError(f"{name}: official checkpoint tensor not loaded exactly: {key}")

        report[name] = {
            "transferable_source_items": len(transferable),
            "baseline_native_nontransfer_keys": sorted(baseline_nontransfer),
            "new_adapter_nontransfer_items": sum(
                1 for k in nontransferred_target if ".sc_adapters." in k
            ),
            "unexpected_native_nontransfer_items": 0,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("ARCH_CORR_01B_AUDIT.json"))
    args = parser.parse_args()

    baseline = build(BASELINE_YAML)
    early = build(EARLY_YAML)
    g4 = build(G4_YAML)

    counts = {
        "baseline": parameter_count(baseline),
        "tpsc_early": parameter_count(early),
        "tpsc_g4": parameter_count(g4),
    }
    expected = {
        "baseline": EXPECTED_BASELINE_PARAMS,
        "tpsc_early": EXPECTED_EARLY_PARAMS,
        "tpsc_g4": EXPECTED_G4_PARAMS,
    }
    if counts != expected:
        raise RuntimeError(f"Parameter-count contract failed: {counts} != {expected}")

    early_shared = shared_state_contract(baseline, early)
    early_identity = identity_contract(baseline, early)

    # Reload baseline because identity_contract consumes no state but keeping each audit independent is clearer.
    baseline_g4 = build(BASELINE_YAML)
    g4_shared = shared_state_contract(baseline_g4, g4)
    g4_identity = identity_contract(baseline_g4, g4)

    init_lock = json.loads(INIT_LOCK.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": "ARCH-CORR-01B-audit-v1.0",
        "training_started": False,
        "test_access": "NONE",
        "parameter_counts": counts,
        "added_parameters": {
            "tpsc_early": counts["tpsc_early"] - counts["baseline"],
            "tpsc_g4": counts["tpsc_g4"] - counts["baseline"],
        },
        "shared_state_contract": {
            "tpsc_early": early_shared,
            "tpsc_g4": g4_shared,
        },
        "zero_gate_identity": {
            "tpsc_early": early_identity,
            "tpsc_g4": g4_identity,
        },
        "official_checkpoint": None,
    }

    if args.checkpoint is not None:
        report["official_checkpoint"] = official_checkpoint_contract(
            args.checkpoint.resolve(),
            {"tpsc_early": build(EARLY_YAML), "tpsc_g4": build(G4_YAML)},
            init_lock,
        )

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={args.output.resolve()}")
    print("TRAINING_STARTED=FALSE")
    print("TEST_ACCESS=NONE")
    print("ARCH_CORR_01B_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
