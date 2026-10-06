#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_CSV = ROOT / "research/01_provenance/ARTIFACTS.csv"
DATA_BINDINGS = ROOT / "research/04_data/manifests/DATA01_ACTIVE_DATASET_BINDINGS.json"
VAL_METADATA = ROOT / "research/04_data/split_b/split_B_val.csv"
BASELINE_RUNTIME = ROOT / "research/05_experiments/BASELINE_FREEZE_01_PERCLASS_INPUTS.json"
SMS_REGISTRY = ROOT / "research/05_experiments/SINGLE_MODULE_SCREEN_01_EXPERIMENTS.csv"
COMB_REGISTRY = ROOT / "research/05_experiments/COMBINATION_SCREEN_01_EXPERIMENTS.csv"

EXPERIMENT_IDS = (
    "BASE-B-ORG-PT-S42",
    "BORG-PT-S42-SCCONV-EARLY-E100",
    "BORG-PT-S42-SCCONV-4STAGE-E100",
    "BORG-PT-S42-DYSAMPLE-E100",
    "BORG-PT-S42-CANONICAL-EMA-E100",
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100",
)

DISPLAY_NAMES = {
    "BASE-B-ORG-PT-S42": "YOLO11s baseline",
    "BORG-PT-S42-SCCONV-EARLY-E100": "SCConv-Early",
    "BORG-PT-S42-SCCONV-4STAGE-E100": "SCConv-4Stage",
    "BORG-PT-S42-DYSAMPLE-E100": "DySample",
    "BORG-PT-S42-CANONICAL-EMA-E100": "Canonical EMA",
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100": "SCConv-Early + Canonical EMA",
}

EXPECTED_PARAMS = {
    "BASE-B-ORG-PT-S42": 9431275,
    "BORG-PT-S42-SCCONV-EARLY-E100": 9570093,
    "BORG-PT-S42-SCCONV-4STAGE-E100": 10021679,
    "BORG-PT-S42-DYSAMPLE-E100": 9455915,
    "BORG-PT-S42-CANONICAL-EMA-E100": 9435423,
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100": 9574241,
}

EXPECTED_CUSTOM_COUNTS = {
    "BASE-B-ORG-PT-S42": {},
    "BORG-PT-S42-SCCONV-EARLY-E100": {
        "C3k2_TPSC": 2,
    },
    "BORG-PT-S42-SCCONV-4STAGE-E100": {
        "C3k2_TPSCG4": 4,
    },
    "BORG-PT-S42-DYSAMPLE-E100": {
        "DySample": 2,
    },
    "BORG-PT-S42-CANONICAL-EMA-E100": {
        "C3k2_TPEMA": 4,
    },
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100": {
        "C3k2_TPSC": 2,
        "C3k2_TPEMA": 4,
    },
}

CUSTOM_CLASS_NAMES = (
    "C3k2_TPSC",
    "C3k2_TPSCG4",
    "C3k2_TPEMA",
    "DySample",
)

EXPECTED_CHECKPOINTS = {
    "BASE-B-ORG-PT-S42":
        "65e3c59901e0429f70a7b368dc29ebb698c427a799cfcf66354a882ec48bc4c7",
    "BORG-PT-S42-SCCONV-EARLY-E100":
        "620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1",
    "BORG-PT-S42-SCCONV-4STAGE-E100":
        "5485ec2fd9cc6421bdff7b4f0cd8b724db3ed8b540d4d11c77519cdbbcbfb7c4",
    "BORG-PT-S42-DYSAMPLE-E100":
        "61f3a99e4d29f20fddddd103ef3af049861d0b2278f384a044897be93a5fd556",
    "BORG-PT-S42-CANONICAL-EMA-E100":
        "f9d262c22c2bcb075cafe8eac368c0d4083a99f765f64e8ed6781c337ff0b6ec",
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100":
        "531c9927592781e7e19b6af1a35ce910113758d7ad5842677a5b7065487c897d",
}

CLASS_NAMES = (
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "text",
)

EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.10.0+cu128",
    "ultralytics": "8.4.7",
    "gpu_inventory": ["Tesla T4", "Tesla T4"],
    "evaluation_device": 0,
    "imgsz": 1024,
    "batch": 16,
    "workers": 4,
    "rect": False,
    "conf": None,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "plots": True,
    "save_json": False,
    "deterministic": True,
    "seed": 42,
    "split": "val",
}

EXPECTED_B_ORG = {
    "binding_id": "DATA01:B-ORG:v1",
    "train_count": 14227,
    "train_hash": "4ab032d669ee40daeaa6d37680def3ba4aad4dc64b9d0f16b7b658be9e48812b",
    "val_count": 3050,
    "val_hash": "f36040d4a798cbda113909907ba21e3dac1bb91b268124ec51c4fb2899e4a816",
    "val_patients": 914,
    "operational_val": 3049,
    "known_unreadable": "1502_0635264266_05_WRI-R2_M015.png",
}


class D1PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D1PreflightError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise D1PreflightError(
            "Git command failed: "
            + "git "
            + " ".join(args)
            + "\n"
            + result.stderr
        )
    return result.stdout.strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def discover_checkpoint(input_root: Path, expected_sha: str) -> Path:
    matches: list[Path] = []

    for path in input_root.rglob("best.pt"):
        if not path.is_file():
            continue
        if sha256_file(path) == expected_sha:
            matches.append(path.resolve())

    matches = sorted(set(matches))

    require(
        len(matches) == 1,
        f"Expected exactly one best.pt with SHA {expected_sha}; observed {matches}",
    )

    return matches[0]


def verify_artifact_registry() -> dict[str, dict[str, str]]:
    _fields, rows = load_csv(ARTIFACTS_CSV)

    selected: dict[str, dict[str, str]] = {}

    for experiment_id in EXPERIMENT_IDS:
        matches = [
            row
            for row in rows
            if row["experiment_id"] == experiment_id
            and row["artifact_role"] == "selected_checkpoint"
            and row["logical_name"] == "best.pt"
            and row["status"] == "VERIFIED"
        ]

        require(
            len(matches) == 1,
            f"Selected checkpoint must resolve exactly once: {experiment_id}",
        )

        row = matches[0]

        require(
            row["sha256"] == EXPECTED_CHECKPOINTS[experiment_id],
            f"Checkpoint registry SHA drift: {experiment_id}",
        )

        selected[experiment_id] = row

    return selected


def verify_historical_experiment_contracts() -> dict[str, Any]:
    _fields, sms_rows = load_csv(SMS_REGISTRY)
    _fields, comb_rows = load_csv(COMB_REGISTRY)

    sms_by_id = {row["experiment_id"]: row for row in sms_rows}
    comb_by_id = {row["experiment_id"]: row for row in comb_rows}

    for experiment_id in (
        "BORG-PT-S42-SCCONV-EARLY-E100",
        "BORG-PT-S42-SCCONV-4STAGE-E100",
        "BORG-PT-S42-DYSAMPLE-E100",
        "BORG-PT-S42-CANONICAL-EMA-E100",
    ):
        require(experiment_id in sms_by_id, f"SMS row missing: {experiment_id}")
        row = sms_by_id[experiment_id]
        require(row["status"] == "COMPLETE", f"SMS not complete: {experiment_id}")
        require(row["data_binding"] == "DATA01:B-ORG:v1", f"SMS data drift: {experiment_id}")
        require(row["initialization"] == "pretrained", f"SMS init drift: {experiment_id}")
        require(row["seed"] == "42", f"SMS seed drift: {experiment_id}")
        require(row["test_access"] == "NONE", f"SMS test-access drift: {experiment_id}")

    comb_id = "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100"
    require(comb_id in comb_by_id, "COMB row missing")
    comb = comb_by_id[comb_id]
    require(comb["status"] == "COMPLETE", "COMB is not complete")
    require(comb["data_binding"] == "DATA01:B-ORG:v1", "COMB data-binding drift")
    require(comb["initialization"] == "pretrained", "COMB init drift")
    require(comb["seed"] == "42", "COMB seed drift")
    require(comb["test_access"] == "NONE", "COMB test-access drift")

    baseline_registry = load_json(BASELINE_RUNTIME)
    baseline_rows = [
        x
        for x in baseline_registry["experiments"]
        if x["experiment_id"] == "BASE-B-ORG-PT-S42"
    ]
    require(len(baseline_rows) == 1, "Baseline row must resolve exactly once")
    baseline = baseline_rows[0]
    require(baseline["data_binding"] == "DATA01:B-ORG:v1", "Baseline data drift")
    require(baseline["initialization"] == "pretrained", "Baseline init drift")

    return {
        "baseline": baseline,
        "sms": {k: sms_by_id[k] for k in sms_by_id if k in EXPERIMENT_IDS},
        "combination": comb,
    }


def verify_runtime_contract() -> dict[str, Any]:
    baseline_inputs = load_json(BASELINE_RUNTIME)
    governed = baseline_inputs["runtime_contract"]

    require(governed == EXPECTED_RUNTIME, "Frozen baseline runtime contract drift")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        "/kaggle/working/.ultralytics_config",
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import torch
    import ultralytics

    require(platform.python_version() == EXPECTED_RUNTIME["python"], "Python version drift")
    require(torch.__version__ == EXPECTED_RUNTIME["torch"], "Torch version drift")
    require(ultralytics.__version__ == EXPECTED_RUNTIME["ultralytics"], "Ultralytics version drift")

    source = Path(ultralytics.__file__).resolve()
    require(ROOT.resolve() in source.parents, f"Ultralytics not imported from current checkout: {source}")

    gpu_names = [
        torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())
    ]
    require(gpu_names == EXPECTED_RUNTIME["gpu_inventory"], f"GPU inventory drift: {gpu_names}")

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_source": str(source),
        "gpu_inventory": gpu_names,
    }


def verify_dataset_binding(input_root: Path, output_dir: Path) -> dict[str, Any]:
    from research.runtime import baseline_runner as br

    manifest = load_json(DATA_BINDINGS)
    binding = manifest["bindings"]["B_ORG"]

    require(binding["binding_id"] == EXPECTED_B_ORG["binding_id"], "B-ORG binding ID drift")
    require(binding["runtime_counts"]["train_images"] == EXPECTED_B_ORG["train_count"], "B-ORG train count drift")
    require(binding["runtime_counts"]["validation_images"] == EXPECTED_B_ORG["val_count"], "B-ORG val count drift")
    require(
        binding["membership_sha256_final_newline"]["train"] == EXPECTED_B_ORG["train_hash"],
        "B-ORG train membership hash drift",
    )
    require(
        binding["membership_sha256_final_newline"]["validation"] == EXPECTED_B_ORG["val_hash"],
        "B-ORG validation membership hash drift",
    )
    require(binding["patients"]["validation"] == EXPECTED_B_ORG["val_patients"], "B-ORG validation patient count drift")
    require(
        binding["inherited_operational_validation_note"]["operational_readable_images"]
        == EXPECTED_B_ORG["operational_val"],
        "B-ORG operational validation count drift",
    )

    train_images = br.discover_membership_directory(
        input_root,
        kind="images",
        expected_count=EXPECTED_B_ORG["train_count"],
        expected_hash=EXPECTED_B_ORG["train_hash"],
    )
    train_labels = br.discover_membership_directory(
        input_root,
        kind="labels",
        expected_count=EXPECTED_B_ORG["train_count"],
        expected_hash=EXPECTED_B_ORG["train_hash"],
    )
    val_images = br.discover_membership_directory(
        input_root,
        kind="images",
        expected_count=EXPECTED_B_ORG["val_count"],
        expected_hash=EXPECTED_B_ORG["val_hash"],
    )
    val_labels = br.discover_membership_directory(
        input_root,
        kind="labels",
        expected_count=EXPECTED_B_ORG["val_count"],
        expected_hash=EXPECTED_B_ORG["val_hash"],
    )

    br.verify_image_label_pair(train_images, train_labels)
    br.verify_image_label_pair(val_images, val_labels)

    runtime_yaml = output_dir / "runtime_data_train_val_only.yaml"
    runtime_yaml_sha = br.write_runtime_data_yaml(
        runtime_yaml,
        train_images=train_images,
        validation_images=val_images,
    )

    payload = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    require("test" not in payload, "D1 runtime YAML contains forbidden test key")
    require(payload["nc"] == 9, "Runtime YAML nc drift")
    require(tuple(payload["names"].values()) == CLASS_NAMES, "Runtime YAML class-name drift")

    _fields, val_rows = load_csv(VAL_METADATA)
    require(len(val_rows) == EXPECTED_B_ORG["val_count"], "Validation metadata row-count drift")
    patient_ids = {row["patient_id"] for row in val_rows}
    require(len(patient_ids) == EXPECTED_B_ORG["val_patients"], "Validation metadata patient-count drift")

    unreadable_stem = Path(EXPECTED_B_ORG["known_unreadable"]).stem
    require(
        sum(row["filestem"] == unreadable_stem for row in val_rows) == 1,
        "Known unreadable validation filestem not uniquely represented in metadata",
    )

    return {
        "binding_id": binding["binding_id"],
        "train_images": str(train_images),
        "train_labels": str(train_labels),
        "validation_images": str(val_images),
        "validation_labels": str(val_labels),
        "frozen_validation_images": EXPECTED_B_ORG["val_count"],
        "operational_validation_images": EXPECTED_B_ORG["operational_val"],
        "validation_patients": EXPECTED_B_ORG["val_patients"],
        "validation_membership_sha256": EXPECTED_B_ORG["val_hash"],
        "known_unreadable_validation_image": EXPECTED_B_ORG["known_unreadable"],
        "runtime_yaml": str(runtime_yaml),
        "runtime_yaml_sha256": runtime_yaml_sha,
        "runtime_yaml_contains_test": False,
        "test_directory_scan_pruned": True,
    }


def inspect_model(experiment_id: str, checkpoint: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    yolo = YOLO(str(checkpoint))
    model = yolo.model

    require(getattr(model, "nc", None) == 9, f"nc drift: {experiment_id}")

    params = sum(p.numel() for p in model.parameters())
    require(params == EXPECTED_PARAMS[experiment_id], f"parameter-count drift: {experiment_id}")

    names = yolo.names
    if isinstance(names, dict):
        ordered_names = tuple(names[i] for i in sorted(names))
    else:
        ordered_names = tuple(names)

    require(ordered_names == CLASS_NAMES, f"class-name drift: {experiment_id}")

    observed_counts = {
        name: sum(1 for module in model.modules() if module.__class__.__name__ == name)
        for name in CUSTOM_CLASS_NAMES
    }
    observed_nonzero = {
        name: count
        for name, count in observed_counts.items()
        if count
    }

    require(
        observed_nonzero == EXPECTED_CUSTOM_COUNTS[experiment_id],
        f"corrected-module signature drift for {experiment_id}: {observed_nonzero}",
    )

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameters": params,
        "nc": 9,
        "class_names": list(ordered_names),
        "custom_module_counts": observed_nonzero,
        "validation_inference_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A12-D1 standardized six-model validation diagnostic preflight."
    )
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/A12_D1_PREFLIGHT"),
    )
    args = parser.parse_args()

    observed_head = git_value("rev-parse", "HEAD")
    require(observed_head == args.expected_repo_head, "Evaluation checkout HEAD mismatch")
    require(git_value("status", "--porcelain") == "", "Evaluation checkout must be clean")

    require(not args.output_root.exists(), f"Refusing existing output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    registry = verify_artifact_registry()
    historical = verify_historical_experiment_contracts()
    runtime = verify_runtime_contract()
    data = verify_dataset_binding(args.input_root, args.output_root)

    checkpoint_bindings: list[dict[str, Any]] = []

    for experiment_id in EXPERIMENT_IDS:
        expected_sha = EXPECTED_CHECKPOINTS[experiment_id]
        checkpoint = discover_checkpoint(args.input_root, expected_sha)
        model = inspect_model(experiment_id, checkpoint)

        checkpoint_bindings.append(
            {
                "experiment_id": experiment_id,
                "display_name": DISPLAY_NAMES[experiment_id],
                "registry_sha256": registry[experiment_id]["sha256"],
                "model": model,
            }
        )

    preflight = {
        "schema_version": "A12-D1-standardized-validation-diagnostic-preflight-v1.0",
        "status": "PASS",
        "purpose": "Freeze one common validation-only diagnostic environment for six frozen best.pt checkpoints.",
        "repository": {
            "branch": git_value("branch", "--show-current"),
            "head": observed_head,
            "worktree_clean": True,
        },
        "runtime_contract": EXPECTED_RUNTIME,
        "observed_runtime": runtime,
        "data": data,
        "experiment_ids": list(EXPERIMENT_IDS),
        "checkpoint_bindings": checkpoint_bindings,
        "historical_training_bindings_preserved": True,
        "selection_metrics_replaced": False,
        "planned_d2": {
            "evaluation_split": "val",
            "save_txt": True,
            "save_conf": True,
            "prediction_floor": 0.001,
            "confidence_thresholds": [0.05, 0.10, 0.25, 0.50],
            "localization_iou_thresholds": [0.50, 0.75],
            "normalized_area_bins": {
                "small": "[0,0.01)",
                "medium": "[0.01,0.05)",
                "large": "[0.05,1]",
            },
            "patient_mapping": "research/04_data/split_b/split_B_val.csv",
        },
        "training_started": False,
        "validation_inference_started": False,
        "test_access": "NONE",
        "new_gpu_training_authorized": False,
        "a12_d2_authorized": False,
        "next_action": "REVIEW_A12_D1_PREFLIGHT_BEFORE_A12_D2",
    }

    preflight_path = args.output_root / "A12_D1_PREFLIGHT.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=" * 110)
    print("A12-D1 STANDARDIZED VALIDATION DIAGNOSTIC PREFLIGHT")
    print("=" * 110)
    print(f"REPOSITORY_HEAD={observed_head}")
    print(f"MODEL_COUNT={len(EXPERIMENT_IDS)}")
    print(f"DATA_BINDING={data['binding_id']}")
    print(f"FROZEN_VALIDATION_IMAGES={data['frozen_validation_images']}")
    print(f"OPERATIONAL_VALIDATION_IMAGES={data['operational_validation_images']}")
    print(f"VALIDATION_PATIENTS={data['validation_patients']}")
    print(f"RUNTIME_YAML_CONTAINS_TEST={data['runtime_yaml_contains_test']}")
    for record in checkpoint_bindings:
        print(
            "MODEL_BINDING="
            + record["experiment_id"]
            + "|"
            + record["model"]["checkpoint_sha256"]
            + "|params="
            + str(record["model"]["parameters"])
            + "|custom="
            + json.dumps(record["model"]["custom_module_counts"], sort_keys=True)
        )
    print(f"PREFLIGHT_MANIFEST={preflight_path}")
    print(f"PREFLIGHT_SHA256={sha256_file(preflight_path)}")
    print("TRAINING_STARTED=FALSE")
    print("VALIDATION_INFERENCE_STARTED=FALSE")
    print("TEST_ACCESS=NONE")
    print("NEW_GPU_TRAINING_AUTHORIZED=FALSE")
    print("A12_D2_AUTHORIZED=FALSE")
    print("A12_D1_PREFLIGHT=PASS")
    print("NEXT_ACTION=REVIEW_A12_D1_PREFLIGHT_BEFORE_A12_D2")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
