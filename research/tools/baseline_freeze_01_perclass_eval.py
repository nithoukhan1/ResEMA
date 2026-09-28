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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_experiment(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    matches = [x for x in registry["experiments"] if x["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise RuntimeError(f"Experiment must resolve exactly once: {experiment_id}")
    return matches[0]


def discover_checkpoint(input_root: Path, expected_sha: str) -> Path:
    matches: list[Path] = []
    for path in input_root.rglob("best.pt"):
        if not path.is_file():
            continue
        if sha256_file(path) == expected_sha:
            matches.append(path.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one best.pt with SHA {expected_sha}; observed {matches}"
        )
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/BASELINE_FREEZE_01_PERCLASS"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    registry_path = args.registry.resolve()
    registry = load_registry(registry_path)
    exp = find_experiment(registry, args.experiment_id)
    runtime_contract = registry["runtime_contract"]

    observed_head = git_head(repo)
    if observed_head != exp["execution_commit"]:
        raise RuntimeError(
            f"Execution checkout mismatch: {observed_head} != {exp['execution_commit']}"
        )

    # Force imports to the exact historical runtime checkout.
    sys.path.insert(0, str(repo))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics_config")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import torch
    import ultralytics

    if platform.python_version() != runtime_contract["python"]:
        raise RuntimeError(
            f"Python drift: {platform.python_version()} != {runtime_contract['python']}"
        )
    if torch.__version__ != runtime_contract["torch"]:
        raise RuntimeError(f"Torch drift: {torch.__version__} != {runtime_contract['torch']}")
    if ultralytics.__version__ != runtime_contract["ultralytics"]:
        raise RuntimeError(
            f"Ultralytics drift: {ultralytics.__version__} != {runtime_contract['ultralytics']}"
        )
    ultralytics_source = Path(ultralytics.__file__).resolve()
    if repo not in ultralytics_source.parents:
        raise RuntimeError(
            f"Ultralytics import is not from exact checkout: {ultralytics_source}"
        )

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if gpu_names != runtime_contract["gpu_inventory"]:
        raise RuntimeError(
            f"GPU inventory drift: {gpu_names} != {runtime_contract['gpu_inventory']}"
        )

    # Import governed DATA-01 discovery helpers from the exact historical checkout.
    from research.runtime import baseline_runner as br

    _fields, matrix_rows = br._read_current_experiment_matrix()
    matrix_matches = [r for r in matrix_rows if r["experiment_id"] == args.experiment_id]
    if len(matrix_matches) != 1:
        raise RuntimeError("Historical experiment row does not resolve exactly once.")
    row = matrix_matches[0]

    if row["source_commit"] != exp["source_commit"]:
        raise RuntimeError(
            f"Source commit binding mismatch: {row['source_commit']} != {exp['source_commit']}"
        )
    if row["data_binding"] != exp["data_binding"]:
        raise RuntimeError(
            f"Data binding mismatch: {row['data_binding']} != {exp['data_binding']}"
        )
    if row["initialization"] != exp["initialization"]:
        raise RuntimeError(
            f"Initialization mismatch: {row['initialization']} != {exp['initialization']}"
        )

    data_manifest = br.load_json(br.DATA_BINDINGS_JSON)
    expectation = br.binding_expectation(row, data_manifest)

    # Membership discovery explicitly prunes test-like directories.
    train_images = br.discover_membership_directory(
        args.input_root,
        kind="images",
        expected_count=expectation["train_count"],
        expected_hash=expectation["train_hash"],
    )
    train_labels = br.discover_membership_directory(
        args.input_root,
        kind="labels",
        expected_count=expectation["train_count"],
        expected_hash=expectation["train_hash"],
    )
    val_images = br.discover_membership_directory(
        args.input_root,
        kind="images",
        expected_count=expectation["validation_count"],
        expected_hash=expectation["validation_hash"],
    )
    val_labels = br.discover_membership_directory(
        args.input_root,
        kind="labels",
        expected_count=expectation["validation_count"],
        expected_hash=expectation["validation_hash"],
    )
    br.verify_image_label_pair(train_images, train_labels)
    br.verify_image_label_pair(val_images, val_labels)

    checkpoint = discover_checkpoint(args.input_root, exp["checkpoint_sha256"])

    output_dir = args.output_root / args.experiment_id
    if output_dir.exists():
        raise RuntimeError(f"Refusing existing validation output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    runtime_yaml = output_dir / "runtime_data_train_val_only.yaml"
    runtime_yaml_sha = br.write_runtime_data_yaml(
        runtime_yaml,
        train_images=train_images,
        validation_images=val_images,
    )
    runtime_payload = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    if "test" in runtime_payload:
        raise RuntimeError("Validation runtime YAML must not contain test.")

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    if getattr(model.model, "nc", None) != 9:
        raise RuntimeError(f"Checkpoint nc drift: {getattr(model.model, 'nc', None)}")

    val_project = output_dir / "native_val"
    val_name = "standardized_validation_only"

    metrics = model.val(
        data=str(runtime_yaml),
        split="val",
        imgsz=int(runtime_contract["imgsz"]),
        batch=int(runtime_contract["batch"]),
        device=int(runtime_contract["evaluation_device"]),
        workers=int(runtime_contract["workers"]),
        rect=bool(runtime_contract["rect"]),
        conf=runtime_contract["conf"],
        iou=float(runtime_contract["iou"]),
        max_det=int(runtime_contract["max_det"]),
        half=bool(runtime_contract["half"]),
        plots=bool(runtime_contract["plots"]),
        save_json=bool(runtime_contract["save_json"]),
        save_txt=False,
        save_conf=False,
        project=str(val_project),
        name=val_name,
        exist_ok=False,
        verbose=True,
        seed=int(runtime_contract["seed"]),
        deterministic=bool(runtime_contract["deterministic"]),
    )

    class_names = registry["class_names"]
    nt_class = [int(x) for x in metrics.nt_per_class.tolist()]
    nt_image = [int(x) for x in metrics.nt_per_image.tolist()]
    ap_index = [int(x) for x in metrics.ap_class_index.tolist()]
    position = {class_id: i for i, class_id in enumerate(ap_index)}

    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        images = nt_image[class_id]
        instances = nt_class[class_id]
        if class_id in position:
            i = position[class_id]
            p, r, ap50, ap = metrics.class_result(i)
            f1 = float(metrics.box.f1[i])
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "support_images": images,
                    "support_instances": instances,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": f1,
                    "ap50": float(ap50),
                    "ap50_95": float(ap),
                    "status": "EVALUATED",
                }
            )
        else:
            if instances != 0 or images != 0:
                raise RuntimeError(
                    f"Class {class_id} has support but no metric index: images={images}, instances={instances}"
                )
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "support_images": 0,
                    "support_instances": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ap50": None,
                    "ap50_95": None,
                    "status": "NO_VALIDATION_SUPPORT",
                }
            )

    mp, mr, map50, map5095 = [float(x) for x in metrics.mean_results()]
    aggregate_f1 = 0.0 if mp + mr == 0 else 2.0 * mp * mr / (mp + mr)
    mean_class_f1 = (
        float(metrics.box.f1.mean()) if len(metrics.box.f1) else 0.0
    )

    summary = {
        "schema_version": "BASELINE-FREEZE-01-standardized-val-v1.0",
        "experiment_id": args.experiment_id,
        "training_started": False,
        "test_access": "NONE",
        "evaluation_split": "val",
        "selection_metrics_replaced": False,
        "provenance": {
            "source_commit": exp["source_commit"],
            "execution_commit": exp["execution_commit"],
            "runtime_group": exp["runtime_group"],
            "data_binding": exp["data_binding"],
            "initialization": exp["initialization"],
            "checkpoint_kaggle_ref": exp["checkpoint_kaggle_ref"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "canonical_archive": exp["canonical_archive"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "ultralytics_source": str(ultralytics_source),
            "gpu_inventory": gpu_names,
            "evaluation_device": runtime_contract["evaluation_device"],
            "imgsz": runtime_contract["imgsz"],
            "batch": runtime_contract["batch"],
            "workers": runtime_contract["workers"],
            "rect": runtime_contract["rect"],
            "conf": runtime_contract["conf"],
            "iou": runtime_contract["iou"],
            "max_det": runtime_contract["max_det"],
            "half": runtime_contract["half"],
            "plots": runtime_contract["plots"],
            "deterministic": runtime_contract["deterministic"],
            "seed": runtime_contract["seed"],
        },
        "data": {
            "runtime_yaml": str(runtime_yaml),
            "runtime_yaml_sha256": runtime_yaml_sha,
            "runtime_yaml_contains_test": False,
            "train_images": str(train_images),
            "validation_images": str(val_images),
            "frozen_validation_membership": expectation["validation_count"],
            "validation_membership_sha256": expectation["validation_hash"],
            "expected_operational_validation_images": expectation["operational_validation_images"],
            "test_directory_scan_pruned": True,
        },
        "aggregate": {
            "precision": mp,
            "recall": mr,
            "f1_from_mean_pr": aggregate_f1,
            "mean_class_f1": mean_class_f1,
            "map50": map50,
            "map50_95": map5095,
            "evaluation_class_count": len(ap_index),
        },
        "per_class": per_class,
    }

    per_class_path = output_dir / "PER_CLASS_METRICS.csv"
    write_csv(
        per_class_path,
        per_class,
        [
            "class_id","class_name","support_images","support_instances",
            "precision","recall","f1","ap50","ap50_95","status",
        ],
    )

    summary_path = output_dir / "VALIDATION_ONLY_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Manifest all generated validation artifacts except the manifest itself.
    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        if path.name == "VALIDATION_ARTIFACT_MANIFEST.csv":
            continue
        manifest_rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path = output_dir / "VALIDATION_ARTIFACT_MANIFEST.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        ["relative_path","bytes","sha256"],
    )

    print("=" * 110)
    print("BASELINE-FREEZE-01 STANDARDIZED VALIDATION-ONLY PASS")
    print("=" * 110)
    print(f"EXPERIMENT_ID={args.experiment_id}")
    print(f"EXECUTION_COMMIT={observed_head}")
    print(f"DATA_BINDING={exp['data_binding']}")
    print(f"CHECKPOINT_SHA256={sha256_file(checkpoint)}")
    print(f"FROZEN_VALIDATION_MEMBERSHIP={expectation['validation_count']}")
    print(f"EXPECTED_OPERATIONAL_VALIDATION_IMAGES={expectation['operational_validation_images']}")
    print(f"EVALUATION_CLASS_COUNT={len(ap_index)}")
    print(f"VAL_PRECISION={mp:.8f}")
    print(f"VAL_RECALL={mr:.8f}")
    print(f"VAL_F1={aggregate_f1:.8f}")
    print(f"VAL_MAP50={map50:.8f}")
    print(f"VAL_MAP50_95={map5095:.8f}")
    print(f"SUMMARY={summary_path}")
    print(f"PER_CLASS={per_class_path}")
    print(f"MANIFEST={manifest_path}")
    print("TRAINING_STARTED=FALSE")
    print("TEST_ACCESS=NONE")
    print("VALIDATION_ONLY_PASS=TRUE")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
