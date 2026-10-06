#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_CSV = ROOT / "research/01_provenance/ARTIFACTS.csv"
D1_RUNNER = ROOT / "research/runtime/a12_d1_validation_diagnostic.py"
D1_PROTOCOL = (
    ROOT
    / "research/06_diagnostics/"
      "A12_D1_STANDARDIZED_VALIDATION_DIAGNOSTIC_PROTOCOL.md"
)
D2_AUTHORIZATION = (
    ROOT
    / "research/06_diagnostics/"
      "A12_D2_EXECUTION_AUTHORIZATION.json"
)

EXPECTED_BRANCH = "research/combination-screen-01"

D1_SOURCE_COMMIT = "e2367057e3a4ffabcb5cde1c2ff569df264625ef"
D1_CLOSURE_COMMIT = "7aff6edf2623135f90776c0760a8e80b6fc5298f"
CHRONOLOGY_COMMIT = "5aa7eb6acffcb5c2922e6b2c16aae7e80a5a20e9"

D1_RUNNER_SHA256 = (
    "8fe88a327b01b71bf3b9279dd4e8a6dffb5487cc2bd3f173143c665fffc997a7"
)

D1_EVIDENCE = {
    "runtime_preflight_manifest": {
        "logical_name": "A12_D1_PREFLIGHT.json",
        "sha256": "d33ead0f712aa432e4afdd67aa89f6a4481acc543ae0486b8424a95761909de8",
    },
    "runtime_data_yaml": {
        "logical_name": "runtime_data_train_val_only.yaml",
        "sha256": "b2114397eefc1ac377352fb4e0429df0fe6fc954a307e4b0df89139b4bed82c4",
    },
    "runtime_preflight_preservation_manifest": {
        "logical_name": "A12_D1_RUNTIME_PREFLIGHT_PRESERVATION.json",
        "sha256": "217d67b401afbeca76033a7eef381649ef36e4f0f3b829c76b47216a8cc6039d",
    },
    "runtime_preflight_archive": {
        "logical_name": "A12_D1_RUNTIME_PREFLIGHT_E2367057.zip",
        "sha256": "f9f47b7816caec3556fd1700aee79f8e716a8efa866bb9030e571a947b4d9de8",
    },
}

EXPECTED_NATIVE_PLOTS = (
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxPR_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)

NATIVE_CONFUSION_MATRIX_CONTRACT = {
    "effective_confidence": 0.25,
    "iou_threshold": 0.45,
    "role": (
        "Ultralytics native visualization only; not the frozen offline "
        "A12 diagnostic error taxonomy."
    ),
}

PER_CLASS_FIELDS = [
    "class_id",
    "class_name",
    "support_images",
    "support_instances",
    "precision",
    "recall",
    "f1",
    "ap50",
    "ap75",
    "ap50_95",
    "status",
]

AGGREGATE_FIELDS = [
    "experiment_id",
    "display_name",
    "checkpoint_sha256",
    "precision",
    "recall",
    "f1_from_mean_pr",
    "mean_class_f1",
    "map50",
    "map75",
    "map50_95",
    "evaluation_class_count",
    "prediction_file_count",
    "prediction_row_count",
    "minimum_exported_confidence",
    "maximum_exported_confidence",
    "summary_sha256",
    "per_class_sha256",
    "predictions_csv_sha256",
    "prediction_audit_sha256",
    "native_validation_manifest_sha256",
]

COMBINED_PER_CLASS_FIELDS = [
    "experiment_id",
    "display_name",
    *PER_CLASS_FIELDS,
]

IMAGE_INDEX_FIELDS = [
    "filestem",
    "patient_id",
    "operational_readable",
    "known_unreadable",
    "ground_truth_box_count",
]

GROUND_TRUTH_FIELDS = [
    "filestem",
    "patient_id",
    "box_index",
    "class_id",
    "class_name",
    "x_center",
    "y_center",
    "width",
    "height",
    "area_norm",
    "size_bin",
]

PREDICTION_FIELDS = [
    "filestem",
    "prediction_index",
    "class_id",
    "class_name",
    "x_center",
    "y_center",
    "width",
    "height",
    "area_norm",
    "confidence",
]


class GovernanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GovernanceError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def git_file_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{commit}:{relative_path}",
        ]
    )


def markdown_status(text: str) -> str:
    lines = text.splitlines()
    try:
        heading_index = lines.index("## Status")
    except ValueError as exc:
        raise GovernanceError("Protocol is missing the exact '## Status' heading") from exc

    for line in lines[heading_index + 1:]:
        stripped = line.strip()
        if stripped:
            return stripped

    raise GovernanceError("Protocol Status heading has no status value")


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None, f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def size_bin(area_norm: float) -> str:
    if area_norm < 0.01:
        return "small"
    if area_norm < 0.05:
        return "medium"
    return "large"


def verify_repository(expected_head: str) -> dict[str, Any]:
    branch = git_value("branch", "--show-current")
    head = git_value("rev-parse", "HEAD")
    dirty = git_value(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    require(branch == EXPECTED_BRANCH, f"Branch drift: {branch}")
    require(head == expected_head, f"Execution HEAD drift: {head}")
    require(dirty == "", "A12-D2 requires a clean source checkout")

    for required_commit, label in (
        (D1_SOURCE_COMMIT, "D1 source freeze"),
        (D1_CLOSURE_COMMIT, "D1 closure"),
        (CHRONOLOGY_COMMIT, "chronology correction"),
    ):
        require(
            git_is_ancestor(required_commit, head),
            f"{label} is not an ancestor of A12-D2 execution source",
        )

    model_source_drift = git_value(
        "diff",
        "--name-only",
        D1_SOURCE_COMMIT,
        head,
        "--",
        "ultralytics",
    )
    require(
        model_source_drift == "",
        "Ultralytics/model source changed after the frozen D1 source:\n"
        + model_source_drift,
    )

    require(
        sha256_file(D1_RUNNER) == D1_RUNNER_SHA256,
        "Frozen D1 preflight runner SHA256 drift",
    )

    return {
        "branch": branch,
        "head": head,
        "worktree_clean": True,
        "d1_source_commit": D1_SOURCE_COMMIT,
        "d1_closure_commit": D1_CLOSURE_COMMIT,
        "chronology_commit": CHRONOLOGY_COMMIT,
        "d1_runner_sha256": D1_RUNNER_SHA256,
        "ultralytics_drift_since_d1_source": False,
    }


def verify_execution_authorization(current_head: str) -> dict[str, Any]:
    require(
        D2_AUTHORIZATION.is_file(),
        "A12-D2 execution authorization file is absent. "
        "Validation inference is not authorized.",
    )

    payload = json.loads(
        D2_AUTHORIZATION.read_text(encoding="utf-8")
    )

    required = {
        "schema_version": "A12-D2-execution-authorization-v1.0",
        "status": "AUTHORIZED",
        "d1_source_commit": D1_SOURCE_COMMIT,
        "d1_preflight_sha256":
            D1_EVIDENCE["runtime_preflight_manifest"]["sha256"],
        "data_binding": "DATA01:B-ORG:v1",
        "model_count": 6,
        "test_access": "NONE",
        "new_gpu_training_authorized": False,
        "validation_execution_authorized": True,
    }

    for key, expected in required.items():
        require(
            payload.get(key) == expected,
            f"A12-D2 authorization field drift: "
            f"{key}={payload.get(key)!r} expected={expected!r}",
        )

    source_commit = payload.get("source_commit", "")
    runner_sha = payload.get("runner_sha256", "")

    require(
        len(source_commit) == 40,
        "A12-D2 authorization source_commit is not a 40-character Git SHA",
    )
    require(
        len(runner_sha) == 64,
        "A12-D2 authorization runner_sha256 is not a SHA256",
    )
    require(
        git_is_ancestor(source_commit, current_head),
        "Authorized D2 source commit is not an ancestor of execution HEAD",
    )

    current_runner_sha = sha256_file(Path(__file__).resolve())
    require(
        current_runner_sha == runner_sha,
        "Current D2 runner does not match authorization runner SHA256",
    )

    frozen_runner_bytes = git_file_bytes(
        source_commit,
        "research/runtime/a12_d2_standardized_validation.py",
    )
    require(
        sha256_bytes(frozen_runner_bytes) == runner_sha,
        "Authorized D2 source commit does not contain the authorized runner",
    )

    protocol_text = D1_PROTOCOL.read_text(encoding="utf-8")
    protocol_state = markdown_status(protocol_text)
    require(
        protocol_state == "`A12_D2_EXECUTION_AUTHORIZED`",
        "Protocol Status is not the exact A12-D2 authorized state: "
        + protocol_state,
    )

    return {
        **payload,
        "authorization_file": str(D2_AUTHORIZATION),
        "authorization_file_sha256": sha256_file(D2_AUTHORIZATION),
        "current_runner_sha256": current_runner_sha,
    }


def verify_d1_registered_evidence() -> list[dict[str, str]]:
    _fields, rows = load_csv(ARTIFACTS_CSV)
    matches = [
        row
        for row in rows
        if row["experiment_id"] == "A12-D1"
    ]

    require(
        len(matches) == 4,
        f"Expected exactly four registered A12-D1 artifacts; observed {len(matches)}",
    )

    by_role = {row["artifact_role"]: row for row in matches}
    require(
        set(by_role) == set(D1_EVIDENCE),
        f"A12-D1 artifact-role drift: {sorted(by_role)}",
    )

    for role, expected in D1_EVIDENCE.items():
        row = by_role[role]
        require(row["status"] == "VERIFIED", f"D1 artifact not VERIFIED: {role}")
        require(
            row["source_commit"] == D1_SOURCE_COMMIT,
            f"D1 artifact source-commit drift: {role}",
        )
        require(
            row["logical_name"] == expected["logical_name"],
            f"D1 artifact logical-name drift: {role}",
        )
        require(
            row["sha256"] == expected["sha256"],
            f"D1 artifact SHA drift: {role}",
        )

    return [by_role[role] for role in D1_EVIDENCE]


def build_validation_reference(
    data: dict[str, Any],
    d1: Any,
    output_root: Path,
) -> dict[str, Any]:
    _fields, metadata_rows = load_csv(d1.VAL_METADATA)

    require(
        len(metadata_rows) == d1.EXPECTED_B_ORG["val_count"],
        "Validation metadata row-count drift",
    )

    by_stem: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        stem = row["filestem"]
        require(stem not in by_stem, f"Duplicate validation filestem: {stem}")
        by_stem[stem] = row

    require(
        len({row["patient_id"] for row in metadata_rows})
        == d1.EXPECTED_B_ORG["val_patients"],
        "Validation patient-count drift",
    )

    label_dir = Path(data["validation_labels"])
    require(label_dir.is_dir(), f"Validation labels directory missing: {label_dir}")

    label_files = sorted(
        p
        for p in label_dir.glob("*.txt")
        if p.is_file()
    )
    require(
        len(label_files) == d1.EXPECTED_B_ORG["val_count"],
        f"Validation label-file count drift: {len(label_files)}",
    )

    label_by_stem = {p.stem: p for p in label_files}
    require(
        set(label_by_stem) == set(by_stem),
        "Validation metadata/label filestem membership mismatch",
    )

    known_unreadable_stem = Path(
        d1.EXPECTED_B_ORG["known_unreadable"]
    ).stem

    image_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    class_support = {class_id: 0 for class_id in range(len(d1.CLASS_NAMES))}

    for stem in sorted(by_stem):
        metadata = by_stem[stem]
        label_path = label_by_stem[stem]
        box_count = 0

        for line_number, raw in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            stripped = raw.strip()
            if not stripped:
                continue

            parts = stripped.split()
            require(
                len(parts) == 5,
                f"Expected YOLO class+xywh ground truth: "
                f"{label_path}:{line_number}",
            )

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = (
                    float(x) for x in parts[1:]
                )
            except ValueError as exc:
                raise GovernanceError(
                    f"Non-numeric validation ground truth: "
                    f"{label_path}:{line_number}"
                ) from exc

            require(
                0 <= class_id < len(d1.CLASS_NAMES),
                f"Ground-truth class out of range: "
                f"{label_path}:{line_number}",
            )
            require(
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0,
                f"Ground-truth normalized geometry out of range: "
                f"{label_path}:{line_number}",
            )

            area = width * height
            gt_rows.append(
                {
                    "filestem": stem,
                    "patient_id": metadata["patient_id"],
                    "box_index": box_count,
                    "class_id": class_id,
                    "class_name": d1.CLASS_NAMES[class_id],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area_norm": area,
                    "size_bin": size_bin(area),
                }
            )
            box_count += 1
            class_support[class_id] += 1

        is_unreadable = stem == known_unreadable_stem
        image_rows.append(
            {
                "filestem": stem,
                "patient_id": metadata["patient_id"],
                "operational_readable": not is_unreadable,
                "known_unreadable": is_unreadable,
                "ground_truth_box_count": box_count,
            }
        )

    require(
        sum(not row["operational_readable"] for row in image_rows) == 1,
        "Expected exactly one governed unreadable validation image",
    )
    require(
        class_support[2] == 0,
        "foreignbody unexpectedly has Split-B validation ground-truth support",
    )

    image_index_path = output_root / "VALIDATION_IMAGE_INDEX.csv"
    ground_truth_path = output_root / "VALIDATION_GROUND_TRUTH.csv"

    write_csv(
        image_index_path,
        image_rows,
        IMAGE_INDEX_FIELDS,
    )
    write_csv(
        ground_truth_path,
        gt_rows,
        GROUND_TRUTH_FIELDS,
    )

    return {
        "image_index": str(image_index_path),
        "image_index_sha256": sha256_file(image_index_path),
        "image_count": len(image_rows),
        "operational_readable_image_count":
            sum(row["operational_readable"] for row in image_rows),
        "known_unreadable_image_count":
            sum(row["known_unreadable"] for row in image_rows),
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "ground_truth_box_count": len(gt_rows),
        "ground_truth_class_support": {
            str(class_id): class_support[class_id]
            for class_id in sorted(class_support)
        },
        "patient_count": len(
            {row["patient_id"] for row in metadata_rows}
        ),
        "known_unreadable_filestem": known_unreadable_stem,
    }


def audit_prediction_exports(
    labels_dir: Path,
    predictions_csv: Path,
    class_names: tuple[str, ...],
    valid_filestems: set[str],
    known_unreadable_stem: str,
) -> dict[str, Any]:
    require(
        labels_dir.is_dir(),
        f"Prediction labels directory missing despite save_txt=True: {labels_dir}",
    )

    files = sorted(
        p
        for p in labels_dir.glob("*.txt")
        if p.is_file()
    )

    prediction_rows: list[dict[str, Any]] = []
    confidences: list[float] = []

    for path in files:
        require(
            path.stem in valid_filestems,
            f"Prediction export contains non-validation filestem: {path.stem}",
        )
        require(
            path.stem != known_unreadable_stem,
            "Known unreadable validation image unexpectedly has predictions",
        )

        for prediction_index, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            stripped = raw.strip()
            if not stripped:
                continue

            parts = stripped.split()
            require(
                len(parts) == 6,
                f"Expected class+xywh+confidence (6 fields): "
                f"{path}:{prediction_index + 1} -> {parts}",
            )

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = (
                    float(x) for x in parts[1:5]
                )
                confidence = float(parts[5])
            except ValueError as exc:
                raise GovernanceError(
                    f"Non-numeric prediction export row: "
                    f"{path}:{prediction_index + 1}"
                ) from exc

            require(
                0 <= class_id < len(class_names),
                f"Prediction class out of range: "
                f"{path}:{prediction_index + 1}",
            )
            require(
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 <= width <= 1.0
                and 0.0 <= height <= 1.0,
                f"Normalized prediction geometry out of range: "
                f"{path}:{prediction_index + 1}",
            )
            require(
                0.0 <= confidence <= 1.0,
                f"Prediction confidence out of range: "
                f"{path}:{prediction_index + 1}",
            )
            require(
                confidence + 1e-12 >= 0.001,
                f"Exported confidence below frozen validation floor: "
                f"{path}:{prediction_index + 1} -> {confidence}",
            )

            area = width * height
            prediction_rows.append(
                {
                    "filestem": path.stem,
                    "prediction_index": prediction_index,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area_norm": area,
                    "confidence": confidence,
                }
            )
            confidences.append(confidence)

    require(
        len(files) <= len(valid_filestems) - 1,
        "Prediction txt file count exceeds operational readable validation images",
    )

    write_csv(
        predictions_csv,
        prediction_rows,
        PREDICTION_FIELDS,
    )

    return {
        "labels_directory": str(labels_dir),
        "prediction_file_count": len(files),
        "prediction_row_count": len(prediction_rows),
        "minimum_exported_confidence": (
            min(confidences) if confidences else None
        ),
        "maximum_exported_confidence": (
            max(confidences) if confidences else None
        ),
        "predictions_csv": str(predictions_csv),
        "predictions_csv_sha256": sha256_file(predictions_csv),
        "save_txt": True,
        "save_conf": True,
        "validation_conf_argument": None,
        "effective_prediction_floor": 0.001,
    }


def build_per_class(
    metrics: Any,
    class_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nt_class = [int(x) for x in metrics.nt_per_class.tolist()]
    nt_image = [int(x) for x in metrics.nt_per_image.tolist()]
    ap_index = [int(x) for x in metrics.ap_class_index.tolist()]
    position = {
        class_id: index
        for index, class_id in enumerate(ap_index)
    }

    require(
        len(nt_class) == len(class_names),
        f"Target-count class vector drift: {len(nt_class)}",
    )
    require(
        len(nt_image) == len(class_names),
        f"Target-image class vector drift: {len(nt_image)}",
    )

    all_ap = metrics.box.all_ap
    require(
        len(all_ap) == len(ap_index),
        "AP matrix row count does not match AP class index",
    )
    if len(ap_index):
        require(
            all_ap.shape[1] == 10,
            f"Expected 10 IoU AP columns; observed {all_ap.shape}",
        )

    rows: list[dict[str, Any]] = []

    for class_id, class_name in enumerate(class_names):
        images = nt_image[class_id]
        instances = nt_class[class_id]

        if class_id in position:
            i = position[class_id]
            p, r, ap50, ap5095 = metrics.class_result(i)
            f1 = float(metrics.box.f1[i])
            ap75 = float(metrics.box.all_ap[i, 5])

            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "support_images": images,
                    "support_instances": instances,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": f1,
                    "ap50": float(ap50),
                    "ap75": ap75,
                    "ap50_95": float(ap5095),
                    "status": "EVALUATED",
                }
            )
        else:
            require(
                instances == 0 and images == 0,
                f"Class has validation support but no AP index: "
                f"{class_name} images={images} instances={instances}",
            )

            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "support_images": 0,
                    "support_instances": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ap50": None,
                    "ap75": None,
                    "ap50_95": None,
                    "status": "NO_VALIDATION_SUPPORT",
                }
            )

    foreignbody = rows[2]
    require(
        foreignbody["class_name"] == "foreignbody",
        "Class-2 identity drift",
    )
    require(
        foreignbody["status"] == "NO_VALIDATION_SUPPORT",
        "foreignbody must remain N/A on Split-B validation",
    )

    mp, mr, map50, map5095 = [
        float(x)
        for x in metrics.mean_results()
    ]
    aggregate_f1 = (
        0.0
        if mp + mr == 0
        else 2.0 * mp * mr / (mp + mr)
    )
    mean_class_f1 = (
        float(metrics.box.f1.mean())
        if len(metrics.box.f1)
        else 0.0
    )

    aggregate = {
        "precision": mp,
        "recall": mr,
        "f1_from_mean_pr": aggregate_f1,
        "mean_class_f1": mean_class_f1,
        "map50": map50,
        "map75": float(metrics.box.map75),
        "map50_95": map5095,
        "evaluation_class_count": len(ap_index),
    }

    return rows, aggregate


def manifest_directory(
    directory: Path,
    manifest_name: str,
) -> tuple[Path, str, int]:
    rows: list[dict[str, Any]] = []

    for path in sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.name != manifest_name
    ):
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest_path = directory / manifest_name
    write_csv(
        manifest_path,
        rows,
        ["relative_path", "bytes", "sha256"],
    )
    return manifest_path, sha256_file(manifest_path), len(rows)


def evaluate_one(
    experiment_id: str,
    display_name: str,
    checkpoint: Path,
    runtime_yaml: Path,
    output_root: Path,
    runtime_contract: dict[str, Any],
    class_names: tuple[str, ...],
    valid_filestems: set[str],
    known_unreadable_stem: str,
) -> dict[str, Any]:
    from ultralytics import YOLO
    import torch

    experiment_dir = output_root / experiment_id
    require(
        not experiment_dir.exists(),
        f"Refusing existing experiment output directory: {experiment_dir}",
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)

    val_project = experiment_dir / "native_val"
    val_name = "standardized_validation_only"

    start = time.time()

    model = YOLO(str(checkpoint))

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
        save_txt=True,
        save_conf=True,
        project=str(val_project),
        name=val_name,
        exist_ok=False,
        verbose=True,
        seed=int(runtime_contract["seed"]),
        deterministic=bool(runtime_contract["deterministic"]),
    )

    elapsed = time.time() - start

    native_dir = val_project / val_name
    require(
        native_dir.is_dir(),
        f"Native validation output directory missing: {native_dir}",
    )

    for filename in EXPECTED_NATIVE_PLOTS:
        require(
            (native_dir / filename).is_file(),
            f"Expected native validation plot missing: "
            f"{experiment_id}/{filename}",
        )

    per_class, aggregate = build_per_class(
        metrics,
        class_names,
    )

    predictions_csv = experiment_dir / "PREDICTIONS.csv"
    prediction_audit = audit_prediction_exports(
        native_dir / "labels",
        predictions_csv,
        class_names,
        valid_filestems,
        known_unreadable_stem,
    )

    prediction_audit_path = (
        experiment_dir / "PREDICTION_EXPORT_AUDIT.json"
    )
    write_json(
        prediction_audit_path,
        prediction_audit,
    )

    per_class_path = (
        experiment_dir / "PER_CLASS_METRICS.csv"
    )
    write_csv(
        per_class_path,
        per_class,
        PER_CLASS_FIELDS,
    )

    summary = {
        "schema_version":
            "A12-D2-standardized-validation-model-v1.1",
        "experiment_id": experiment_id,
        "display_name": display_name,
        "evaluation_split": "val",
        "selection_metrics_replaced": False,
        "historical_training_selection_metrics_preserved": True,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "runtime": {
            "python": platform.python_version(),
            "runtime_contract": runtime_contract,
            "elapsed_seconds": elapsed,
        },
        "aggregate": aggregate,
        "per_class": per_class,
        "prediction_export": prediction_audit,
        "native_confusion_matrix_contract":
            NATIVE_CONFUSION_MATRIX_CONTRACT,
        "native_validation_output": str(native_dir),
        "training_started": False,
        "test_access": "NONE",
    }

    summary_path = (
        experiment_dir / "VALIDATION_ONLY_SUMMARY.json"
    )
    write_json(
        summary_path,
        summary,
    )

    native_manifest_path, native_manifest_sha, native_count = (
        manifest_directory(
            native_dir,
            "NATIVE_VALIDATION_ARTIFACT_MANIFEST.csv",
        )
    )

    model_manifest_path, model_manifest_sha, model_file_count = (
        manifest_directory(
            experiment_dir,
            "VALIDATION_ARTIFACT_MANIFEST.csv",
        )
    )

    # Release per-model resources before the next checkpoint is evaluated.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment_id": experiment_id,
        "display_name": display_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        **aggregate,
        "prediction_file_count":
            prediction_audit["prediction_file_count"],
        "prediction_row_count":
            prediction_audit["prediction_row_count"],
        "minimum_exported_confidence":
            prediction_audit["minimum_exported_confidence"],
        "maximum_exported_confidence":
            prediction_audit["maximum_exported_confidence"],
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "per_class_path": str(per_class_path),
        "per_class_sha256": sha256_file(per_class_path),
        "predictions_csv": str(predictions_csv),
        "predictions_csv_sha256":
            prediction_audit["predictions_csv_sha256"],
        "prediction_audit_path": str(prediction_audit_path),
        "prediction_audit_sha256": sha256_file(prediction_audit_path),
        "native_validation_manifest": str(native_manifest_path),
        "native_validation_manifest_sha256": native_manifest_sha,
        "native_validation_file_count": native_count,
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": model_manifest_sha,
        "model_file_count": model_file_count,
        "per_class_rows": per_class,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A12-D2 common standardized validation execution for the six "
            "frozen Split-B Original checkpoints."
        )
    )
    parser.add_argument(
        "--expected-repo-head",
        required=True,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/kaggle/working/"
            "A12_D2_STANDARDIZED_VALIDATION"
        ),
    )
    args = parser.parse_args()

    # Critical: authorization is verified before output creation, dataset
    # discovery, checkpoint discovery/loading, or validation inference.
    repository = verify_repository(
        args.expected_repo_head
    )
    authorization = verify_execution_authorization(
        args.expected_repo_head
    )
    d1_evidence = verify_d1_registered_evidence()

    require(
        not args.output_root.exists(),
        f"Refusing existing A12-D2 output root: {args.output_root}",
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        "/kaggle/working/.ultralytics_config",
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    from research.runtime import (
        a12_d1_validation_diagnostic as d1,
    )

    require(
        d1.EXPECTED_RUNTIME["conf"] is None,
        "Frozen validation confidence contract drift",
    )
    require(
        d1.EXPECTED_RUNTIME["split"] == "val",
        "Frozen validation split contract drift",
    )

    # These checks do not access dataset images/labels or load checkpoints.
    registry = d1.verify_artifact_registry()
    historical = d1.verify_historical_experiment_contracts()
    observed_runtime = d1.verify_runtime_contract()

    args.output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    completed_ids: list[str] = []

    try:
        data = d1.verify_dataset_binding(
            args.input_root,
            args.output_root,
        )

        runtime_yaml = Path(data["runtime_yaml"])
        require(
            runtime_yaml.is_file(),
            "Runtime train+validation YAML missing",
        )
        require(
            sha256_file(runtime_yaml)
            == data["runtime_yaml_sha256"],
            "Runtime YAML changed after dataset binding",
        )
        require(
            data["runtime_yaml_contains_test"] is False,
            "Runtime YAML test firewall drift",
        )

        validation_reference = build_validation_reference(
            data,
            d1,
            args.output_root,
        )

        valid_filestems = set(
            row["filestem"]
            for row in load_csv(d1.VAL_METADATA)[1]
        )

        structural_bindings: list[dict[str, Any]] = []
        checkpoint_paths: dict[str, Path] = {}

        for experiment_id in d1.EXPERIMENT_IDS:
            expected_sha = d1.EXPECTED_CHECKPOINTS[
                experiment_id
            ]
            checkpoint = d1.discover_checkpoint(
                args.input_root,
                expected_sha,
            )
            model_info = d1.inspect_model(
                experiment_id,
                checkpoint,
            )

            require(
                model_info["checkpoint_sha256"]
                == registry[experiment_id]["sha256"],
                f"Registry/checkpoint mismatch: {experiment_id}",
            )

            checkpoint_paths[experiment_id] = checkpoint
            structural_bindings.append(
                {
                    "experiment_id": experiment_id,
                    "display_name":
                        d1.DISPLAY_NAMES[experiment_id],
                    "model": model_info,
                }
            )

        aggregate_rows: list[dict[str, Any]] = []
        combined_per_class: list[dict[str, Any]] = []
        execution_records: list[dict[str, Any]] = []

        for index, experiment_id in enumerate(
            d1.EXPERIMENT_IDS,
            start=1,
        ):
            display_name = d1.DISPLAY_NAMES[
                experiment_id
            ]

            print("\n" + "#" * 110)
            print(
                f"A12-D2 MODEL {index}/{len(d1.EXPERIMENT_IDS)}"
                f" :: {experiment_id}"
            )
            print("#" * 110, flush=True)

            record = evaluate_one(
                experiment_id=experiment_id,
                display_name=display_name,
                checkpoint=checkpoint_paths[
                    experiment_id
                ],
                runtime_yaml=runtime_yaml,
                output_root=args.output_root,
                runtime_contract=d1.EXPECTED_RUNTIME,
                class_names=d1.CLASS_NAMES,
                valid_filestems=valid_filestems,
                known_unreadable_stem=
                    validation_reference["known_unreadable_filestem"],
            )

            aggregate_rows.append(
                {
                    field: record[field]
                    for field in AGGREGATE_FIELDS
                }
            )

            for row in record["per_class_rows"]:
                combined_per_class.append(
                    {
                        "experiment_id": experiment_id,
                        "display_name": display_name,
                        **row,
                    }
                )

            execution_records.append(
                {
                    key: value
                    for key, value in record.items()
                    if key != "per_class_rows"
                }
            )

            completed_ids.append(experiment_id)

            print(
                f"A12_D2_MODEL_PASS={experiment_id}"
            )
            print(
                "VAL_MAP50_95="
                f"{record['map50_95']:.8f}"
            )
            print(
                "PREDICTION_ROWS="
                f"{record['prediction_row_count']}"
            )

        aggregate_path = (
            args.output_root
            / "STANDARDIZED_VALIDATION_AGGREGATE.csv"
        )
        write_csv(
            aggregate_path,
            aggregate_rows,
            AGGREGATE_FIELDS,
        )

        combined_per_class_path = (
            args.output_root
            / "STANDARDIZED_VALIDATION_PER_CLASS.csv"
        )
        write_csv(
            combined_per_class_path,
            combined_per_class,
            COMBINED_PER_CLASS_FIELDS,
        )

        execution_manifest = {
            "schema_version":
                "A12-D2-standardized-validation-execution-v1.1",
            "status": "PASS",
            "purpose": (
                "Run one common validation-only evaluation of six "
                "frozen checkpoints and preserve low-confidence "
                "post-NMS predictions for later offline diagnostics."
            ),
            "repository": repository,
            "authorization": authorization,
            "d1_evidence": d1_evidence,
            "runtime_contract": d1.EXPECTED_RUNTIME,
            "observed_runtime": observed_runtime,
            "data": data,
            "validation_reference": validation_reference,
            "historical_training_bindings_preserved": True,
            "historical_training_contracts": historical,
            "selection_metrics_replaced": False,
            "prediction_export_contract": {
                "save_txt": True,
                "save_conf": True,
                "validation_conf_argument": None,
                "effective_prediction_floor": 0.001,
                "derived_predictions_csv_per_model": True,
            },
            "native_confusion_matrix_contract":
                NATIVE_CONFUSION_MATRIX_CONTRACT,
            "frozen_offline_diagnostic_thresholds": {
                "confidence": [0.05, 0.10, 0.25, 0.50],
                "localization_iou": [0.50, 0.75],
                "normalized_area_bins": {
                    "small": "[0,0.01)",
                    "medium": "[0.01,0.05)",
                    "large": "[0.05,1]",
                },
            },
            "offline_diagnostics_executed": False,
            "structural_bindings": structural_bindings,
            "validation_records": execution_records,
            "aggregate_csv": str(aggregate_path),
            "aggregate_csv_sha256":
                sha256_file(aggregate_path),
            "combined_per_class_csv":
                str(combined_per_class_path),
            "combined_per_class_csv_sha256":
                sha256_file(combined_per_class_path),
            "training_started": False,
            "test_access": "NONE",
            "new_gpu_training_authorized": False,
            "next_action":
                "PRESERVE_AND_REVIEW_A12_D2_OUTPUTS_BEFORE_OFFLINE_DIAGNOSTICS",
        }

        execution_manifest_path = (
            args.output_root
            / "A12_D2_EXECUTION_MANIFEST.json"
        )
        write_json(
            execution_manifest_path,
            execution_manifest,
        )

        global_manifest_path, global_manifest_sha, global_file_count = (
            manifest_directory(
                args.output_root,
                "A12_D2_ARTIFACT_MANIFEST.csv",
            )
        )

        require(
            git_value(
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ) == "",
            "Repository became dirty during A12-D2 execution",
        )
        require(
            git_value("rev-parse", "HEAD")
            == args.expected_repo_head,
            "Repository HEAD changed during A12-D2 execution",
        )

        print("\n" + "=" * 110)
        print("A12-D2 STANDARDIZED VALIDATION EXECUTION PASS")
        print("=" * 110)
        print(
            f"REPOSITORY_HEAD={args.expected_repo_head}"
        )
        print(
            f"AUTHORIZED_SOURCE_COMMIT="
            f"{authorization['source_commit']}"
        )
        print(
            f"MODEL_COUNT={len(d1.EXPERIMENT_IDS)}"
        )
        print(
            f"DATA_BINDING={data['binding_id']}"
        )
        print(
            "FROZEN_VALIDATION_MEMBERSHIP="
            f"{data['frozen_validation_images']}"
        )
        print(
            "OPERATIONAL_VALIDATION_IMAGES="
            f"{data['operational_validation_images']}"
        )
        print(
            f"VALIDATION_PATIENTS="
            f"{data['validation_patients']}"
        )
        print(
            "VALIDATION_GROUND_TRUTH_BOXES="
            f"{validation_reference['ground_truth_box_count']}"
        )
        print(
            f"RUNTIME_YAML_SHA256="
            f"{data['runtime_yaml_sha256']}"
        )

        for row in aggregate_rows:
            print(
                "STANDARDIZED_RESULT="
                + row["experiment_id"]
                + "|P="
                + f"{float(row['precision']):.8f}"
                + "|R="
                + f"{float(row['recall']):.8f}"
                + "|F1="
                + f"{float(row['f1_from_mean_pr']):.8f}"
                + "|mAP50="
                + f"{float(row['map50']):.8f}"
                + "|mAP75="
                + f"{float(row['map75']):.8f}"
                + "|mAP50-95="
                + f"{float(row['map50_95']):.8f}"
                + "|pred_rows="
                + str(row["prediction_row_count"])
            )

        print(
            f"VALIDATION_IMAGE_INDEX="
            f"{validation_reference['image_index']}"
        )
        print(
            "VALIDATION_IMAGE_INDEX_SHA256="
            f"{validation_reference['image_index_sha256']}"
        )
        print(
            f"VALIDATION_GROUND_TRUTH="
            f"{validation_reference['ground_truth']}"
        )
        print(
            "VALIDATION_GROUND_TRUTH_SHA256="
            f"{validation_reference['ground_truth_sha256']}"
        )
        print(
            f"AGGREGATE_CSV={aggregate_path}"
        )
        print(
            "AGGREGATE_CSV_SHA256="
            + sha256_file(aggregate_path)
        )
        print(
            f"COMBINED_PER_CLASS_CSV="
            f"{combined_per_class_path}"
        )
        print(
            "COMBINED_PER_CLASS_CSV_SHA256="
            + sha256_file(combined_per_class_path)
        )
        print(
            f"EXECUTION_MANIFEST="
            f"{execution_manifest_path}"
        )
        print(
            "EXECUTION_MANIFEST_SHA256="
            + sha256_file(execution_manifest_path)
        )
        print(
            f"GLOBAL_ARTIFACT_MANIFEST="
            f"{global_manifest_path}"
        )
        print(
            f"GLOBAL_ARTIFACT_MANIFEST_SHA256="
            f"{global_manifest_sha}"
        )
        print(
            f"GLOBAL_ARTIFACT_FILE_COUNT="
            f"{global_file_count}"
        )
        print("SAVE_TXT=TRUE")
        print("SAVE_CONF=TRUE")
        print("PREDICTION_FLOOR=0.001")
        print("OFFLINE_DIAGNOSTICS_EXECUTED=FALSE")
        print("VALIDATION_INFERENCE_COMPLETED=TRUE")
        print("TRAINING_STARTED=FALSE")
        print("TEST_ACCESS=NONE")
        print("NEW_GPU_TRAINING_AUTHORIZED=FALSE")
        print("A12_D2_EXECUTION=PASS")
        print(
            "NEXT_ACTION="
            "PRESERVE_AND_REVIEW_A12_D2_OUTPUTS_BEFORE_OFFLINE_DIAGNOSTICS"
        )
        print("=" * 110)

        return 0

    except Exception as exc:
        failure_path = (
            args.output_root
            / "A12_D2_FAILURE_MANIFEST.json"
        )
        failure_payload = {
            "schema_version":
                "A12-D2-failure-manifest-v1.0",
            "status": "FAILED",
            "repository": repository,
            "authorization": authorization,
            "completed_experiment_ids": completed_ids,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_started": False,
            "test_access": "NONE",
        }
        write_json(
            failure_path,
            failure_payload,
        )
        print(
            f"A12_D2_FAILURE_MANIFEST={failure_path}"
        )
        print(
            "A12_D2_FAILURE_MANIFEST_SHA256="
            + sha256_file(failure_path)
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
