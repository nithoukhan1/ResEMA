from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np

import A12_D3_OFFLINE_DIAGNOSTIC_ENGINE as engine

D2_ARCHIVE_BYTES = 43_856_067
D2_ARCHIVE_SHA256 = "419ac71b1e42b791168a5ea24cf87d71b8a3d1eba9333d183a73009733056b93"
D2_ZIP_MEMBER_COUNT = 18_410
D2_PREFIX = "A12_D2_STANDARDIZED_VALIDATION_791566CB/"
KNOWN_UNREADABLE = "1502_0635264266_05_WRI-R2_M015"
EXPECTED_FROZEN_IMAGES = 3_050
EXPECTED_OPERATIONAL_IMAGES = 3_049
EXPECTED_PATIENTS = 914
EXPECTED_FROZEN_GT = 7_113
EXPECTED_OPERATIONAL_GT = 7_110

CLASS_NAMES: tuple[str, ...] = (
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

BASELINE_ID = "BASE-B-ORG-PT-S42"
MODEL_SPECS: tuple[dict[str, object], ...] = (
    {
        "experiment_id": BASELINE_ID,
        "display_name": "YOLO11s baseline",
        "prediction_rows": 40_236,
        "prediction_sha256": "303df9b14c08c69973af8c78c21cb69650da2cf2d7c23db6b1936af086697a56",
    },
    {
        "experiment_id": "BORG-PT-S42-SCCONV-EARLY-E100",
        "display_name": "SCConv-Early",
        "prediction_rows": 40_830,
        "prediction_sha256": "35c7de5b4dad7817fd60699aec8a269b14879e2c57d284e9b13e6752c9184269",
    },
    {
        "experiment_id": "BORG-PT-S42-SCCONV-4STAGE-E100",
        "display_name": "SCConv-4Stage",
        "prediction_rows": 40_183,
        "prediction_sha256": "6df32649272bc968df64bcba23db56dcb13afd92aba1d1e294e8aa98498f82d1",
    },
    {
        "experiment_id": "BORG-PT-S42-DYSAMPLE-E100",
        "display_name": "DySample",
        "prediction_rows": 37_209,
        "prediction_sha256": "ed07ecfbdf46e6fca0d33dc5893410b80d305d63a79a749bdf8cbc6b27d840a9",
    },
    {
        "experiment_id": "BORG-PT-S42-CANONICAL-EMA-E100",
        "display_name": "Canonical EMA",
        "prediction_rows": 40_357,
        "prediction_sha256": "e69f1b521028acf98300bdadf1b0a1bd2853d35cbad165515b7ca6a0a954a44f",
    },
    {
        "experiment_id": "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100",
        "display_name": "SCConv-Early + Canonical EMA",
        "prediction_rows": 37_917,
        "prediction_sha256": "06d8f5497ae8cffb669b436989ef9087bc8edb99e8dda2ddaef895bb15d9fd8b",
    },
)

EARLY_ID = "BORG-PT-S42-SCCONV-EARLY-E100"
STAGE4_ID = "BORG-PT-S42-SCCONV-4STAGE-E100"
DYSAMPLE_ID = "BORG-PT-S42-DYSAMPLE-E100"
EMA_ID = "BORG-PT-S42-CANONICAL-EMA-E100"
COMB_ID = "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100"

# Frozen Chat12 minimum comparison lattice. Direction is comparison minus reference.
COMPARISON_SPECS: tuple[tuple[str, str], ...] = (
    (BASELINE_ID, EARLY_ID),
    (BASELINE_ID, STAGE4_ID),
    (BASELINE_ID, DYSAMPLE_ID),
    (BASELINE_ID, EMA_ID),
    (BASELINE_ID, COMB_ID),
    (EARLY_ID, EMA_ID),
    (EARLY_ID, STAGE4_ID),
    (EARLY_ID, COMB_ID),
)


CRITICAL_MEMBER_SHA256: dict[str, str] = {
    D2_PREFIX + "A12_D2_EXECUTION_MANIFEST.json": "4c19fc8a478ff12aeee93c1f7f8fb786f69ffabe35798741333ac5ac5b3d1bd8",
    D2_PREFIX + "A12_D2_ARTIFACT_MANIFEST.csv": "c9754d034b44f24ccc10ac86720225d143fbdc2f9e1b08ac4f7e674dfda36fab",
    D2_PREFIX + "STANDARDIZED_VALIDATION_AGGREGATE.csv": "89064830001ec4a292eff4c5fe3171e37a4c01cf5af0e3996f61cf40d93f3306",
    D2_PREFIX + "STANDARDIZED_VALIDATION_PER_CLASS.csv": "ee2a8ec8c82b72b8f44f7f66c68b9484318f31bbd6a3da74678957d0df49884e",
    D2_PREFIX + "VALIDATION_IMAGE_INDEX.csv": "f7b414c700f8f523184ff669b6fc8467ee71a176adcf07fdbbd936c240c70954",
    D2_PREFIX + "VALIDATION_GROUND_TRUTH.csv": "537ca772ca84284f038b128a62c6a0b7203eac8f1360f7245fdad344ccb8ca13",
    "PRESERVATION/A12_D2_PRESERVATION_REVIEW.json": "9805009b0fb757bcd3e84ef7ec28b471cced7da0b9a700d3ad56cac6932a84bd",
}

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_CHUNK = 128

PREDICTION_EVENT_COLUMNS: tuple[str, ...] = (
    "experiment_id", "display_name", "confidence_threshold", "filestem", "patient_id",
    "prediction_index", "predicted_class_id", "predicted_class_name", "confidence",
    "prediction_area_norm", "prediction_size_bin", "event_type", "match_iou",
    "strict_iou75", "reference_gt_box_index", "reference_gt_class_id",
    "reference_gt_class_name", "reference_gt_area_norm", "reference_gt_size_bin",
    "reference_iou", "max_iou_any", "max_iou_same_class", "max_iou_different_class",
)

GT_EVENT_COLUMNS: tuple[str, ...] = (
    "experiment_id", "display_name", "confidence_threshold", "filestem", "patient_id",
    "box_index", "class_id", "class_name", "area_norm", "size_bin", "event_type",
    "matched_prediction_index", "matched_confidence", "match_iou", "strict_iou75",
)


class RunnerContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_member_bytes(zf: zipfile.ZipFile, member: str) -> bytes:
    try:
        with zf.open(member, "r") as f:
            return f.read()
    except KeyError as exc:
        raise RunnerContractError(f"missing ZIP member: {member}") from exc


def zip_member_sha256(zf: zipfile.ZipFile, member: str) -> str:
    return sha256_bytes(zip_member_bytes(zf, member))


def csv_rows_from_bytes(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def require_columns(rows: Sequence[Mapping[str, object]], required: Iterable[str], label: str) -> None:
    require(bool(rows), f"{label} is empty")
    missing = [name for name in required if name not in rows[0]]
    require(not missing, f"{label} missing columns: {missing}")


def validate_class_pair(class_id: int, class_name: str, label: str) -> None:
    require(0 <= class_id < len(CLASS_NAMES), f"{label} class_id outside 0..8: {class_id}")
    require(CLASS_NAMES[class_id] == class_name, f"{label} class-name mismatch for id {class_id}: {class_name}")


def validate_prediction_identity(rows: Sequence[engine.Prediction], experiment_id: str) -> None:
    seen: set[tuple[str, int]] = set()
    for row in rows:
        validate_class_pair(row.class_id, row.class_name, experiment_id)
        key = (row.filestem, row.prediction_index)
        require(key not in seen, f"duplicate prediction identity in {experiment_id}: {key}")
        seen.add(key)


def load_frozen_d2_inputs(archive_path: Path) -> tuple[dict[str, str], list[engine.GroundTruth], dict[str, list[engine.Prediction]], dict[str, object]]:
    archive_path = archive_path.resolve()
    require(archive_path.is_file(), f"D2 archive not found: {archive_path}")
    require(archive_path.stat().st_size == D2_ARCHIVE_BYTES, "D2 archive byte-size drift")
    require(sha256_file(archive_path) == D2_ARCHIVE_SHA256, "D2 archive SHA256 drift")

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        require(len(names) == D2_ZIP_MEMBER_COUNT, "D2 ZIP member-count drift")
        require(len(names) == len(set(names)), "duplicate ZIP member names detected")
        bad_member = zf.testzip()
        require(bad_member is None, f"ZIP CRC failure: {bad_member}")

        for member, expected_sha in CRITICAL_MEMBER_SHA256.items():
            require(zip_member_sha256(zf, member) == expected_sha, f"critical member SHA drift: {member}")

        image_rows = csv_rows_from_bytes(zip_member_bytes(zf, D2_PREFIX + "VALIDATION_IMAGE_INDEX.csv"))
        gt_rows_raw = csv_rows_from_bytes(zip_member_bytes(zf, D2_PREFIX + "VALIDATION_GROUND_TRUTH.csv"))
        require(len(image_rows) == EXPECTED_FROZEN_IMAGES, "frozen image-index row-count drift")
        require(len(gt_rows_raw) == EXPECTED_FROZEN_GT, "frozen GT row-count drift")

        require_columns(
            image_rows,
            ("filestem", "patient_id", "operational_readable", "known_unreadable", "ground_truth_box_count"),
            "validation image index",
        )
        require_columns(
            gt_rows_raw,
            ("filestem", "patient_id", "box_index", "class_id", "class_name", "x_center", "y_center", "width", "height", "area_norm", "size_bin"),
            "validation ground truth",
        )

        operational_map: dict[str, str] = {}
        unreadable_rows: list[dict[str, str]] = []
        all_patients: set[str] = set()
        for row in image_rows:
            filestem = row["filestem"]
            patient_id = row["patient_id"]
            require(filestem not in operational_map or not bool_text(row["operational_readable"]), f"duplicate operational filestem: {filestem}")
            all_patients.add(patient_id)
            if bool_text(row["known_unreadable"]):
                unreadable_rows.append(row)
            if bool_text(row["operational_readable"]):
                require(filestem not in operational_map, f"duplicate operational filestem: {filestem}")
                operational_map[filestem] = patient_id

        require(len(operational_map) == EXPECTED_OPERATIONAL_IMAGES, "operational image-count drift")
        require(len(all_patients) == EXPECTED_PATIENTS, "validation patient-count drift")
        require(len(unreadable_rows) == 1, "expected exactly one known unreadable image")
        require(unreadable_rows[0]["filestem"] == KNOWN_UNREADABLE, "known unreadable identity drift")
        require(int(unreadable_rows[0]["ground_truth_box_count"]) == 3, "known unreadable GT-count drift")

        operational_gt: list[engine.GroundTruth] = []
        unreadable_gt = 0
        for row in gt_rows_raw:
            class_id = int(row["class_id"])
            class_name = row["class_name"]
            validate_class_pair(class_id, class_name, "GT")
            if row["filestem"] == KNOWN_UNREADABLE:
                unreadable_gt += 1
                continue
            require(row["filestem"] in operational_map, f"GT references non-operational image: {row['filestem']}")
            require(row["patient_id"] == operational_map[row["filestem"]], f"GT patient mismatch: {row['filestem']}")
            operational_gt.append(engine.GroundTruth.from_mapping(row))

        require(unreadable_gt == 3, "unreadable GT-count drift")
        require(len(operational_gt) == EXPECTED_OPERATIONAL_GT, "operational GT-count drift")

        predictions: dict[str, list[engine.Prediction]] = {}
        for spec in MODEL_SPECS:
            experiment_id = str(spec["experiment_id"])
            member = D2_PREFIX + experiment_id + "/PREDICTIONS.csv"
            raw = zip_member_bytes(zf, member)
            require(sha256_bytes(raw) == spec["prediction_sha256"], f"prediction SHA drift: {experiment_id}")
            rows = csv_rows_from_bytes(raw)
            require(len(rows) == int(spec["prediction_rows"]), f"prediction row-count drift: {experiment_id}")
            require_columns(
                rows,
                ("filestem", "prediction_index", "class_id", "class_name", "x_center", "y_center", "width", "height", "area_norm", "confidence"),
                experiment_id + " predictions",
            )
            parsed = [engine.Prediction.from_mapping(row) for row in rows]
            validate_prediction_identity(parsed, experiment_id)
            require(all(row.filestem in operational_map for row in parsed), f"non-operational prediction filestem: {experiment_id}")
            predictions[experiment_id] = parsed

        preservation = json.loads(zip_member_bytes(zf, "PRESERVATION/A12_D2_PRESERVATION_REVIEW.json").decode("utf-8"))
        require(preservation["status"] == "PASS", "preservation review not PASS")
        require(preservation["offline_diagnostics_executed"] is False, "D3 already marked executed in preservation review")
        require(preservation["training_started"] is False, "unexpected training state")
        require(preservation["test_access"] == "NONE", "test firewall drift")

    metadata = {
        "archive_path": str(archive_path),
        "archive_bytes": D2_ARCHIVE_BYTES,
        "archive_sha256": D2_ARCHIVE_SHA256,
        "frozen_images": EXPECTED_FROZEN_IMAGES,
        "operational_images": EXPECTED_OPERATIONAL_IMAGES,
        "patients": EXPECTED_PATIENTS,
        "frozen_gt": EXPECTED_FROZEN_GT,
        "operational_gt": EXPECTED_OPERATIONAL_GT,
        "test_access": "NONE",
        "training_started": False,
        "validation_rerun": False,
        "checkpoint_loading": False,
    }
    return operational_map, operational_gt, predictions, metadata


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _f1_from_counts(tp: int | float, fp: int | float, fn: int | float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f1)


def _tp_ious(prediction_events: Sequence[Mapping[str, object]]) -> list[float]:
    return [float(row["match_iou"]) for row in prediction_events if row["event_type"] == "tp"]


def global_summary(experiment_id: str, display_name: str, threshold: float, prediction_events: Sequence[Mapping[str, object]], gt_events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metrics = engine.precision_recall_f1(prediction_events, gt_events)
    taxonomy = engine.taxonomy_counts(prediction_events)
    ious = _tp_ious(prediction_events)
    tp75 = sum(1 for row in gt_events if row["event_type"] == "tp" and bool(row["strict_iou75"]))
    tp = int(metrics["tp"])
    return {
        "experiment_id": experiment_id,
        "display_name": display_name,
        "confidence_threshold": threshold,
        "tp": tp,
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp_iou75": tp75,
        "tp_iou75_rate": _safe_ratio(tp75, tp),
        "mean_tp_iou": statistics.fmean(ious) if ious else None,
        "median_tp_iou": statistics.median(ious) if ious else None,
        "duplicate_fp": taxonomy["duplicate"],
        "class_confusion_fp": taxonomy["class_confusion"],
        "localization_fp": taxonomy["localization"],
        "background_fp": taxonomy["background"],
        "other_overlap_fp": taxonomy["other_overlap"],
        "active_predictions": len(prediction_events),
        "gt_support": len(gt_events),
    }


def per_class_summary(experiment_id: str, display_name: str, threshold: float, prediction_events: Sequence[Mapping[str, object]], gt_events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        gt_class = [row for row in gt_events if int(row["class_id"]) == class_id]
        pred_class = [row for row in prediction_events if int(row["predicted_class_id"]) == class_id]
        support = len(gt_class)
        tp = sum(1 for row in gt_class if row["event_type"] == "tp")
        fn = support - tp
        fp = sum(1 for row in pred_class if row["event_type"] != "tp")
        tp75 = sum(1 for row in gt_class if row["event_type"] == "tp" and bool(row["strict_iou75"]))
        if support == 0:
            status = "NO_VALIDATION_SUPPORT"
            precision = recall = f1 = None
            tp75_rate = None
        else:
            status = "SUPPORTED"
            precision, recall, f1 = _f1_from_counts(tp, fp, fn)
            tp75_rate = _safe_ratio(tp75, tp)
        out.append({
            "experiment_id": experiment_id,
            "display_name": display_name,
            "confidence_threshold": threshold,
            "class_id": class_id,
            "class_name": class_name,
            "status": status,
            "gt_support": support,
            "prediction_count": len(pred_class),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_iou75": tp75,
            "tp_iou75_rate": tp75_rate,
            "duplicate_fp": sum(1 for row in pred_class if row["event_type"] == "duplicate"),
            "class_confusion_fp": sum(1 for row in pred_class if row["event_type"] == "class_confusion"),
            "localization_fp": sum(1 for row in pred_class if row["event_type"] == "localization"),
            "background_fp": sum(1 for row in pred_class if row["event_type"] == "background"),
            "other_overlap_fp": sum(1 for row in pred_class if row["event_type"] == "other_overlap"),
        })
    return out


def gt_size_summary(experiment_id: str, display_name: str, threshold: float, gt_events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for size in ("small", "medium", "large"):
        rows = [row for row in gt_events if row["size_bin"] == size]
        support = len(rows)
        tp = sum(1 for row in rows if row["event_type"] == "tp")
        fn = support - tp
        tp75 = sum(1 for row in rows if row["event_type"] == "tp" and bool(row["strict_iou75"]))
        out.append({
            "experiment_id": experiment_id,
            "display_name": display_name,
            "confidence_threshold": threshold,
            "gt_size_bin": size,
            "gt_support": support,
            "tp": tp,
            "fn": fn,
            "recall": _safe_ratio(tp, support),
            "tp_iou75": tp75,
            "tp_iou75_rate": _safe_ratio(tp75, tp),
        })
    return out


def fp_taxonomy_summary(experiment_id: str, display_name: str, threshold: float, prediction_events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fp_rows = [row for row in prediction_events if row["event_type"] != "tp"]
    total_fp = len(fp_rows)
    out: list[dict[str, object]] = []
    for error_type in engine.TAXONOMY_PRIORITY:
        rows = [row for row in fp_rows if row["event_type"] == error_type]
        out.append({
            "experiment_id": experiment_id,
            "display_name": display_name,
            "confidence_threshold": threshold,
            "error_type": error_type,
            "count": len(rows),
            "fraction_of_fp": _safe_ratio(len(rows), total_fp),
        })
    require(sum(int(row["count"]) for row in out) == total_fp, "FP taxonomy conservation failure")
    return out


def fp_prediction_size_summary(experiment_id: str, display_name: str, threshold: float, prediction_events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for size in ("small", "medium", "large"):
        rows = [row for row in prediction_events if row["event_type"] != "tp" and row["prediction_size_bin"] == size]
        record: dict[str, object] = {
            "experiment_id": experiment_id,
            "display_name": display_name,
            "confidence_threshold": threshold,
            "prediction_size_bin": size,
            "fp_count": len(rows),
        }
        for error_type in engine.TAXONOMY_PRIORITY:
            record[error_type + "_fp"] = sum(1 for row in rows if row["event_type"] == error_type)
        out.append(record)
    return out


def patient_metrics(experiment_id: str, display_name: str, threshold: float, prediction_events: Sequence[Mapping[str, object]], gt_events: Sequence[Mapping[str, object]], patient_ids: Sequence[str]) -> list[dict[str, object]]:
    pred_by_patient: dict[str, list[Mapping[str, object]]] = {pid: [] for pid in patient_ids}
    gt_by_patient: dict[str, list[Mapping[str, object]]] = {pid: [] for pid in patient_ids}
    for row in prediction_events:
        pid = str(row["patient_id"])
        require(pid in pred_by_patient, f"prediction event has unknown patient: {pid}")
        pred_by_patient[pid].append(row)
    for row in gt_events:
        pid = str(row["patient_id"])
        require(pid in gt_by_patient, f"GT event has unknown patient: {pid}")
        gt_by_patient[pid].append(row)

    out: list[dict[str, object]] = []
    for pid in patient_ids:
        pred_rows = pred_by_patient[pid]
        gt_rows = gt_by_patient[pid]
        tp = sum(1 for row in gt_rows if row["event_type"] == "tp")
        fn = sum(1 for row in gt_rows if row["event_type"] == "fn")
        fp = sum(1 for row in pred_rows if row["event_type"] != "tp")
        tp75 = sum(1 for row in gt_rows if row["event_type"] == "tp" and bool(row["strict_iou75"]))
        precision, recall, f1 = _f1_from_counts(tp, fp, fn)
        out.append({
            "experiment_id": experiment_id,
            "display_name": display_name,
            "confidence_threshold": threshold,
            "patient_id": pid,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_iou75": tp75,
            "tp_iou75_rate": _safe_ratio(tp75, tp),
        })
    return out


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def confidence_summary(
    experiment_id: str,
    display_name: str,
    threshold: float,
    prediction_events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: list[tuple[str, list[Mapping[str, object]]]] = [
        ("all_predictions", list(prediction_events)),
        ("tp", [row for row in prediction_events if row["event_type"] == "tp"]),
        ("all_fp", [row for row in prediction_events if row["event_type"] != "tp"]),
    ]
    groups.extend(
        (error_type, [row for row in prediction_events if row["event_type"] == error_type])
        for error_type in engine.TAXONOMY_PRIORITY
    )

    out: list[dict[str, object]] = []
    for group, rows in groups:
        values = [float(row["confidence"]) for row in rows]
        out.append({
            "experiment_id": experiment_id,
            "display_name": display_name,
            "confidence_threshold": threshold,
            "event_group": group,
            "count": len(values),
            "confidence_min": min(values) if values else None,
            "confidence_q25": _quantile(values, 0.25),
            "confidence_median": statistics.median(values) if values else None,
            "confidence_mean": statistics.fmean(values) if values else None,
            "confidence_q75": _quantile(values, 0.75),
            "confidence_max": max(values) if values else None,
        })
    return out


def paired_patient_changes(
    patient_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    patients = sorted({str(row["patient_id"]) for row in patient_rows})
    thresholds = list(engine.CONFIDENCE_THRESHOLDS)
    display = {str(spec["experiment_id"]): str(spec["display_name"]) for spec in MODEL_SPECS}
    index = {
        (str(row["experiment_id"]), float(row["confidence_threshold"]), str(row["patient_id"])): row
        for row in patient_rows
    }
    expected = len(MODEL_SPECS) * len(thresholds) * len(patients)
    require(len(index) == expected, f"patient metric lattice incomplete: {len(index)} != {expected}")

    out: list[dict[str, object]] = []
    for reference_id, comparison_id in COMPARISON_SPECS:
        require(reference_id in display, f"unknown reference model in comparison: {reference_id}")
        require(comparison_id in display, f"unknown comparison model: {comparison_id}")
        for threshold in thresholds:
            for pid in patients:
                reference = index[(reference_id, threshold, pid)]
                comparison = index[(comparison_id, threshold, pid)]
                reference_tp = int(reference["tp"])
                reference_fp = int(reference["fp"])
                reference_fn = int(reference["fn"])
                comparison_tp = int(comparison["tp"])
                comparison_fp = int(comparison["fp"])
                comparison_fn = int(comparison["fn"])
                reference_precision, reference_recall, reference_f1 = _f1_from_counts(
                    reference_tp, reference_fp, reference_fn
                )
                comparison_precision, comparison_recall, comparison_f1 = _f1_from_counts(
                    comparison_tp, comparison_fp, comparison_fn
                )
                out.append({
                    "confidence_threshold": threshold,
                    "patient_id": pid,
                    "reference_experiment_id": reference_id,
                    "reference_display_name": display[reference_id],
                    "comparison_experiment_id": comparison_id,
                    "comparison_display_name": display[comparison_id],
                    "reference_tp": reference_tp,
                    "comparison_tp": comparison_tp,
                    "delta_tp": comparison_tp - reference_tp,
                    "reference_fp": reference_fp,
                    "comparison_fp": comparison_fp,
                    "delta_fp": comparison_fp - reference_fp,
                    "reference_fn": reference_fn,
                    "comparison_fn": comparison_fn,
                    "delta_fn": comparison_fn - reference_fn,
                    "reference_precision": reference_precision,
                    "comparison_precision": comparison_precision,
                    "delta_precision": comparison_precision - reference_precision,
                    "reference_recall": reference_recall,
                    "comparison_recall": comparison_recall,
                    "delta_recall": comparison_recall - reference_recall,
                    "reference_f1": reference_f1,
                    "comparison_f1": comparison_f1,
                    "delta_f1": comparison_f1 - reference_f1,
                    "reference_tp_iou75": int(reference["tp_iou75"]),
                    "comparison_tp_iou75": int(comparison["tp_iou75"]),
                    "delta_tp_iou75": int(comparison["tp_iou75"]) - int(reference["tp_iou75"]),
                })
    return out


def _metric_from_counts(counts: np.ndarray, metric: str) -> np.ndarray:
    tp = counts[:, 0].astype(np.float64)
    fp = counts[:, 1].astype(np.float64)
    fn = counts[:, 2].astype(np.float64)
    tp75 = counts[:, 3].astype(np.float64)
    if metric == "precision":
        den = tp + fp
        return np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    if metric == "recall":
        den = tp + fn
        return np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    if metric == "f1":
        den = 2.0 * tp + fp + fn
        return np.divide(2.0 * tp, den, out=np.zeros_like(tp), where=den > 0)
    if metric == "tp_iou75_rate":
        return np.divide(tp75, tp, out=np.zeros_like(tp), where=tp > 0)
    raise RunnerContractError(f"unknown bootstrap metric: {metric}")


def _point_metric(counts: np.ndarray, metric: str) -> float:
    return float(_metric_from_counts(counts.reshape(1, 4), metric)[0])


def paired_patient_bootstrap(
    patient_rows: Sequence[Mapping[str, object]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = BOOTSTRAP_CI,
    chunk_size: int = BOOTSTRAP_CHUNK,
) -> list[dict[str, object]]:
    require(replicates > 0, "bootstrap replicates must be positive")
    require(0.0 < ci < 1.0, "bootstrap CI must be in (0,1)")
    require(chunk_size > 0, "bootstrap chunk size must be positive")

    patients = sorted({str(row["patient_id"]) for row in patient_rows})
    thresholds = list(engine.CONFIDENCE_THRESHOLDS)
    model_ids = [str(spec["experiment_id"]) for spec in MODEL_SPECS]
    display = {str(spec["experiment_id"]): str(spec["display_name"]) for spec in MODEL_SPECS}
    require(BASELINE_ID in model_ids, "baseline model missing from frozen model list")
    for reference_id, comparison_id in COMPARISON_SPECS:
        require(reference_id in model_ids, f"unknown reference model in comparison: {reference_id}")
        require(comparison_id in model_ids, f"unknown comparison model: {comparison_id}")

    index = {(str(row["experiment_id"]), float(row["confidence_threshold"]), str(row["patient_id"])): row for row in patient_rows}
    expected = len(model_ids) * len(thresholds) * len(patients)
    require(len(index) == expected, f"patient metric lattice incomplete: {len(index)} != {expected}")

    counts: dict[tuple[str, float], np.ndarray] = {}
    for model_id in model_ids:
        for threshold in thresholds:
            arr = np.zeros((len(patients), 4), dtype=np.int64)
            for i, pid in enumerate(patients):
                row = index[(model_id, threshold, pid)]
                arr[i, :] = [int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tp_iou75"])]
            counts[(model_id, threshold)] = arr

    metrics = ("precision", "recall", "f1", "tp_iou75_rate")
    delta_samples: dict[tuple[float, str, str, str], np.ndarray] = {
        (thr, reference_id, comparison_id, metric): np.empty(replicates, dtype=np.float64)
        for thr in thresholds
        for reference_id, comparison_id in COMPARISON_SPECS
        for metric in metrics
    }

    rng = np.random.default_rng(seed)
    n = len(patients)
    start = 0
    while start < replicates:
        stop = min(start + chunk_size, replicates)
        draw = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        for threshold in thresholds:
            sampled_metric: dict[tuple[str, str], np.ndarray] = {}
            for model_id in model_ids:
                sample_sum = counts[(model_id, threshold)][draw].sum(axis=1)
                for metric in metrics:
                    sampled_metric[(model_id, metric)] = _metric_from_counts(sample_sum, metric)
            for reference_id, comparison_id in COMPARISON_SPECS:
                for metric in metrics:
                    delta_samples[(threshold, reference_id, comparison_id, metric)][start:stop] = (
                        sampled_metric[(comparison_id, metric)] - sampled_metric[(reference_id, metric)]
                    )
        start = stop

    alpha = (1.0 - ci) / 2.0
    out: list[dict[str, object]] = []
    for threshold in thresholds:
        total = {model_id: counts[(model_id, threshold)].sum(axis=0) for model_id in model_ids}
        for reference_id, comparison_id in COMPARISON_SPECS:
            for metric in metrics:
                point_delta = _point_metric(total[comparison_id], metric) - _point_metric(total[reference_id], metric)
                samples = delta_samples[(threshold, reference_id, comparison_id, metric)]
                low, high = np.quantile(samples, [alpha, 1.0 - alpha], method="linear")
                out.append({
                    "confidence_threshold": threshold,
                    "reference_experiment_id": reference_id,
                    "reference_display_name": display[reference_id],
                    "comparison_experiment_id": comparison_id,
                    "comparison_display_name": display[comparison_id],
                    "metric": metric,
                    "point_delta_comparison_minus_reference": point_delta,
                    "ci_level": ci,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_seed": seed,
                    "bootstrap_replicates": replicates,
                    "resampling_unit": "patient",
                    "paired_resamples": True,
                    "patient_count": len(patients),
                })
    return out

def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path.name}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def event_filename(experiment_id: str, threshold: float, kind: str) -> str:
    code = f"{int(round(threshold * 100)):02d}"
    return f"{experiment_id}__CONF_{code}__{kind}.csv"


def write_execution_outputs(
    output_dir: Path,
    operational_map: Mapping[str, str],
    gt_rows: Sequence[engine.GroundTruth],
    predictions_by_model: Mapping[str, Sequence[engine.Prediction]],
    input_metadata: Mapping[str, object],
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        require(output_dir.is_dir(), f"output path exists and is not a directory: {output_dir}")
        require(not any(output_dir.iterdir()), f"output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    events_dir = output_dir / "events"
    events_dir.mkdir()

    patient_ids = sorted(set(operational_map.values()))
    require(len(patient_ids) == EXPECTED_PATIENTS, "patient count drift before analysis")

    global_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []
    taxonomy_rows: list[dict[str, object]] = []
    fp_size_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []

    for spec in MODEL_SPECS:
        experiment_id = str(spec["experiment_id"])
        display_name = str(spec["display_name"])
        require(experiment_id in predictions_by_model, f"missing predictions for {experiment_id}")
        predictions = predictions_by_model[experiment_id]
        for threshold in engine.CONFIDENCE_THRESHOLDS:
            prediction_events, gt_events = engine.analyze_dataset(gt_rows, predictions, operational_map, threshold)
            for row in prediction_events:
                row["experiment_id"] = experiment_id
                row["display_name"] = display_name
                row["confidence_threshold"] = threshold
                row.setdefault("reference_iou", "")
            for row in gt_events:
                row["experiment_id"] = experiment_id
                row["display_name"] = display_name
                row["confidence_threshold"] = threshold

            write_csv(events_dir / event_filename(experiment_id, threshold, "PREDICTION_EVENTS"), prediction_events, PREDICTION_EVENT_COLUMNS)
            write_csv(events_dir / event_filename(experiment_id, threshold, "GT_EVENTS"), gt_events, GT_EVENT_COLUMNS)

            global_rows.append(global_summary(experiment_id, display_name, threshold, prediction_events, gt_events))
            class_rows.extend(per_class_summary(experiment_id, display_name, threshold, prediction_events, gt_events))
            size_rows.extend(gt_size_summary(experiment_id, display_name, threshold, gt_events))
            taxonomy_rows.extend(fp_taxonomy_summary(experiment_id, display_name, threshold, prediction_events))
            fp_size_rows.extend(fp_prediction_size_summary(experiment_id, display_name, threshold, prediction_events))
            confidence_rows.extend(confidence_summary(experiment_id, display_name, threshold, prediction_events))
            patient_rows.extend(patient_metrics(experiment_id, display_name, threshold, prediction_events, gt_events, patient_ids))

    require(len(global_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS), "global summary lattice drift")
    require(len(class_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * len(CLASS_NAMES), "per-class summary lattice drift")
    require(len(size_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * 3, "GT-size summary lattice drift")
    require(len(taxonomy_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * len(engine.TAXONOMY_PRIORITY), "taxonomy summary lattice drift")
    require(len(fp_size_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * 3, "FP-size summary lattice drift")
    require(len(confidence_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * (3 + len(engine.TAXONOMY_PRIORITY)), "confidence summary lattice drift")
    require(len(patient_rows) == len(MODEL_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * EXPECTED_PATIENTS, "patient summary lattice drift")

    paired_change_rows = paired_patient_changes(patient_rows)
    require(
        len(paired_change_rows) == len(COMPARISON_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * EXPECTED_PATIENTS,
        "paired patient-change lattice drift",
    )
    bootstrap_rows = paired_patient_bootstrap(patient_rows)
    require(
        len(bootstrap_rows) == len(COMPARISON_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * 4,
        "paired bootstrap lattice drift",
    )

    write_csv(output_dir / "A12_D3_GLOBAL_SUMMARY.csv", global_rows)
    write_csv(output_dir / "A12_D3_PER_CLASS_SUMMARY.csv", class_rows)
    write_csv(output_dir / "A12_D3_GT_SIZE_SUMMARY.csv", size_rows)
    write_csv(output_dir / "A12_D3_FP_TAXONOMY_SUMMARY.csv", taxonomy_rows)
    write_csv(output_dir / "A12_D3_FP_PREDICTION_SIZE_SUMMARY.csv", fp_size_rows)
    write_csv(output_dir / "A12_D3_CONFIDENCE_SUMMARY.csv", confidence_rows)
    write_csv(output_dir / "A12_D3_PATIENT_METRICS.csv", patient_rows)
    write_csv(output_dir / "A12_D3_PAIRED_PATIENT_CHANGES.csv", paired_change_rows)
    write_csv(output_dir / "A12_D3_PAIRED_PATIENT_BOOTSTRAP.csv", bootstrap_rows)

    output_hashes: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(output_dir).as_posix()
            output_hashes[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    manifest = {
        "schema": "A12_D3_OFFLINE_DIAGNOSTIC_EXECUTION_V1",
        "status": "EXECUTED_OFFLINE_FROM_PRESERVED_D2_OUTPUTS",
        "input": dict(input_metadata),
        "models": [dict(spec) for spec in MODEL_SPECS],
        "confidence_thresholds": list(engine.CONFIDENCE_THRESHOLDS),
        "primary_match_iou": engine.PRIMARY_MATCH_IOU,
        "strict_localization_iou": engine.STRICT_LOCALIZATION_IOU,
        "localization_floor_iou": engine.LOCALIZATION_FLOOR_IOU,
        "area_bins": {"small": "<0.01", "medium": "[0.01,0.05)", "large": ">=0.05"},
        "bootstrap": {
            "unit": "patient",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "ci": BOOTSTRAP_CI,
            "paired_resampling": True,
            "comparison_direction": "comparison_minus_reference",
            "comparisons": [
                {"reference_experiment_id": reference_id, "comparison_experiment_id": comparison_id}
                for reference_id, comparison_id in COMPARISON_SPECS
            ],
            "metrics": ["precision", "recall", "f1", "tp_iou75_rate"],
        },
        "foreignbody_rule": "NO_VALIDATION_SUPPORT_FOR_SUPPORT_DEPENDENT_METRICS",
        "test_access": "NONE",
        "checkpoint_loading": "NONE",
        "validation_rerun": "NONE",
        "training": "NONE",
        "output_files": output_hashes,
    }
    manifest_path = output_dir / "A12_D3_EXECUTION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A12-D3 offline diagnostic runner over preserved D2 CSV evidence only")
    parser.add_argument("--archive", type=Path, required=True, help="Canonical preserved A12-D2 ZIP")
    parser.add_argument("--output-dir", type=Path, required=True, help="New empty output directory")
    args = parser.parse_args(argv)

    print("=" * 92)
    print("A12-D3 OFFLINE DIAGNOSTIC RUNNER")
    print("PRESERVED CSV EVIDENCE ONLY - NO CHECKPOINT LOAD - NO INFERENCE - NO TRAINING - TEST NONE")
    print("=" * 92)
    operational_map, gt_rows, predictions, metadata = load_frozen_d2_inputs(args.archive)
    print(f"INPUT_ARCHIVE_SHA256={metadata['archive_sha256']}")
    print(f"OPERATIONAL_IMAGES={len(operational_map)}")
    print(f"PATIENTS={len(set(operational_map.values()))}")
    print(f"OPERATIONAL_GT={len(gt_rows)}")
    print(f"MODEL_COUNT={len(predictions)}")
    print("INPUT_GATE=PASS")

    manifest = write_execution_outputs(args.output_dir, operational_map, gt_rows, predictions, metadata)
    print(f"OUTPUT_DIR={args.output_dir.resolve()}")
    print(f"OUTPUT_FILE_COUNT={len(manifest['output_files']) + 1}")
    print("OFFLINE_DIAGNOSTIC_EXECUTION=COMPLETE")
    print("CHECKPOINT_LOADING=NONE")
    print("VALIDATION_RERUN=NONE")
    print("TRAINING=NONE")
    print("TEST_ACCESS=NONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunnerContractError, engine.DiagnosticContractError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
