from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping, Sequence
import math

CONFIDENCE_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
PRIMARY_MATCH_IOU = 0.50
STRICT_LOCALIZATION_IOU = 0.75
LOCALIZATION_FLOOR_IOU = 0.10
SIZE_SMALL_MAX = 0.01
SIZE_MEDIUM_MAX = 0.05
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000

TAXONOMY_PRIORITY: tuple[str, ...] = (
    "duplicate",
    "class_confusion",
    "localization",
    "background",
    "other_overlap",
)


class DiagnosticContractError(RuntimeError):
    """Raised when frozen D3 input or matching invariants are violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticContractError(message)


def _finite01(value: float, label: str, *, positive: bool = False) -> float:
    value = float(value)
    require(math.isfinite(value), f"{label} must be finite")
    if positive:
        require(0.0 < value <= 1.0, f"{label} must be in (0, 1]")
    else:
        require(0.0 <= value <= 1.0, f"{label} must be in [0, 1]")
    return value


def size_bin(area_norm: float) -> str:
    area = float(area_norm)
    require(math.isfinite(area) and area >= 0.0, "area_norm must be finite and non-negative")
    if area < SIZE_SMALL_MAX:
        return "small"
    if area < SIZE_MEDIUM_MAX:
        return "medium"
    return "large"


@dataclass(frozen=True, slots=True)
class GroundTruth:
    filestem: str
    patient_id: str
    box_index: int
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    area_norm: float
    size_bin: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "GroundTruth":
        width = _finite01(float(row["width"]), "GT width", positive=True)
        height = _finite01(float(row["height"]), "GT height", positive=True)
        area = float(row["area_norm"])
        require(abs(area - width * height) <= 1e-8, "GT area_norm does not equal width*height")
        frozen_size = str(row["size_bin"])
        require(frozen_size == size_bin(area), "GT size_bin disagrees with frozen D3 boundaries")
        return cls(
            filestem=str(row["filestem"]),
            patient_id=str(row["patient_id"]),
            box_index=int(row["box_index"]),
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            x_center=_finite01(float(row["x_center"]), "GT x_center"),
            y_center=_finite01(float(row["y_center"]), "GT y_center"),
            width=width,
            height=height,
            area_norm=area,
            size_bin=frozen_size,
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    filestem: str
    prediction_index: int
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    area_norm: float
    confidence: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "Prediction":
        width = _finite01(float(row["width"]), "prediction width")
        height = _finite01(float(row["height"]), "prediction height")
        area = float(row["area_norm"])
        require(math.isfinite(area) and area >= 0.0, "prediction area_norm must be finite and non-negative")
        require(abs(area - width * height) <= 1e-8, "prediction area_norm does not equal width*height")
        confidence = _finite01(float(row["confidence"]), "prediction confidence")
        require(confidence + 1e-12 >= 0.001, "prediction confidence below frozen D2 export floor")
        return cls(
            filestem=str(row["filestem"]),
            prediction_index=int(row["prediction_index"]),
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            x_center=_finite01(float(row["x_center"]), "prediction x_center"),
            y_center=_finite01(float(row["y_center"]), "prediction y_center"),
            width=width,
            height=height,
            area_norm=area,
            confidence=confidence,
        )

    @property
    def size_bin(self) -> str:
        return size_bin(self.area_norm)


def _xyxy(box: GroundTruth | Prediction) -> tuple[float, float, float, float]:
    half_w = box.width / 2.0
    half_h = box.height / 2.0
    return (
        box.x_center - half_w,
        box.y_center - half_h,
        box.x_center + half_w,
        box.y_center + half_h,
    )


def iou_xywh(a: GroundTruth | Prediction, b: GroundTruth | Prediction) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _best_gt(
    prediction: Prediction,
    gt_rows: Sequence[GroundTruth],
    candidate_indices: Iterable[int],
) -> tuple[int | None, float]:
    scored: list[tuple[float, int, int]] = []
    for idx in candidate_indices:
        gt = gt_rows[idx]
        scored.append((iou_xywh(prediction, gt), gt.box_index, idx))
    if not scored:
        return None, 0.0
    # Highest IoU first; deterministic tie-break by frozen GT box_index, then list index.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_iou, _box_index, best_idx = scored[0]
    return best_idx, best_iou


def _prediction_output_sort_key(prediction: Prediction) -> tuple[int, int, str, float]:
    # Output ordering only. Primary matching is governed by globally sorted same-class IoU pairs.
    return (prediction.prediction_index, prediction.class_id, prediction.class_name, -prediction.confidence)


def match_image(
    gt_rows: Sequence[GroundTruth],
    prediction_rows: Sequence[Prediction],
    confidence_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply the frozen D3 one-to-one matcher and unmatched-prediction taxonomy to one image.

    Returns (prediction_events, gt_events). Prediction events contain one row for every
    prediction at/above the threshold. GT events contain one row for every operational GT.
    """
    threshold = float(confidence_threshold)
    require(threshold in CONFIDENCE_THRESHOLDS, f"confidence threshold is not frozen: {threshold}")

    filestems = {row.filestem for row in gt_rows} | {row.filestem for row in prediction_rows}
    require(len(filestems) <= 1, "match_image received records from multiple images")
    filestem = next(iter(filestems), "")

    active_predictions = sorted(
        (row for row in prediction_rows if row.confidence + 1e-12 >= threshold),
        key=_prediction_output_sort_key,
    )

    unmatched_gt: set[int] = set(range(len(gt_rows)))
    matched_gt: set[int] = set()
    matched_prediction_positions: set[int] = set()
    matched_prediction_to_gt: dict[int, tuple[int, float]] = {}

    # Frozen repository protocol: form all same-class prediction/GT IoU pairs,
    # sort the candidate pairs by descending IoU, then greedily assign one-to-one
    # matches at IoU >= 0.50. Equality tie-breaks are deterministic artifact-order
    # details only: prediction_index, GT box_index, then stable list positions.
    candidate_pairs: list[tuple[float, int, int, int, int]] = []
    for pred_pos, prediction in enumerate(active_predictions):
        for gt_idx, gt in enumerate(gt_rows):
            if gt.class_id != prediction.class_id:
                continue
            pair_iou = iou_xywh(prediction, gt)
            if pair_iou + 1e-12 < PRIMARY_MATCH_IOU:
                continue
            candidate_pairs.append(
                (
                    pair_iou,
                    prediction.prediction_index,
                    gt.box_index,
                    pred_pos,
                    gt_idx,
                )
            )

    candidate_pairs.sort(
        key=lambda item: (-item[0], item[1], item[2], item[3], item[4])
    )

    for pair_iou, _prediction_index, _gt_box_index, pred_pos, gt_idx in candidate_pairs:
        if pred_pos in matched_prediction_positions or gt_idx not in unmatched_gt:
            continue
        matched_prediction_to_gt[pred_pos] = (gt_idx, pair_iou)
        matched_prediction_positions.add(pred_pos)
        unmatched_gt.remove(gt_idx)
        matched_gt.add(gt_idx)

    prediction_events: list[dict[str, object]] = []
    for pred_pos, prediction in enumerate(active_predictions):
        if pred_pos in matched_prediction_to_gt:
            gt_idx, match_iou = matched_prediction_to_gt[pred_pos]
            gt = gt_rows[gt_idx]
            prediction_events.append(
                {
                    "filestem": filestem,
                    "prediction_index": prediction.prediction_index,
                    "predicted_class_id": prediction.class_id,
                    "predicted_class_name": prediction.class_name,
                    "confidence": prediction.confidence,
                    "prediction_area_norm": prediction.area_norm,
                    "prediction_size_bin": prediction.size_bin,
                    "event_type": "tp",
                    "match_iou": match_iou,
                    "strict_iou75": match_iou >= STRICT_LOCALIZATION_IOU,
                    "reference_gt_box_index": gt.box_index,
                    "reference_gt_class_id": gt.class_id,
                    "reference_gt_class_name": gt.class_name,
                    "reference_gt_area_norm": gt.area_norm,
                    "reference_gt_size_bin": gt.size_bin,
                    "max_iou_any": match_iou,
                    "max_iou_same_class": match_iou,
                    "max_iou_different_class": 0.0,
                }
            )
            continue

        all_indices = list(range(len(gt_rows)))
        same_indices = [i for i, gt in enumerate(gt_rows) if gt.class_id == prediction.class_id]
        diff_indices = [i for i, gt in enumerate(gt_rows) if gt.class_id != prediction.class_id]
        matched_same_indices = [i for i in matched_gt if gt_rows[i].class_id == prediction.class_id]

        any_idx, max_any = _best_gt(prediction, gt_rows, all_indices)
        same_idx, max_same = _best_gt(prediction, gt_rows, same_indices)
        diff_idx, max_diff = _best_gt(prediction, gt_rows, diff_indices)
        duplicate_idx, max_duplicate = _best_gt(prediction, gt_rows, matched_same_indices)

        reference_idx: int | None
        if duplicate_idx is not None and max_duplicate >= PRIMARY_MATCH_IOU:
            event_type = "duplicate"
            reference_idx = duplicate_idx
        elif diff_idx is not None and max_diff >= PRIMARY_MATCH_IOU:
            event_type = "class_confusion"
            reference_idx = diff_idx
        elif (
            same_idx is not None
            and max_same >= LOCALIZATION_FLOOR_IOU
            and max_same < PRIMARY_MATCH_IOU
        ):
            event_type = "localization"
            reference_idx = same_idx
        elif max_any < LOCALIZATION_FLOOR_IOU:
            event_type = "background"
            reference_idx = None
        else:
            event_type = "other_overlap"
            reference_idx = any_idx

        if reference_idx is None:
            ref_gt = None
            ref_iou = 0.0
        else:
            ref_gt = gt_rows[reference_idx]
            ref_iou = iou_xywh(prediction, ref_gt)

        prediction_events.append(
            {
                "filestem": filestem,
                "prediction_index": prediction.prediction_index,
                "predicted_class_id": prediction.class_id,
                "predicted_class_name": prediction.class_name,
                "confidence": prediction.confidence,
                "prediction_area_norm": prediction.area_norm,
                "prediction_size_bin": prediction.size_bin,
                "event_type": event_type,
                "match_iou": "",
                "strict_iou75": False,
                "reference_gt_box_index": "" if ref_gt is None else ref_gt.box_index,
                "reference_gt_class_id": "" if ref_gt is None else ref_gt.class_id,
                "reference_gt_class_name": "" if ref_gt is None else ref_gt.class_name,
                "reference_gt_area_norm": "" if ref_gt is None else ref_gt.area_norm,
                "reference_gt_size_bin": "" if ref_gt is None else ref_gt.size_bin,
                "reference_iou": ref_iou,
                "max_iou_any": max_any,
                "max_iou_same_class": max_same,
                "max_iou_different_class": max_diff,
            }
        )

    gt_events: list[dict[str, object]] = []
    gt_to_prediction: dict[int, tuple[Prediction, float]] = {}
    for pred_pos, (gt_idx, match_iou) in matched_prediction_to_gt.items():
        gt_to_prediction[gt_idx] = (active_predictions[pred_pos], match_iou)

    for gt_idx, gt in enumerate(gt_rows):
        if gt_idx in gt_to_prediction:
            prediction, match_iou = gt_to_prediction[gt_idx]
            gt_events.append(
                {
                    "filestem": filestem,
                    "patient_id": gt.patient_id,
                    "box_index": gt.box_index,
                    "class_id": gt.class_id,
                    "class_name": gt.class_name,
                    "area_norm": gt.area_norm,
                    "size_bin": gt.size_bin,
                    "event_type": "tp",
                    "matched_prediction_index": prediction.prediction_index,
                    "matched_confidence": prediction.confidence,
                    "match_iou": match_iou,
                    "strict_iou75": match_iou >= STRICT_LOCALIZATION_IOU,
                }
            )
        else:
            gt_events.append(
                {
                    "filestem": filestem,
                    "patient_id": gt.patient_id,
                    "box_index": gt.box_index,
                    "class_id": gt.class_id,
                    "class_name": gt.class_name,
                    "area_norm": gt.area_norm,
                    "size_bin": gt.size_bin,
                    "event_type": "fn",
                    "matched_prediction_index": "",
                    "matched_confidence": "",
                    "match_iou": "",
                    "strict_iou75": False,
                }
            )

    require(
        len([e for e in prediction_events if e["event_type"] == "tp"])
        == len([e for e in gt_events if e["event_type"] == "tp"]),
        "TP conservation failure between prediction and GT events",
    )
    require(len(gt_events) == len(gt_rows), "GT event conservation failure")
    require(len(prediction_events) == len(active_predictions), "prediction event conservation failure")
    return prediction_events, gt_events


def analyze_dataset(
    gt_rows: Sequence[GroundTruth],
    prediction_rows: Sequence[Prediction],
    operational_image_to_patient: Mapping[str, str],
    confidence_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run frozen object matching over an already-loaded operational-readable dataset."""
    gt_by_image: dict[str, list[GroundTruth]] = defaultdict(list)
    pred_by_image: dict[str, list[Prediction]] = defaultdict(list)

    operational = dict(operational_image_to_patient)
    require(len(operational) == len(set(operational)), "duplicate operational image mapping")

    for gt in gt_rows:
        require(gt.filestem in operational, f"GT references non-operational image: {gt.filestem}")
        require(
            gt.patient_id == operational[gt.filestem],
            f"GT patient_id mismatch: {gt.filestem}",
        )
        gt_by_image[gt.filestem].append(gt)

    for prediction in prediction_rows:
        require(
            prediction.filestem in operational,
            f"prediction references non-operational image: {prediction.filestem}",
        )
        pred_by_image[prediction.filestem].append(prediction)

    prediction_events: list[dict[str, object]] = []
    gt_events: list[dict[str, object]] = []

    for filestem in sorted(operational):
        pred_events, image_gt_events = match_image(
            gt_by_image.get(filestem, []),
            pred_by_image.get(filestem, []),
            confidence_threshold,
        )
        patient_id = operational[filestem]
        for event in pred_events:
            event["patient_id"] = patient_id
        prediction_events.extend(pred_events)
        gt_events.extend(image_gt_events)

    require(len(gt_events) == len(gt_rows), "dataset GT event conservation failure")
    return prediction_events, gt_events


def precision_recall_f1(
    prediction_events: Sequence[Mapping[str, object]],
    gt_events: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    tp = sum(1 for event in gt_events if event["event_type"] == "tp")
    fn = sum(1 for event in gt_events if event["event_type"] == "fn")
    fp = sum(1 for event in prediction_events if event["event_type"] != "tp")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def taxonomy_counts(prediction_events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {name: 0 for name in TAXONOMY_PRIORITY}
    for event in prediction_events:
        event_type = str(event["event_type"])
        if event_type in counts:
            counts[event_type] += 1
    return counts
