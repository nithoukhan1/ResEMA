"""REVIEW-ONLY source candidate: compare production B-TRAIN counts/weights with R3-11 evidence.

Pure Python standard library. This module does NOT access datasets or model files,
import torch, load checkpoints, train, or authorize deployment. To use it in a
future approved trainer, pass the already computed in-memory counts/weights and
an explicit path to the immutable private R3-11 aggregate evidence file.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ACCEPTED_EVIDENCE_SHA256 = "9de034fa7df668d17effa843674c097341af9dd6461963f9e2c454c5f358b289"
FROZEN_ASSIGNMENT_SHA256 = "87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4"
ACCEPTED_DEVELOPMENT_YAML_SHA256 = "d223511100ef04d2436961b5e36a4793eabc4deb83729f20e36f53ae63917974"
CLASS_ORDER = (
    "boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
    "periostealreaction", "pronatorsign", "softtissue", "text",
)
STATE_ORDER = ("00", "10", "01", "11")
PAIRS = 6496
# Recovered from the exact SHA-verified R3-11 evidence output; no names or patients.
EXPECTED_COUNTS = (
    (6408, 49, 3, 36),
    (6478, 10, 0, 8),
    (6492, 0, 0, 4),
    (1940, 92, 374, 4090),
    (6264, 0, 2, 230),
    (5573, 218, 159, 546),
    (6130, 0, 365, 1),
    (6231, 29, 219, 17),
    (1, 14, 14, 6467),
)


class BindingMismatch(ValueError):
    """Fail-close release error; never substitute VAL or defaults."""


def frozen_weights(counts: tuple[tuple[int, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Exact D4-C5A formula in FP64 for independent runtime cross-check."""
    if len(counts) != 9 or any(len(row) != 4 for row in counts):
        raise BindingMismatch("CLASS_STATE_SHAPE")
    results = []
    for row in counts:
        if any(type(x) is not int or x < 0 for x in row) or sum(row) != PAIRS:
            raise BindingMismatch("CLASS_STATE_COUNT_INVALID")
        raw = [1 / math.sqrt(n + 1) for n in row]
        denominator = sum(raw)
        results.append(tuple(4 * x / denominator for x in raw))
    return tuple(results)


def _count_matrix(value: Any) -> tuple[tuple[int, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)
    except TypeError as ex:
        raise BindingMismatch("INVALID_COUNT_MATRIX") from ex
    if len(rows) != 9 or any(len(row) != 4 for row in rows):
        raise BindingMismatch("COUNT_SHAPE")
    if any(type(v) is not int or v < 0 for row in rows for v in row):
        raise BindingMismatch("COUNT_TYPE_OR_VALUE")
    return rows


def _weights_equal(actual: Any, expected: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        rows = tuple(tuple(float(x) for x in row) for row in actual)
    except (TypeError, ValueError, OverflowError):
        return False
    if len(rows) != 9 or any(len(row) != 4 for row in rows):
        return False
    return all(math.isfinite(a) and abs(a - b) <= tolerance for ra, rb in zip(rows, expected) for a, b in zip(ra, rb))


def verify_aggregate_record(record: dict[str, Any], runtime_counts: Any, runtime_weights: Any) -> dict[str, Any]:
    """Validate authenticated record contents *and* runtime model-derived values.

    External caller must additionally compare actual JSON bytes against the pinned
    SHA256 in verify_binding() before using this function as production evidence.
    """
    expected_fields = {
        "schema_version": "D4E_R311_BTRAIN_VISIBILITY_READ_ONLY_R1",
        "source_assignment_sha256": FROZEN_ASSIGNMENT_SHA256,
        "candidate_yaml_sha256": ACCEPTED_DEVELOPMENT_YAML_SHA256,
        "state_order": list(STATE_ORDER),
        "class_order": list(CLASS_ORDER),
        "exact_train_pairs": PAIRS,
        "train_single_images_ignored_for_visibility": 1235,
        "val_label_access": "NONE",
        "b_test_access": "NONE",
        "model_execution": "NONE",
        "checkpoint_deserialization": "NONE",
        "d4e_recipe_frozen": False,
    }
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected_fields.items()):
        raise BindingMismatch("R311_EVIDENCE_FIELDS")
    evidence_counts = _count_matrix(record.get("train_only_visibility_counts"))
    runtime_counts = _count_matrix(runtime_counts)
    if evidence_counts != EXPECTED_COUNTS or runtime_counts != EXPECTED_COUNTS:
        raise BindingMismatch("RUNTIME_VS_FROZEN_TRAIN_COUNTS")
    target_weights = frozen_weights(EXPECTED_COUNTS)
    if not _weights_equal(record.get("train_only_visibility_weights"), target_weights, tolerance=1e-11):
        raise BindingMismatch("R311_WEIGHT_FORMULA")
    if not _weights_equal(runtime_weights, target_weights):
        raise BindingMismatch("RUNTIME_WEIGHT_FORMULA")
    return {"status": "PASS", "train_pairs": PAIRS, "classes": 9,
            "assignment_sha256": FROZEN_ASSIGNMENT_SHA256,
            "visibility_evidence_sha256": ACCEPTED_EVIDENCE_SHA256}


def verify_binding(evidence_path: str | Path, runtime_counts: Any, runtime_weights: Any) -> dict[str, Any]:
    """Production-integration candidate: never read VAL images/labels, TEST, checkpoint."""
    p = Path(evidence_path)
    if any(item.is_symlink() for item in (p, *p.parents)):
        raise BindingMismatch("EVIDENCE_SYMLINK")
    if not p.is_file() or p.stat().st_size > 1024 * 1024:
        raise BindingMismatch("EVIDENCE_PATH_OR_SIZE")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACCEPTED_EVIDENCE_SHA256:
        raise BindingMismatch("EVIDENCE_SHA256")
    try:
        record = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as ex:
        raise BindingMismatch("EVIDENCE_JSON") from ex
    return verify_aggregate_record(record, runtime_counts, runtime_weights)


if __name__ == "__main__":
    raise SystemExit("REVIEW_ONLY_NO_STANDALONE_EXECUTION")
