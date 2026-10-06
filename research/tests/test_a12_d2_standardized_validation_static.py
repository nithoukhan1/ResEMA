from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RUNNER = (
    ROOT
    / "research/runtime/a12_d2_standardized_validation.py"
)
PROTOCOL = (
    ROOT
    / "research/06_diagnostics/"
      "A12_D1_STANDARDIZED_VALIDATION_DIAGNOSTIC_PROTOCOL.md"
)


def test_d2_runner_exists_and_parses():
    assert RUNNER.is_file()
    ast.parse(
        RUNNER.read_text(encoding="utf-8")
    )


def test_d2_runner_is_validation_only():
    text = RUNNER.read_text(encoding="utf-8")

    assert ".train(" not in text
    assert ".predict(" not in text
    assert 'split="test"' not in text
    assert "split='test'" not in text

    assert text.count("model.val(") == 1
    assert 'split="val"' in text
    assert "save_txt=True" in text
    assert "save_conf=True" in text
    assert 'conf=runtime_contract["conf"]' in text

    assert '"training_started": False' in text
    assert '"test_access": "NONE"' in text
    assert 'print("TRAINING_STARTED=FALSE")' in text
    assert 'print("TEST_ACCESS=NONE")' in text


def test_d2_requires_explicit_source_bound_execution_authorization():
    text = RUNNER.read_text(encoding="utf-8")

    assert "A12_D2_EXECUTION_AUTHORIZATION.json" in text
    assert "def verify_execution_authorization" in text
    assert '"status": "AUTHORIZED"' in text
    assert '"validation_execution_authorized": True' in text
    assert '"new_gpu_training_authorized": False' in text
    assert "def markdown_status" in text
    assert (
        'protocol_state == "`A12_D2_EXECUTION_AUTHORIZED`"'
        in text
    )
    assert "A12-D2 execution authorization: TRUE" not in text
    assert "current_runner_sha == runner_sha" in text
    assert "git_file_bytes(" in text
    assert "git_is_ancestor(source_commit, current_head)" in text

    main_start = text.index("def main() -> int:")
    main_text = text[main_start:]
    authorization_call = main_text.index(
        "authorization = verify_execution_authorization("
    )
    output_creation = main_text.index(
        "args.output_root.mkdir("
    )
    dataset_binding = main_text.index(
        "data = d1.verify_dataset_binding("
    )

    assert authorization_call < output_creation < dataset_binding


def test_d2_reuses_frozen_d1_preflight_contract():
    text = RUNNER.read_text(encoding="utf-8")

    for commit in (
        "e2367057e3a4ffabcb5cde1c2ff569df264625ef",
        "7aff6edf2623135f90776c0760a8e80b6fc5298f",
        "5aa7eb6acffcb5c2922e6b2c16aae7e80a5a20e9",
    ):
        assert commit in text

    assert (
        '"d33ead0f712aa432e4afdd67aa89f6a4481acc543ae0486b8424a95761909de8"'
        in text
    )
    assert (
        '"b2114397eefc1ac377352fb4e0429df0fe6fc954a307e4b0df89139b4bed82c4"'
        in text
    )
    assert (
        '"f9f47b7816caec3556fd1700aee79f8e716a8efa866bb9030e571a947b4d9de8"'
        in text
    )

    assert "d1.verify_artifact_registry()" in text
    assert "d1.verify_historical_experiment_contracts()" in text
    assert "d1.verify_runtime_contract()" in text
    assert "d1.verify_dataset_binding(" in text
    assert "d1.discover_checkpoint(" in text
    assert "d1.inspect_model(" in text


def test_d2_freezes_required_metric_and_export_outputs():
    text = RUNNER.read_text(encoding="utf-8")

    for token in (
        '"ap50"',
        '"ap75"',
        '"ap50_95"',
        '"support_images"',
        '"support_instances"',
        '"precision"',
        '"recall"',
        '"f1"',
        "PREDICTION_EXPORT_AUDIT.json",
        "PREDICTIONS.csv",
        "PER_CLASS_METRICS.csv",
        "VALIDATION_ONLY_SUMMARY.json",
        "VALIDATION_IMAGE_INDEX.csv",
        "VALIDATION_GROUND_TRUTH.csv",
        "STANDARDIZED_VALIDATION_AGGREGATE.csv",
        "STANDARDIZED_VALIDATION_PER_CLASS.csv",
        "A12_D2_EXECUTION_MANIFEST.json",
        "A12_D2_ARTIFACT_MANIFEST.csv",
        "A12_D2_FAILURE_MANIFEST.json",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
        "BoxR_curve.png",
    ):
        assert token in text

    assert "metrics.box.all_ap[i, 5]" in text
    assert "metrics.box.map75" in text
    assert '"effective_prediction_floor": 0.001' in text
    assert '"save_txt": True' in text
    assert '"save_conf": True' in text


def test_d2_preserves_self_contained_validation_reference():
    text = RUNNER.read_text(encoding="utf-8")

    assert "def build_validation_reference" in text
    assert "patient_id" in text
    assert "ground_truth_box_count" in text
    assert "area_norm" in text
    assert "size_bin" in text
    assert '"small"' in text
    assert '"medium"' in text
    assert '"large"' in text
    assert "known_unreadable_filestem" in text
    assert "class_support[2] == 0" in text


def test_d2_native_confusion_matrix_semantics_are_explicit():
    text = RUNNER.read_text(encoding="utf-8")

    assert "NATIVE_CONFUSION_MATRIX_CONTRACT" in text
    assert '"effective_confidence": 0.25' in text
    assert '"iou_threshold": 0.45' in text
    assert "not the frozen offline" in text


def test_d2_foreignbody_zero_support_remains_na():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'foreignbody["class_name"] == "foreignbody"' in text
    assert (
        'foreignbody["status"] == "NO_VALIDATION_SUPPORT"'
        in text
    )


def test_d2_does_not_run_offline_threshold_analysis():
    text = RUNNER.read_text(encoding="utf-8")

    # D2 exports one low-confidence prediction set.
    # Fixed threshold analyses occur later, offline.
    assert '"confidence": [0.05, 0.10, 0.25, 0.50]' in text
    assert '"localization_iou": [0.50, 0.75]' in text
    assert '"offline_diagnostics_executed": False' in text
    assert "greedy_match" not in text
    assert "background false positive" not in text.lower()
    assert "duplicate_error" not in text


def test_protocol_status_marks_source_frozen_not_execution_authorized():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert (
        "A12_D2_SOURCE_FROZEN_PENDING_EXECUTION_AUTHORIZATION"
        in text
    )
    assert "A12-D2 execution authorization: FALSE" in text
    assert "A12-D2 execution authorization: TRUE" not in text
    assert "A12-D2 will preserve post-NMS validation detections" in text
    assert "save_txt=True" in text
    assert "save_conf=True" in text
    assert "0.05" in text
    assert "0.10" in text
    assert "0.25" in text
    assert "0.50" in text
