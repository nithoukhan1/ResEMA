from __future__ import annotations

import ast
import json
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
AUTHORIZATION = (
    ROOT
    / "research/06_diagnostics/"
      "A12_D2_EXECUTION_AUTHORIZATION.json"
)
CLOSURE = (
    ROOT
    / "research/06_diagnostics/"
      "A12_D2_STANDARDIZED_VALIDATION_CLOSURE.json"
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


def test_protocol_marks_d2_closed_and_preserved():
    text = PROTOCOL.read_text(encoding="utf-8")
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))

    lines = text.splitlines()
    status_index = lines.index("## Status")
    state = next(line.strip() for line in lines[status_index + 1:] if line.strip())
    assert state == "`A12_D2_EXECUTION_COMPLETE_PRESERVED_CLOSED`"

    assert authorization["status"] == "AUTHORIZED"
    assert authorization["source_commit"] == "758f0643e2bceb8d996d8e9603fb836d9319d8eb"
    assert authorization["runner_sha256"] == "9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250"
    assert authorization["validation_execution_authorized"] is True
    assert authorization["new_gpu_training_authorized"] is False
    assert authorization["test_access"] == "NONE"

    assert closure["schema_version"] == "A12-D2-standardized-validation-closure-v1.0"
    assert closure["status"] == "CLOSED"
    assert closure["execution_head"] == "791566cb04c91258eadbbb8ee6120edac9cc4c0f"
    assert closure["d2_source_commit"] == "758f0643e2bceb8d996d8e9603fb836d9319d8eb"
    assert closure["d2_runner_sha256"] == "9f31b762374e5b195cd624cdf903791f08056a1282db0355c25aa71b768d3250"
    assert closure["authorization_sha256"] == "8cbb1e999d3ffb2e7e4b2e30d9882c5510c076d116468609962cd06e36b3dcb1"
    assert closure["preservation_archive_sha256"] == "419ac71b1e42b791168a5ea24cf87d71b8a3d1eba9333d183a73009733056b93"
    assert closure["global_artifact_manifest_rows"] == 18407
    assert closure["frozen_validation_images"] == 3050
    assert closure["operational_validation_images"] == 3049
    assert closure["validation_patients"] == 914
    assert closure["frozen_ground_truth_boxes"] == 7113
    assert closure["operational_ground_truth_boxes"] == 7110
    assert closure["unreadable_image_ground_truth_boxes"] == 3
    assert closure["authorization_consumed_by_execution"] is True
    assert closure["rerun_requires_new_authorization"] is True
    assert closure["offline_diagnostics_executed"] is False
    assert closure["new_gpu_training_authorized"] is False
    assert closure["test_access"] == "NONE"

    assert "A12-D2 will preserve post-NMS validation detections" in text
    assert "save_txt=True" in text
    assert "save_conf=True" in text
    assert "0.05" in text
    assert "0.10" in text
    assert "0.25" in text
    assert "0.50" in text
