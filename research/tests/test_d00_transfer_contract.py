from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

D00A = (
    ROOT
    / "research"
    / "06_diagnostics"
    / "D00_A"
)

REGISTRY_PATH = (
    D00A
    / "ARCHITECTURE_REGISTRY.json"
)

BINDING_PATH = (
    D00A
    / "D00_A0_RUNTIME_BINDING.json"
)

RUNNER_PATH = (
    D00A
    / "run_transfer_coverage.py"
)

WORKFLOW_PATH = (
    ROOT
    / ".github"
    / "workflows"
    / "research-integrity.yml"
)

EXPECTED_IDS = [
    "baseline_yolo11s",
    "sc_e50",
    "dysample",
    "resema",
    "sc_dysample_e50",
    "sc_resema_e50",
    "dysample_resema",
    "full_e50",
]

EXPECTED_MODULE_SETS = {
    "baseline_yolo11s":
        (),

    "sc_e50":
        ("C3k2_SC",),

    "dysample":
        ("DySample",),

    "resema":
        ("ResEMA",),

    "sc_dysample_e50":
        ("C3k2_SC", "DySample"),

    "sc_resema_e50":
        ("C3k2_SC", "ResEMA"),

    "dysample_resema":
        ("DySample", "ResEMA"),

    "full_e50":
        (
            "C3k2_SC",
            "DySample",
            "ResEMA",
        ),
}

EXPECTED_OUTPUTS = {
    "transfer_coverage.csv",
    "transfer_unmatched_tensors.csv",
    "transfer_unmatched_layers.csv",
    "transfer_type_mismatch_matches.csv",
    "D00_A_RUN_PROVENANCE.json",
    "D00_A_TRANSFER_COVERAGE_REPORT.md",
}


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def test_d00_a0_runtime_binding_is_frozen():
    record = load_json(
        BINDING_PATH
    )

    assert (
        record["schema_version"]
        == "D00-A0-runtime-binding-v1.0"
    )

    assert (
        record[
            "runtime_binding_sha256"
        ]
        ==
        "eddffe25ded04c7df598d76769a3dd224a794efe542da040a25dadc3c16f90ef"
    )

    assert (
        record["test_seal"]
        == "SEALED_UNTIL_P10"
    )

    assert (
        record[
            "test_label_content_read"
        ]
        is False
    )

    assert (
        record[
            "test_predictions_run"
        ]
        is False
    )


def test_d00_a_registry_has_exact_factorial_matrix():
    registry = load_json(
        REGISTRY_PATH
    )

    architectures = (
        registry["architectures"]
    )

    assert [
        row["id"]
        for row in architectures
    ] == EXPECTED_IDS

    assert len(
        architectures
    ) == 8

    for row in architectures:
        assert (
            tuple(row["modules"])
            == EXPECTED_MODULE_SETS[
                row["id"]
            ]
        )

        assert row["scale"] == "s"
        assert row["target_nc"] == 9


def test_d00_a_registry_yaml_paths_exist():
    registry = load_json(
        REGISTRY_PATH
    )

    for row in (
        registry["architectures"]
    ):
        path = (
            ROOT
            / row["yaml_path"]
        )

        assert path.is_file(), path


def test_d00_a_source_checkpoint_is_external_and_explicit():
    registry = load_json(
        REGISTRY_PATH
    )

    source = registry[
        "source_checkpoint"
    ]

    assert (
        source["logical_name"]
        == "yolo11s.pt"
    )

    assert (
        source["policy"]
        == "EXTERNAL_REQUIRED_NO_AUTO_DOWNLOAD"
    )

    assert (
        source["expected_nc"]
        == 80
    )

    assert (
        source["expected_scale"]
        == "s"
    )

    assert (
        source["sha256"]
        == "TO_BE_RECORDED_AT_RUNTIME"
    )


def test_d00_a_runner_is_result_only_and_has_no_dataset_access():
    text = RUNNER_PATH.read_text(
        encoding="utf-8",
    )

    ast.parse(text)

    forbidden = (
        "images/test",
        "labels/test",
        ".train(",
        ".val(",
        ".predict(",
        "train_aug_historical",
    )

    for token in forbidden:
        assert token not in text

    assert "intersect_dicts" in text
    assert "--weights" in text
    assert "--expected-repo-sha" in text


def test_d00_a_runner_declares_complete_output_contract():
    text = RUNNER_PATH.read_text(
        encoding="utf-8",
    )

    for filename in (
        EXPECTED_OUTPUTS
    ):
        assert filename in text

    assert (
        "loader_non_detect_parameter_coverage_pct"
        in text
    )

    assert (
        "same_type_non_detect_parameter_coverage_pct"
        in text
    )

    assert (
        "aggregate_unmatched_layers"
        in text
    )

    assert (
        "unmatched_top_level_layer_count"
        in text
    )


def test_research_integrity_ci_runs_d00_transfer_contract():
    text = WORKFLOW_PATH.read_text(
        encoding="utf-8",
    )

    assert (
        text.count(
            "test_d00_transfer_contract.py"
        )
        == 1
    )
