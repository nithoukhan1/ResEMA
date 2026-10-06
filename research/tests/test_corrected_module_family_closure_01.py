import csv
import json
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
EXP = RESEARCH / "05_experiments"


def read_csv(name):
    with (EXP / name).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def test_corrected_module_family_closure_is_exact():
    closure = json.loads(
        (
            EXP
            / "CORRECTED_MODULE_FAMILY_CLOSURE_01.json"
        ).read_text(encoding="utf-8")
    )

    assert closure["status"] == "CLOSED"

    decision = closure["controlling_decision"]

    assert (
        decision["corrected_module_family_screen_closed"]
        is True
    )

    assert (
        decision["selected_family_candidate"]
        == "YOLO11s + SCConv-Early"
    )

    assert (
        decision["global_final_paper_architecture_frozen"]
        is False
    )

    assert (
        decision[
            "replacement_research_pending_residual_diagnosis"
        ]
        is True
    )

    assert (
        decision["new_gpu_training_authorized"]
        is False
    )

    assert decision["split_b_test_access"] == "NONE"


def test_single_module_ledger_is_closed_exactly():
    rows = read_csv(
        "SINGLE_MODULE_SCREEN_01_EXPERIMENTS.csv"
    )

    by_id = {
        row["experiment_id"]: row
        for row in rows
    }

    assert set(by_id) == {
        "BORG-PT-S42-SCCONV-EARLY-E100",
        "BORG-PT-S42-SCCONV-4STAGE-E100",
        "BORG-PT-S42-DYSAMPLE-E100",
        "BORG-PT-S42-CANONICAL-EMA-E100",
    }

    assert {
        row["status"]
        for row in rows
    } == {"COMPLETE"}

    assert {
        row["authorization_commit"]
        for row in rows
    } == {
        "9d65b7adae3d488f1cb70856476fef0488e9749a"
    }

    assert (
        by_id[
            "BORG-PT-S42-SCCONV-EARLY-E100"
        ]["decision"]
        == "SELECTED_FAMILY_CANDIDATE_PRIMARY_METRIC"
    )

    assert (
        by_id[
            "BORG-PT-S42-DYSAMPLE-E100"
        ]["decision"]
        == "DO_NOT_PROMOTE_NEGATIVE_DELTA_VS_BASELINE"
    )


def test_combination_ledger_is_closed_exactly():
    rows = read_csv(
        "COMBINATION_SCREEN_01_EXPERIMENTS.csv"
    )

    assert len(rows) == 1

    row = rows[0]

    assert row["status"] == "COMPLETE"

    assert (
        row["authorization_commit"]
        == "22117d7334b53c7782a872a0114d0831fe41cb57"
    )

    assert (
        row["decision"]
        == "DO_NOT_PROMOTE_COMBINATION_PRIMARY_METRIC"
    )

    assert row["test_access"] == "NONE"


def test_current_state_keeps_global_architecture_open():
    current = (
        RESEARCH / "CURRENT.md"
    ).read_text(encoding="utf-8")

    assert (
        "CORRECTED_MODULE_FAMILY_SCREEN_CLOSED=TRUE"
        in current
    )

    assert (
        "GLOBAL_FINAL_PAPER_ARCHITECTURE_FROZEN=FALSE"
        in current
    )

    assert (
        "REPLACEMENT_RESEARCH_PENDING_RESIDUAL_DIAGNOSIS=TRUE"
        in current
    )

    assert (
        "NEW_GPU_TRAINING_AUTHORIZED=FALSE"
        in current
    )

    assert "TEST_ACCESS=NONE" in current


def test_diagnostic_scope_includes_dysample_and_all_family_controls():
    closure = json.loads(
        (
            EXP
            / "CORRECTED_MODULE_FAMILY_CLOSURE_01.json"
        ).read_text(encoding="utf-8")
    )

    diagnostic = closure["diagnostic_scope"]

    ids = {
        row["experiment_id"]
        for row in diagnostic["models"]
    }

    assert ids == {
        "BASE-B-ORG-PT-S42",
        "BORG-PT-S42-SCCONV-EARLY-E100",
        "BORG-PT-S42-SCCONV-4STAGE-E100",
        "BORG-PT-S42-DYSAMPLE-E100",
        "BORG-PT-S42-CANONICAL-EMA-E100",
        "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100",
    }

    dysample = diagnostic["dysample_observation"]

    assert dysample["dysample"]["precision"] == 0.68413
    assert dysample["dysample"]["recall"] == 0.60415
    assert dysample["dysample"]["map50"] == 0.64235
    assert dysample["dysample"]["map50_95"] == 0.41525

    assert dysample["delta"]["map50_95"] == -0.00179

    policy = closure["replacement_policy"]

    assert (
        policy["dysample_branch"][
            "attention_is_default_replacement"
        ]
        is False
    )

    historical = diagnostic[
        "historical_combination_policy"
    ]

    assert (
        historical[
            "corrected_scconv_plus_dysample_run_exists"
        ]
        is False
    )

    assert (
        historical[
            "corrected_three_way_scconv_dysample_ema_run_exists"
        ]
        is False
    )
