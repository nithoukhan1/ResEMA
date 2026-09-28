from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent


def test_epoch_calibration_registration_and_recipe():
    matrix = RESEARCH / "05_experiments/EPOCH_CALIBRATION_EXPERIMENTS.csv"
    with matrix.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == "BASE-B-ORG-SCR-S42-E200-CAL"
    assert row["status"] == "NOT_STARTED"
    assert row["data_binding"] == "DATA01:B-ORG:v1"
    assert row["initialization"] == "scratch"
    assert row["seed"] == "42"
    assert row["epochs"] == "200"
    assert row["patience"] == "200"
    assert row["test_access"] == "NONE"
    assert row["source_commit"] == "" or len(row["source_commit"]) == 40

    cal = yaml.safe_load(
        (RESEARCH / "05_experiments/EPOCH_CALIBRATION.yaml").read_text(encoding="utf-8")
    )
    assert cal["status"] == "FROZEN_EPOCH_CALIBRATION_RECIPE_V1"
    assert cal["explicit_overrides"]["training"] == {"epochs": 200, "patience": 200}
    assert cal["preserved"]["cos_lr"] is True
    assert cal["preserved"]["close_mosaic"] == 10
    assert cal["budget_semantics"]["fresh_training"] is True
    assert cal["budget_semantics"]["resume_from_100_epoch_baseline"] is False

    contract = json.loads(
        (RESEARCH / "05_experiments/EPOCH_CALIBRATION_CONTRACT.json").read_text(encoding="utf-8")
    )
    assert contract["status"] == "IMPLEMENTED_RESUME_AWARE"
    assert contract["invariants"]["test_access"] == "NONE"
    assert "resume from the completed canonical 100-epoch baseline" in contract["prohibited"]


def test_epoch_calibration_runner_has_fresh_and_resume_firewalls():
    text = (RESEARCH / "runtime/epoch_calibration_runner.py").read_text(encoding="utf-8")
    for token in [
        "fresh-preflight",
        "fresh-execute",
        "resume-preflight",
        "resume-execute",
        "DATA01:B-ORG:v1",
        "EXPECTED_BRANCH",
        "source_commit is blank",
        "test_directory_scan_pruned",
        "test_predictions",
        "test_metrics",
        "epochs",
        "patience",
        "200",
        "rr._load_checkpoint_metadata",
        "rr._verify_results_against_checkpoint",
        "resume=True",
    ]:
        assert token in text
