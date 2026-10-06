from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from research.runtime import single_module_runner as sms
from ultralytics.nn.tasks import DetectionModel
from ultralytics.research.baseline_trainer import GovernedDetectionTrainer
from ultralytics.research.single_module_trainer import (
    GovernedSingleModuleTrainer,
)


ROOT = Path(__file__).resolve().parents[2]


def test_screen_registry_contract_is_fixed():
    fields, rows = sms.read_registry()

    assert len(rows) == 4
    assert {r["experiment_id"] for r in rows} == set(
        sms.EXPERIMENT_IDS
    )

    for row in rows:
        assert row["data_binding"] == "DATA01:B-ORG:v1"
        assert row["initialization"] == "pretrained"
        assert row["seed"] == "42"
        assert row["epochs"] == "100"
        assert row["imgsz"] == "1024"
        assert row["batch"] == "16"
        assert row["optimizer"] == "SGD"
        assert row["test_access"] == "NONE"
        assert row["status"] in {
            "REGISTERED_NOT_AUTHORIZED",
            "AUTHORIZED",
            "COMPLETE",
        }


def test_screen_contract_keeps_test_and_combination_firewalls():
    contract = sms.load_contract()

    assert (
        contract["development_condition"]["data_binding"]
        == "DATA01:B-ORG:v1"
    )

    assert (
        contract["development_condition"]["initialization"]
        == "pretrained"
    )

    assert (
        contract["development_condition"]["test_access"]
        == "NONE"
    )

    assert (
        contract["advancement_rules"][
            "combination_training_authorized"
        ]
        is False
    )


def test_single_module_trainer_is_governed_extension():
    assert issubclass(
        GovernedSingleModuleTrainer,
        GovernedDetectionTrainer,
    )


def test_candidate_model_identity_contracts():
    baseline = DetectionModel(
        str(sms.BASELINE_YAML),
        ch=3,
        nc=9,
        verbose=False,
    )

    baseline_state = baseline.state_dict()

    assert (
        sum(p.numel() for p in baseline.parameters())
        == 9_431_275
    )

    assert len(baseline_state) == 499

    for experiment_id, spec in sms.CANDIDATES.items():
        path = ROOT / spec["model_yaml"]

        torch.manual_seed(42)

        model = DetectionModel(
            str(path),
            ch=3,
            nc=9,
            verbose=False,
        )

        state = model.state_dict()

        assert (
            sum(p.numel() for p in model.parameters())
            == spec["expected_parameters"]
        )

        assert (
            len(state)
            == spec["expected_state_items"]
        )

        for key, value in baseline_state.items():
            assert key in state
            assert state[key].shape == value.shape

        new_keys = sorted(
            set(state) - set(baseline_state)
        )

        assert (
            len(new_keys)
            == spec["expected_new_state_items"]
        )

        for key in new_keys:
            assert any(
                marker in key
                for marker
                in spec[
                    "allowed_new_state_markers"
                ]
            )


def test_registered_parameter_counts_match_runtime_contract():
    _fields, rows = sms.read_registry()

    by_id = {
        row["experiment_id"]: row
        for row in rows
    }

    for experiment_id, spec in sms.CANDIDATES.items():
        assert int(
            by_id[
                experiment_id
            ][
                "expected_parameters"
            ]
        ) == spec["expected_parameters"]


def test_model_yaml_hashes_match_registered_contract():
    contract = sms.load_contract()

    by_id = {
        item["experiment_id"]: item
        for item in contract["candidates"]
    }

    for experiment_id, spec in sms.CANDIDATES.items():
        path = ROOT / spec["model_yaml"]

        assert (
            sms.br.sha256_file(path)
            == by_id[
                experiment_id
            ][
                "model_yaml_sha256"
            ]
        )


def test_authorization_is_fail_closed_while_registration_only():
    _fields, rows = sms.read_registry()

    statuses = {
        row["status"]
        for row in rows
    }

    # This source test is valid both before and after the later
    # authorization-only commit.
    if statuses == {"REGISTERED_NOT_AUTHORIZED"}:
        assert all(
            not row["source_commit"].strip()
            for row in rows
        )

        contract = sms.load_contract()

        assert (
            contract["firewall"]["training_authorized"]
            is False
        )

        assert (
            contract["provenance"]["source_commit"]
            is None
        )