from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.runtime import (
    combination_screen_runner as comb,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.research.baseline_trainer import (
    GovernedDetectionTrainer,
)
from ultralytics.research.combination_screen_trainer import (
    GovernedCombinationScreenTrainer,
)


ROOT = Path(__file__).resolve().parents[2]


def test_comb_registry_scientific_contract_is_fixed():
    fields, rows = comb.read_registry()

    assert len(rows) == 1

    row = rows[0]

    assert (
        row["experiment_id"]
        == comb.EXPERIMENT_ID
    )
    assert (
        row["model_yaml"]
        == comb.MODEL_RELATIVE
    )
    assert (
        row["data_binding"]
        == "DATA01:B-ORG:v1"
    )
    assert row["initialization"] == "pretrained"
    assert row["seed"] == "42"
    assert row["epochs"] == "100"
    assert row["imgsz"] == "1024"
    assert row["batch"] == "16"
    assert row["optimizer"] == "SGD"
    assert row["primary_metric"] == "val_mAP50-95"
    assert row["test_access"] == "NONE"

    assert (
        row["parameter_hypothesis"]
        == "9574241"
    )
    assert (
        row["state_item_hypothesis"]
        == "565"
    )
    assert (
        row["new_state_item_hypothesis"]
        == "66"
    )

    assert row["status"] in {
        "REGISTERED_NOT_AUTHORIZED",
        "AUTHORIZED",
    }

    if (
        row["status"]
        == "REGISTERED_NOT_AUTHORIZED"
    ):
        assert row["source_commit"].strip() == ""


def test_comb_contract_keeps_frozen_condition_and_test_firewall():
    contract = comb.load_contract()

    condition = contract[
        "development_condition"
    ]

    assert condition["split"] == "B"
    assert condition["train_data"] == "original"
    assert (
        condition["data_binding"]
        == "DATA01:B-ORG:v1"
    )
    assert (
        condition["initialization"]
        == "pretrained"
    )
    assert condition["seed"] == 42
    assert condition["epochs"] == 100
    assert condition["imgsz"] == 1024
    assert condition["batch_global"] == 16
    assert condition["optimizer"] == "SGD"
    assert (
        condition["selection_split"]
        == "val"
    )
    assert (
        condition["test_access"]
        == "NONE"
    )

    firewall = contract["firewall"]

    assert firewall["test_access"] == "NONE"

    assert firewall[
        "training_authorized"
    ] in {
        False,
        True,
    }


def test_technical_verification_is_hash_bound_and_closed():
    assert (
        comb.br.sha256_file(
            comb.TECHNICAL_VERIFICATION
        )
        == comb.TECHNICAL_VERIFICATION_SHA256
    )

    technical = (
        comb.load_technical_verification()
    )

    assert technical["status"] == "PASS"

    assert (
        technical["experiment_id"]
        == comb.EXPERIMENT_ID
    )

    model = technical[
        "model_identity"
    ]

    assert model["parameters"] == 9_574_241
    assert model["state_dict_items"] == 565
    assert model["new_state_items"] == 66
    assert (
        model["scconv_new_state_items"]
        == 38
    )
    assert (
        model[
            "canonical_ema_new_state_items"
        ]
        == 28
    )
    assert model["zero_gates"] == 6

    conclusion = technical[
        "technical_conclusion"
    ]

    assert (
        conclusion[
            "technical_architecture_validation_closed"
        ]
        is True
    )

    assert (
        conclusion[
            "gpu_training_authorized"
        ]
        is False
    )

    assert (
        technical["firewall"][
            "test_access"
        ]
        == "NONE"
    )


def test_combination_trainer_extends_governed_detection_trainer():
    assert issubclass(
        GovernedCombinationScreenTrainer,
        GovernedDetectionTrainer,
    )


def test_candidate_architecture_parameter_state_and_gate_contract():
    torch.manual_seed(42)

    baseline = DetectionModel(
        str(
            comb.BASELINE_YAML
        ),
        ch=3,
        nc=9,
        verbose=False,
    )

    torch.manual_seed(42)

    candidate = DetectionModel(
        str(
            comb.MODEL_YAML
        ),
        ch=3,
        nc=9,
        verbose=False,
    )

    baseline_state = (
        baseline.state_dict()
    )

    candidate_state = (
        candidate.state_dict()
    )

    assert (
        sum(
            p.numel()
            for p in baseline.parameters()
        )
        == 9_431_275
    )

    assert len(baseline_state) == 499

    assert (
        sum(
            p.numel()
            for p in candidate.parameters()
        )
        == 9_574_241
    )

    assert len(candidate_state) == 565

    for key, value in baseline_state.items():
        assert key in candidate_state
        assert (
            candidate_state[key].shape
            == value.shape
        )

    new_keys = (
        set(candidate_state)
        - set(baseline_state)
    )

    assert len(new_keys) == 66

    sc_keys = {
        key
        for key in new_keys
        if ".sc_adapters." in key
    }

    ema_keys = {
        key
        for key in new_keys
        if ".ema_adapter." in key
    }

    assert len(sc_keys) == 38
    assert len(ema_keys) == 28

    assert (
        new_keys
        == (
            sc_keys
            | ema_keys
        )
    )

    gates = [
        (
            name,
            parameter,
        )
        for name, parameter
        in candidate.named_parameters()
        if (
            name.endswith(".alpha")
            and (
                ".sc_adapters." in name
                or ".ema_adapter." in name
            )
        )
    ]

    assert len(gates) == 6

    assert all(
        torch.equal(
            parameter.detach(),
            torch.zeros_like(
                parameter.detach()
            ),
        )
        for _name, parameter
        in gates
    )


def test_runtime_candidate_authority_matches_verified_architecture():
    assert list(
        comb.CANDIDATES
    ) == [
        comb.EXPERIMENT_ID
    ]

    candidate = comb.CANDIDATES[
        comb.EXPERIMENT_ID
    ]

    assert (
        candidate["model_yaml"]
        == comb.MODEL_RELATIVE
    )
    assert (
        candidate[
            "expected_parameters"
        ]
        == 9_574_241
    )
    assert (
        candidate[
            "expected_state_items"
        ]
        == 565
    )
    assert (
        candidate[
            "expected_new_state_items"
        ]
        == 66
    )
    assert (
        candidate[
            "technical_verification_sha256"
        ]
        == comb.TECHNICAL_VERIFICATION_SHA256
    )


def test_fresh_preflight_fails_before_runtime_dataset_or_checkpoint_when_unauthorized(
    monkeypatch,
    tmp_path,
):
    reached = {
        "runtime": False,
        "membership": False,
        "checkpoint": False,
    }

    def denied_authorization(_row):
        raise comb.CombinationGovernanceError(
            "hostile registration-only block"
        )

    def forbidden_runtime(*args, **kwargs):
        reached["runtime"] = True
        raise AssertionError(
            "runtime must not be reached"
        )

    def forbidden_membership(*args, **kwargs):
        reached["membership"] = True
        raise AssertionError(
            "membership must not be reached"
        )

    def forbidden_checkpoint(*args, **kwargs):
        reached["checkpoint"] = True
        raise AssertionError(
            "checkpoint must not be reached"
        )

    monkeypatch.setattr(
        comb,
        "verify_authorization",
        denied_authorization,
    )

    monkeypatch.setattr(
        comb.br,
        "verify_runtime_environment",
        forbidden_runtime,
    )

    monkeypatch.setattr(
        comb.br,
        "discover_membership_directory",
        forbidden_membership,
    )

    monkeypatch.setattr(
        comb.br,
        "discover_checkpoint",
        forbidden_checkpoint,
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="hostile registration-only block",
    ):
        comb.perform_preflight(
            comb.EXPERIMENT_ID,
            input_root=tmp_path,
            work_root=tmp_path,
        )

    assert reached == {
        "runtime": False,
        "membership": False,
        "checkpoint": False,
    }


def test_fresh_runner_has_no_scientific_override_cli():
    text = Path(
        comb.__file__
    ).read_text(
        encoding="utf-8",
    )

    for flag in (
        "--epochs",
        "--imgsz",
        "--batch",
        "--optimizer",
        "--lr0",
        "--lrf",
        "--momentum",
        "--weight-decay",
        "--seed",
        "--data",
        "--device",
    ):
        assert flag not in text

    assert "--experiment-id" in text
    assert "--mode" in text
    assert "preflight-only" in text
    assert "execute" in text
