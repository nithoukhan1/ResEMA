from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.runtime import (
    combination_screen_resume_runner as cr,
)
from research.runtime import (
    combination_screen_runner as comb,
)


def _touch_required_run(
    root: Path,
    experiment_id: str,
) -> Path:
    run = root / experiment_id

    for rel in cr.REQUIRED_PRIOR_RUN_FILES:
        path = run / rel
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(b"x")

    return run


def test_resume_contract_is_fixed_and_test_free():
    contract = cr.load_resume_contract()

    assert (
        contract["schema_version"]
        == "COMB01-resume-contract-v1.0"
    )

    assert (
        contract["status"]
        == "IMPLEMENTED"
    )

    assert (
        contract["experiment_family"]
        == comb.SCREEN_ID
    )

    assert (
        contract["branch"]
        == comb.EXPECTED_BRANCH
    )

    assert (
        contract["experiment_ids"]
        == list(
            comb.EXPERIMENT_IDS
        )
    )

    assert (
        contract["allowed_modes"]
        == [
            "preflight-only",
            "execute",
        ]
    )

    assert (
        contract["test_access"]
        == "NONE"
    )

    assert (
        contract[
            "scientific_overrides_allowed"
        ]
        is False
    )

    assert (
        contract[
            "training_authorization_delegated_to_registry_and_authorization_record"
        ]
        is True
    )

    assert (
        contract[
            "technical_verification_sha256"
        ]
        == comb.TECHNICAL_VERIFICATION_SHA256
    )

    assert (
        contract["expected_parameters"]
        == 9_574_241
    )

    assert (
        contract["expected_state_items"]
        == 565
    )

    assert (
        contract[
            "expected_original_initialization"
        ]
        == "pretrained"
    )

    for key in (
        "require_exact_original_execution_commit",
        "require_full_run_copy_before_resume",
        "multi_session_indexed_provenance",
        "require_checkpoint_optimizer_state",
        "require_checkpoint_scaler_state",
        "require_results_checkpoint_consistency",
        "require_candidate_parameter_state_identity",
        "require_test_free_runtime_yaml",
    ):
        assert contract[key] is True


def test_resume_fails_closed_while_registration_only_before_runtime_or_prior_run(
    monkeypatch,
    tmp_path,
):
    reached = {
        "runtime": False,
        "membership": False,
        "prior_run": False,
    }

    registration_row = dict(
        comb.load_row(
            comb.EXPERIMENT_ID
        )
    )

    registration_row["status"] = (
        "REGISTERED_NOT_AUTHORIZED"
    )

    registration_row["source_commit"] = ""
    registration_row[
        "authorization_commit"
    ] = ""

    def registration_load_row(
        experiment_id,
    ):
        assert (
            experiment_id
            == comb.EXPERIMENT_ID
        )

        return dict(
            registration_row
        )

    def clean_git(*args, **kwargs):
        if args == (
            "branch",
            "--show-current",
        ):
            return comb.EXPECTED_BRANCH

        if args == (
            "rev-parse",
            "HEAD",
        ):
            return "registration-only-head"

        if args == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return ""

        raise AssertionError(args)

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

    def forbidden_prior(*args, **kwargs):
        reached["prior_run"] = True
        raise AssertionError(
            "prior run must not be reached"
        )

    monkeypatch.setattr(
        comb,
        "load_row",
        registration_load_row,
    )

    monkeypatch.setattr(
        comb.br,
        "git",
        clean_git,
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
        cr,
        "_discover_prior_run",
        forbidden_prior,
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="source_commit is blank",
    ):
        cr.analyze_resume(
            comb.EXPERIMENT_ID,
            input_root=tmp_path,
            work_root=tmp_path,
        )

    assert reached == {
        "runtime": False,
        "membership": False,
        "prior_run": False,
    }


def test_prior_run_discovery_requires_exactly_one_complete_candidate(
    tmp_path,
):
    experiment_id = comb.EXPERIMENT_ID

    first = tmp_path / "one"
    second = tmp_path / "two"

    expected = _touch_required_run(
        first,
        experiment_id,
    )

    assert (
        cr._discover_prior_run(
            tmp_path,
            experiment_id,
        )
        == expected.resolve()
    )

    _touch_required_run(
        second,
        experiment_id,
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="exactly one complete prior",
    ):
        cr._discover_prior_run(
            tmp_path,
            experiment_id,
        )


def test_prior_run_discovery_prunes_test_like_directories(
    tmp_path,
):
    hidden = (
        tmp_path
        / "test_hidden"
    )

    _touch_required_run(
        hidden,
        comb.EXPERIMENT_ID,
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="exactly one complete prior",
    ):
        cr._discover_prior_run(
            tmp_path,
            comb.EXPERIMENT_ID,
        )


def test_comb_resume_session_indices_must_match_generic_resume_indices(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        cr.base_resume,
        "_existing_resume_indices",
        lambda _run: [1],
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="provenance is incomplete",
    ):
        cr._existing_comb_resume_indices(
            tmp_path
        )

    sessions = (
        tmp_path
        / "governance/resume_sessions"
    )

    sessions.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        sessions
        / "COMB01_RESUME_MODEL_RUNTIME_001.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    assert (
        cr._existing_comb_resume_indices(
            tmp_path
        )
        == [1]
    )


def _valid_model_contract() -> dict:
    candidate = comb.CANDIDATES[
        comb.EXPERIMENT_ID
    ]

    return {
        "model_yaml":
            candidate["model_yaml"],

        "model_yaml_sha256":
            comb.MODEL_YAML_SHA256,

        "expected_parameters":
            candidate[
                "expected_parameters"
            ],

        "expected_state_items":
            candidate[
                "expected_state_items"
            ],

        "new_target_state_items":
            candidate[
                "expected_new_state_items"
            ],

        "expected_transferable_source_items":
            493,

        "technical_verification_sha256":
            comb.TECHNICAL_VERIFICATION_SHA256,
    }


def test_resume_model_contract_parameter_tamper_fails_closed():
    contract = _valid_model_contract()

    contract[
        "expected_parameters"
    ] += 1

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="model contract drift",
    ):
        cr._verify_model_contract(
            experiment_id=
                comb.EXPERIMENT_ID,
            model_contract=
                contract,
        )


def test_resume_model_contract_technical_sha_tamper_fails_closed():
    contract = _valid_model_contract()

    contract[
        "technical_verification_sha256"
    ] = "0" * 64

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="model contract drift",
    ):
        cr._verify_model_contract(
            experiment_id=
                comb.EXPERIMENT_ID,
            model_contract=
                contract,
        )


def test_original_comb_runtime_rejects_test_access(
    tmp_path,
):
    contract = _valid_model_contract()

    preflight = {
        "model_contract":
            contract,
    }

    payload = {
        "schema_version":
            "COMB01-model-runtime-v1.0",

        "experiment_family":
            comb.SCREEN_ID,

        "experiment_id":
            comb.EXPERIMENT_ID,

        "training_source_commit":
            "source",

        "execution_commit":
            "execution",

        "model_contract":
            contract,

        "initialization_audit": {
            "initialization":
                "pretrained",

            "official_checkpoint_loaded":
                True,

            "transferable_state_items":
                493,
        },

        "technical_verification": {
            "sha256":
                comb.TECHNICAL_VERIFICATION_SHA256,

            "status":
                "PASS",
        },

        "test_access":
            "ALLOWED",
    }

    path = (
        tmp_path
        / "COMB01_MODEL_RUNTIME.json"
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="original model runtime drift",
    ):
        cr._verify_original_comb_runtime(
            experiment_id=
                comb.EXPERIMENT_ID,

            path=
                path,

            original_preflight=
                preflight,

            source_commit=
                "source",

            execution_commit=
                "execution",
        )


def test_original_comb_runtime_rejects_technical_verification_tamper(
    tmp_path,
):
    contract = _valid_model_contract()

    preflight = {
        "model_contract":
            contract,
    }

    payload = {
        "schema_version":
            "COMB01-model-runtime-v1.0",

        "experiment_family":
            comb.SCREEN_ID,

        "experiment_id":
            comb.EXPERIMENT_ID,

        "training_source_commit":
            "source",

        "execution_commit":
            "execution",

        "model_contract":
            contract,

        "initialization_audit": {
            "initialization":
                "pretrained",

            "official_checkpoint_loaded":
                True,

            "transferable_state_items":
                493,
        },

        "technical_verification": {
            "sha256":
                "0" * 64,

            "status":
                "PASS",
        },

        "test_access":
            "NONE",
    }

    path = (
        tmp_path
        / "COMB01_MODEL_RUNTIME.json"
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        comb.CombinationGovernanceError,
        match="technical-verification",
    ):
        cr._verify_original_comb_runtime(
            experiment_id=
                comb.EXPERIMENT_ID,

            path=
                path,

            original_preflight=
                preflight,

            source_commit=
                "source",

            execution_commit=
                "execution",
        )


def test_trainer_uses_indexed_comb_resume_runtime_filename():
    path = (
        comb.ROOT
        / "ultralytics/research/"
        "combination_screen_trainer.py"
    )

    text = path.read_text(
        encoding="utf-8",
    )

    assert (
        '"resume_session_index"'
        in text
    )

    assert (
        '"COMB01_RESUME_MODEL_RUNTIME_"'
        in text
    )

    assert (
        'f"{session_index:03d}.json"'
        in text
    )

    assert (
        '"COMB01_RESUME_MODEL_RUNTIME.json"'
        not in text
    )

    assert '"resume_sessions"' in text


def test_resume_runner_has_no_scientific_override_cli():
    text = Path(
        cr.__file__
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


def test_resume_execute_uses_combination_trainer_and_governed_preflight():
    text = Path(
        cr.__file__
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "GovernedCombinationScreenTrainer"
        in text
    )

    assert "resume=True" in text

    assert (
        "RESEMA_RESUME_PREFLIGHT_MANIFEST"
        in text
    )

    assert (
        "RESEMA_RESUME_PREFLIGHT_SHA256"
        in text
    )

    assert (
        "RESEMA_SCREEN_FAMILY"
        in text
    )

    assert (
        "technical_verification"
        in text
    )


def test_resume_runner_contains_no_residual_sms_namespace():
    text = Path(
        cr.__file__
    ).read_text(
        encoding="utf-8",
    )

    for marker in (
        "sms.",
        "SMS01",
        "SMS-01",
        "GovernedSingleModuleTrainer",
        "single_module_trainer",
    ):
        assert marker not in text
