from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.runtime import (
    single_module_resume_runner as smsr,
)
from research.runtime import (
    single_module_runner as sms,
)


def _touch_required_run(
    root: Path,
    experiment_id: str,
) -> Path:
    run = root / experiment_id

    for rel in smsr.REQUIRED_PRIOR_RUN_FILES:
        path = run / rel
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(b"x")

    return run


def test_resume_contract_is_fixed_and_test_free():
    contract = smsr.load_resume_contract()

    assert (
        contract["schema_version"]
        == "SMS01-resume-contract-v1.0"
    )
    assert contract["status"] == "IMPLEMENTED"
    assert (
        contract["experiment_family"]
        == sms.SCREEN_ID
    )
    assert (
        contract["branch"]
        == sms.EXPECTED_BRANCH
    )
    assert (
        contract["experiment_ids"]
        == list(sms.EXPERIMENT_IDS)
    )
    assert contract["allowed_modes"] == [
        "preflight-only",
        "execute",
    ]
    assert contract["test_access"] == "NONE"
    assert (
        contract["scientific_overrides_allowed"]
        is False
    )
    assert (
        contract["combination_training_authorized"]
        is False
    )


def test_resume_fails_closed_while_registration_only_before_runtime(
    monkeypatch,
    tmp_path,
):
    reached = {
        "runtime": False,
        "membership": False,
    }

    def clean_git(*args, **kwargs):
        if args == (
            "branch",
            "--show-current",
        ):
            return sms.EXPECTED_BRANCH

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

    monkeypatch.setattr(
        sms.br,
        "git",
        clean_git,
    )
    monkeypatch.setattr(
        sms.br,
        "verify_runtime_environment",
        forbidden_runtime,
    )
    monkeypatch.setattr(
        sms.br,
        "discover_membership_directory",
        forbidden_membership,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="source_commit is blank",
    ):
        smsr.analyze_resume(
            "BORG-PT-S42-DYSAMPLE-E100",
            input_root=tmp_path,
            work_root=tmp_path,
        )

    assert reached == {
        "runtime": False,
        "membership": False,
    }


def test_prior_run_discovery_requires_exactly_one_complete_candidate(
    tmp_path,
):
    experiment_id = (
        "BORG-PT-S42-SCCONV-EARLY-E100"
    )

    first = tmp_path / "one"
    second = tmp_path / "two"

    _touch_required_run(
        first,
        experiment_id,
    )

    assert (
        smsr._discover_prior_run(
            tmp_path,
            experiment_id,
        )
        == (
            first
            / experiment_id
        ).resolve()
    )

    _touch_required_run(
        second,
        experiment_id,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="exactly one complete prior",
    ):
        smsr._discover_prior_run(
            tmp_path,
            experiment_id,
        )


def test_prior_run_discovery_prunes_test_like_directories(
    tmp_path,
):
    experiment_id = (
        "BORG-PT-S42-DYSAMPLE-E100"
    )

    hidden = tmp_path / "test_hidden"

    _touch_required_run(
        hidden,
        experiment_id,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="exactly one complete prior",
    ):
        smsr._discover_prior_run(
            tmp_path,
            experiment_id,
        )


def test_sms_resume_session_indices_must_match_baseline_resume_indices(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        smsr.base_resume,
        "_existing_resume_indices",
        lambda _run: [1],
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="provenance is incomplete",
    ):
        smsr._existing_sms_resume_indices(
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
        / "SMS01_RESUME_MODEL_RUNTIME_001.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    assert (
        smsr._existing_sms_resume_indices(
            tmp_path
        )
        == [1]
    )


def test_model_contract_parameter_tamper_fails_closed():
    experiment_id = (
        "BORG-PT-S42-CANONICAL-EMA-E100"
    )

    candidate = sms.CANDIDATES[
        experiment_id
    ]

    contract = {
        "model_yaml":
            candidate["model_yaml"],
        "model_yaml_sha256":
            sms.br.sha256_file(
                sms.ROOT
                / candidate["model_yaml"]
            ),
        "expected_parameters":
            candidate["expected_parameters"] + 1,
        "expected_state_items":
            candidate["expected_state_items"],
        "new_target_state_items":
            candidate[
                "expected_new_state_items"
            ],
        "expected_transferable_source_items":
            493,
    }

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="model contract drift",
    ):
        smsr._verify_model_contract(
            experiment_id=experiment_id,
            model_contract=contract,
        )


def test_original_sms_runtime_requires_test_none(
    tmp_path,
):
    experiment_id = (
        "BORG-PT-S42-DYSAMPLE-E100"
    )

    preflight = {
        "model_contract": {
            "x": 1,
        },
    }

    payload = {
        "schema_version":
            "SMS01-model-runtime-v1.0",
        "experiment_family":
            sms.SCREEN_ID,
        "experiment_id":
            experiment_id,
        "training_source_commit":
            "source",
        "execution_commit":
            "execution",
        "model_contract":
            preflight["model_contract"],
        "initialization_audit": {
            "initialization":
                "pretrained",
            "official_checkpoint_loaded":
                True,
            "transferable_state_items":
                493,
        },
        "test_access":
            "ALLOWED",
    }

    path = (
        tmp_path
        / "SMS01_MODEL_RUNTIME.json"
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="original model runtime drift",
    ):
        smsr._verify_original_sms_runtime(
            experiment_id=experiment_id,
            path=path,
            original_preflight=preflight,
            source_commit="source",
            execution_commit="execution",
        )


def test_trainer_uses_indexed_sms_resume_runtime_filename():
    path = (
        sms.ROOT
        / "ultralytics/research/"
        "single_module_trainer.py"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert '"resume_session_index"' in text
    assert (
        '"SMS01_RESUME_MODEL_RUNTIME_"'
        in text
    )
    assert (
        'f"{session_index:03d}.json"'
        in text
    )
    assert (
        '"SMS01_RESUME_MODEL_RUNTIME.json"'
        not in text
    )

    assert '"resume_sessions"' in text

    assert (
        'payload["resume_session_index"]'
        in text
    )


def test_resume_runner_has_no_scientific_override_cli():
    text = Path(
        smsr.__file__
    ).read_text(
        encoding="utf-8"
    )

    for flag in (
        "--epochs",
        "--imgsz",
        "--batch",
        "--optimizer",
        "--lr0",
        "--seed",
        "--data",
        "--device",
    ):
        assert flag not in text

    assert "--experiment-id" in text
    assert "--mode" in text
    assert "preflight-only" in text
    assert "execute" in text


def test_resume_execute_uses_single_module_trainer():
    text = Path(
        smsr.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "GovernedSingleModuleTrainer"
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
    assert "RESEMA_SCREEN_FAMILY" in text


def test_sms_workflow_watches_resume_runner():
    workflow = (
        sms.ROOT
        / ".github/workflows/"
        "single-module-screen-01.yml"
    )

    text = workflow.read_text(
        encoding="utf-8"
    )

    assert (
        "research/runtime/"
        "single_module_resume_runner.py"
        in text
    )


def _write_valid_prior_resume_session(
    run_dir: Path,
    *,
    experiment_id: str,
    source_commit: str,
    execution_commit: str,
    model_contract: dict,
    original_runtime_manifest_sha256: str,
    original_pretrain_preflight_sha256: str,
    original_sms_model_runtime_sha256: str,
) -> None:
    sessions = (
        run_dir
        / "governance/resume_sessions"
    )
    sessions.mkdir(
        parents=True,
        exist_ok=True,
    )

    args_path = (
        sessions
        / "ARGS_BEFORE_RESUME_001.yaml"
    )
    args_path.write_text(
        "seed: 42\n",
        encoding="utf-8",
    )

    audit = {
        "initialization":
            "resume",
        "original_initialization":
            "pretrained",
        "resume_checkpoint_loaded":
            True,
        "resume_checkpoint_state_items":
            int(
                model_contract[
                    "expected_state_items"
                ]
            ),
        "resume_checkpoint_parameter_count":
            int(
                model_contract[
                    "expected_parameters"
                ]
            ),
        "resume_checkpoint_all_tensors_loaded_exactly":
            True,
    }

    parent_checkpoint = {
        "epoch_zero_based": 4,
        "completed_epochs": 5,
        "next_epoch_one_based": 6,
        "sha256": "d" * 64,
    }

    preflight = {
        "schema_version":
            "SMS01-resume-preflight-v1.0",
        "experiment_family":
            sms.SCREEN_ID,
        "experiment_id":
            experiment_id,
        "resume_session_index":
            1,
        "training_source_commit":
            source_commit,
        "execution_commit":
            execution_commit,
        "data_binding":
            "DATA01:B-ORG:v1",
        "initialization":
            "pretrained",
        "model_contract":
            model_contract,
        "test_access": {
            "runtime_yaml_contains_test":
                False,
            "test_directory_scan_pruned":
                True,
            "test_predictions":
                False,
            "test_metrics":
                False,
            "test_error_analysis":
                False,
        },
        "copy_verification": {
            "exact_file_inventory_match":
                True,
        },
        "archived_args_before_resume": {
            "sha256":
                sms.br.sha256_file(
                    args_path
                ),
        },
        "original_runtime_manifest_sha256":
            original_runtime_manifest_sha256,
        "original_pretrain_preflight_sha256":
            original_pretrain_preflight_sha256,
        "original_sms_model_runtime_sha256":
            original_sms_model_runtime_sha256,
        "parent_checkpoint":
            parent_checkpoint,
        "total_epochs":
            100,
    }

    preflight_path = (
        sessions
        / "RESUME_PREFLIGHT_001.json"
    )
    preflight_path.write_text(
        json.dumps(
            preflight,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    runtime = {
        "schema_version":
            "RESUME01-runtime-session-v1.0",
        "experiment_id":
            experiment_id,
        "resume_session_index":
            1,
        "training_source_commit":
            source_commit,
        "execution_commit":
            execution_commit,
        "data_binding":
            "DATA01:B-ORG:v1",
        "original_initialization":
            "pretrained",
        "test_split_present":
            False,
        "resume_preflight_manifest": {
            "sha256":
                sms.br.sha256_file(
                    preflight_path
                ),
        },
        "original_runtime_manifest": {
            "sha256":
                original_runtime_manifest_sha256,
        },
        "parent_checkpoint":
            parent_checkpoint,
        "restored_start_epoch_zero_based":
            5,
        "next_epoch_one_based":
            6,
        "total_epochs":
            100,
        "resume_model_audit":
            audit,
    }

    (
        sessions
        / "RESUME_RUNTIME_001.json"
    ).write_text(
        json.dumps(
            runtime,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    sms_runtime = {
        "schema_version":
            "SMS01-model-runtime-v1.0",
        "experiment_family":
            sms.SCREEN_ID,
        "experiment_id":
            experiment_id,
        "resume_session_index":
            1,
        "training_source_commit":
            source_commit,
        "execution_commit":
            execution_commit,
        "model_contract":
            model_contract,
        "initialization_audit":
            audit,
        "test_access":
            "NONE",
    }

    (
        sessions
        / "SMS01_RESUME_MODEL_RUNTIME_001.json"
    ).write_text(
        json.dumps(
            sms_runtime,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def test_prior_resume_session_contents_are_verified(
    monkeypatch,
    tmp_path,
):
    experiment_id = (
        "BORG-PT-S42-DYSAMPLE-E100"
    )

    model_contract = {
        "expected_state_items": 505,
        "expected_parameters": 9_455_915,
    }

    source_commit = "source"
    execution_commit = "execution"
    original_runtime_sha = "a" * 64
    original_preflight_sha = "b" * 64
    original_sms_sha = "c" * 64

    monkeypatch.setattr(
        smsr.base_resume,
        "_existing_resume_indices",
        lambda _run: [1],
    )

    _write_valid_prior_resume_session(
        tmp_path,
        experiment_id=experiment_id,
        source_commit=source_commit,
        execution_commit=execution_commit,
        model_contract=model_contract,
        original_runtime_manifest_sha256=
            original_runtime_sha,
        original_pretrain_preflight_sha256=
            original_preflight_sha,
        original_sms_model_runtime_sha256=
            original_sms_sha,
    )

    assert (
        smsr._verify_prior_resume_sessions(
            tmp_path,
            experiment_id=experiment_id,
            source_commit=source_commit,
            execution_commit=execution_commit,
            data_binding="DATA01:B-ORG:v1",
            model_contract=model_contract,
            original_runtime_manifest_sha256=
                original_runtime_sha,
            original_pretrain_preflight_sha256=
                original_preflight_sha,
            original_sms_model_runtime_sha256=
                original_sms_sha,
        )
        == [1]
    )

    sms_runtime = (
        tmp_path
        / "governance/resume_sessions/"
        "SMS01_RESUME_MODEL_RUNTIME_001.json"
    )

    sms_runtime.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="prior resume model runtime drift",
    ):
        smsr._verify_prior_resume_sessions(
            tmp_path,
            experiment_id=experiment_id,
            source_commit=source_commit,
            execution_commit=execution_commit,
            data_binding="DATA01:B-ORG:v1",
            model_contract=model_contract,
            original_runtime_manifest_sha256=
                original_runtime_sha,
            original_pretrain_preflight_sha256=
                original_preflight_sha,
            original_sms_model_runtime_sha256=
                original_sms_sha,
        )
