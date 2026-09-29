from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runtime import single_module_runner as sms


def _fake_clean_git(*args: str, **kwargs) -> str:
    if args == ("branch", "--show-current"):
        return sms.EXPECTED_BRANCH

    if args == ("rev-parse", "HEAD"):
        return "fake-execution-head"

    if args == (
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        return ""

    raise AssertionError(
        f"Unexpected git call in hostile test: {args}"
    )


def _authorized_row(
    experiment_id: str,
    source_commit: str = "fake-source-commit",
):
    _fields, rows = sms.read_registry()

    row = copy.deepcopy(
        next(
            r
            for r in rows
            if r["experiment_id"] == experiment_id
        )
    )

    row["status"] = "AUTHORIZED"
    row["source_commit"] = source_commit
    row["authorization_commit"] = ""

    return row


def _authorization_payload(
    source_commit: str,
):
    return {
        "schema_version":
            "SMS01-authorization-v1.0",
        "status":
            "AUTHORIZED",
        "branch":
            sms.EXPECTED_BRANCH,
        "source_commit":
            source_commit,
        "training_authorized":
            True,
        "test_access":
            "NONE",
        "experiment_ids":
            list(sms.EXPERIMENT_IDS),
    }


def test_unregistered_execution_fails_before_runtime_or_dataset_access(
    monkeypatch,
    tmp_path,
):
    """
    Registration alone must never reach runtime, dataset discovery,
    checkpoint discovery or GPU execution.
    """

    monkeypatch.setattr(
        sms.br,
        "git",
        _fake_clean_git,
    )

    # Make this hostile test independent of the live registry
    # authorization state. It must explicitly simulate the
    # registration-only condition both before and after the
    # later authorization transaction.
    registration_only_row = copy.deepcopy(
        sms.load_row(
            "BORG-PT-S42-DYSAMPLE-E100"
        )
    )

    registration_only_row["status"] = (
        "REGISTERED_NOT_AUTHORIZED"
    )
    registration_only_row["source_commit"] = ""
    registration_only_row["authorization_commit"] = ""

    def registration_only_load_row(experiment_id):
        assert experiment_id == (
            "BORG-PT-S42-DYSAMPLE-E100"
        )

        return copy.deepcopy(
            registration_only_row
        )

    monkeypatch.setattr(
        sms,
        "load_row",
        registration_only_load_row,
    )

    reached = {
        "runtime": False,
        "membership": False,
        "checkpoint": False,
    }

    def forbidden_runtime(*args, **kwargs):
        reached["runtime"] = True
        raise AssertionError(
            "Runtime verification must not be reached "
            "before authorization."
        )

    def forbidden_membership(*args, **kwargs):
        reached["membership"] = True
        raise AssertionError(
            "Dataset discovery must not be reached "
            "before authorization."
        )

    def forbidden_checkpoint(*args, **kwargs):
        reached["checkpoint"] = True
        raise AssertionError(
            "Checkpoint discovery must not be reached "
            "before authorization."
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

    monkeypatch.setattr(
        sms.br,
        "discover_checkpoint",
        forbidden_checkpoint,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="source_commit is blank",
    ):
        sms.perform_preflight(
            "BORG-PT-S42-DYSAMPLE-E100",
            input_root=tmp_path,
            work_root=tmp_path,
        )

    assert reached == {
        "runtime": False,
        "membership": False,
        "checkpoint": False,
    }


def test_dirty_worktree_blocks_execution_before_authorization(
    monkeypatch,
):
    row = _authorized_row(
        "BORG-PT-S42-SCCONV-EARLY-E100"
    )

    def dirty_git(*args: str, **kwargs):
        if args == ("branch", "--show-current"):
            return sms.EXPECTED_BRANCH

        if args == ("rev-parse", "HEAD"):
            return "fake-execution-head"

        if args == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return "?? unexpected.txt"

        raise AssertionError(args)

    monkeypatch.setattr(
        sms.br,
        "git",
        dirty_git,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="clean Git worktree",
    ):
        sms.verify_authorization(row)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("data_binding", "DATA01:B-AUG-HIST:v1"),
        ("initialization", "scratch"),
        ("seed", "41"),
        ("epochs", "200"),
        ("imgsz", "640"),
        ("batch", "8"),
        ("optimizer", "AdamW"),
        ("test_access", "ALLOWED"),
    ],
)
def test_registry_scientific_drift_fails_closed(
    monkeypatch,
    field,
    bad_value,
):
    fields, rows = sms.read_registry()

    hostile_rows = copy.deepcopy(rows)

    hostile_rows[0][field] = bad_value

    monkeypatch.setattr(
        sms,
        "read_registry",
        lambda: (
            copy.deepcopy(fields),
            copy.deepcopy(hostile_rows),
        ),
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="registry drift",
    ):
        sms.load_row(
            hostile_rows[0]["experiment_id"]
        )


def test_registry_model_yaml_substitution_fails_closed(
    monkeypatch,
):
    fields, rows = sms.read_registry()

    hostile_rows = copy.deepcopy(rows)

    hostile_rows[0]["model_yaml"] = (
        "ultralytics/cfg/models/11/yolo11s.yaml"
    )

    monkeypatch.setattr(
        sms,
        "read_registry",
        lambda: (
            copy.deepcopy(fields),
            copy.deepcopy(hostile_rows),
        ),
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="model-YAML binding drift",
    ):
        sms.load_row(
            hostile_rows[0]["experiment_id"]
        )


def test_registry_parameter_substitution_fails_closed(
    monkeypatch,
):
    fields, rows = sms.read_registry()

    hostile_rows = copy.deepcopy(rows)

    hostile_rows[0]["expected_parameters"] = "9431275"

    monkeypatch.setattr(
        sms,
        "read_registry",
        lambda: (
            copy.deepcopy(fields),
            copy.deepcopy(hostile_rows),
        ),
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="parameter registry drift",
    ):
        sms.load_row(
            hostile_rows[0]["experiment_id"]
        )


def test_contract_combination_authorization_drift_fails_closed(
    monkeypatch,
):
    hostile = json.loads(
        sms.CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    hostile[
        "advancement_rules"
    ][
        "combination_training_authorized"
    ] = True

    monkeypatch.setattr(
        sms.br,
        "load_json",
        lambda _path: copy.deepcopy(hostile),
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="combinations unexpectedly authorized",
    ):
        sms.load_contract()


def test_contract_test_access_drift_fails_closed(
    monkeypatch,
):
    hostile = json.loads(
        sms.CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    hostile[
        "development_condition"
    ][
        "test_access"
    ] = "ALLOWED"

    monkeypatch.setattr(
        sms.br,
        "load_json",
        lambda _path: copy.deepcopy(hostile),
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="contract condition drift",
    ):
        sms.load_contract()


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("epochs", 200),
        ("patience", 50),
        ("imgsz", 640),
        ("batch_global", 8),
        ("optimizer", "AdamW"),
        ("seed", 7),
        ("cos_lr", False),
        ("deterministic", False),
    ],
)
def test_frozen_training_recipe_drift_fails_closed(
    monkeypatch,
    key,
    bad_value,
):
    hostile = sms.br.load_yaml(
        sms.TRAINING_YAML
    )

    hostile = copy.deepcopy(hostile)

    hostile["training"][key] = bad_value

    monkeypatch.setattr(
        sms.br,
        "load_yaml",
        lambda _path: copy.deepcopy(hostile),
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="training drift",
    ):
        sms.effective_training()


def test_candidate_parameter_count_tamper_fails_before_checkpoint_use(
    monkeypatch,
    tmp_path,
):
    """
    Exercise the deeper model-parameter-count guard.

    A separate hostile test already proves that a registry/runtime
    parameter mismatch is rejected by load_row(). Here we deliberately
    keep the simulated registry consistent with the tampered runtime
    expectation so execution reaches candidate_model_audit() and must
    then reject the actual instantiated model parameter count before
    opening the checkpoint.
    """

    experiment_id = (
        "BORG-PT-S42-CANONICAL-EMA-E100"
    )

    # Capture the legitimate row BEFORE modifying the runtime candidate
    # contract, otherwise load_row() correctly rejects the mismatch one
    # layer earlier as "parameter registry drift."
    hostile_row = copy.deepcopy(
        sms.load_row(experiment_id)
    )

    spec = sms.CANDIDATES[
        experiment_id
    ]

    tampered_expected_parameters = (
        spec["expected_parameters"] + 1
    )

    monkeypatch.setitem(
        spec,
        "expected_parameters",
        tampered_expected_parameters,
    )

    # Simulate an attacker changing both the runtime candidate table
    # and registry consistently. This bypasses only the earlier registry
    # equality guard so the deeper model identity guard is exercised.
    hostile_row["expected_parameters"] = str(
        tampered_expected_parameters
    )

    def hostile_load_row(requested_id):
        assert requested_id == experiment_id
        return copy.deepcopy(hostile_row)

    monkeypatch.setattr(
        sms,
        "load_row",
        hostile_load_row,
    )

    checkpoint_path = (
        tmp_path
        / "checkpoint-must-not-be-opened.pt"
    )

    assert not checkpoint_path.exists()

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="candidate parameter count drift",
    ):
        sms.candidate_model_audit(
            experiment_id=experiment_id,
            checkpoint=checkpoint_path,
            init_lock={},
            contract=sms.load_contract(),
        )

    # Confirms rejection happened before checkpoint access.
    assert not checkpoint_path.exists()


def test_candidate_yaml_hash_tamper_fails_before_checkpoint_use(
    monkeypatch,
    tmp_path,
):
    experiment_id = (
        "BORG-PT-S42-DYSAMPLE-E100"
    )

    model_path = (
        sms.ROOT
        / sms.CANDIDATES[
            experiment_id
        ]["model_yaml"]
    ).resolve()

    original_sha = sms.br.sha256_file

    def hostile_sha(path):
        resolved = Path(path).resolve()

        if resolved == model_path:
            return "0" * 64

        return original_sha(path)

    monkeypatch.setattr(
        sms.br,
        "sha256_file",
        hostile_sha,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="candidate YAML hash drift",
    ):
        sms.candidate_model_audit(
            experiment_id=experiment_id,
            checkpoint=(
                tmp_path
                / "checkpoint-must-not-be-opened.pt"
            ),
            init_lock={},
            contract=sms.load_contract(),
        )


def test_missing_authorization_record_blocks_execution(
    monkeypatch,
    tmp_path,
):
    source_commit = "fake-source-commit"

    row = _authorized_row(
        "BORG-PT-S42-SCCONV-4STAGE-E100",
        source_commit,
    )

    monkeypatch.setattr(
        sms.br,
        "git",
        _fake_clean_git,
    )

    monkeypatch.setattr(
        sms.subprocess,
        "run",
        lambda *args, **kwargs:
            SimpleNamespace(
                returncode=0
            ),
    )

    monkeypatch.setattr(
        sms,
        "_expected_authorized_registry",
        lambda _source: None,
    )

    monkeypatch.setattr(
        sms,
        "_expected_authorized_contract",
        lambda _source: None,
    )

    monkeypatch.setattr(
        sms,
        "AUTHORIZATION",
        tmp_path / "missing-auth.json",
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="authorization record is missing",
    ):
        sms.verify_authorization(row)


def test_post_freeze_scientific_source_change_is_rejected(
    monkeypatch,
    tmp_path,
):
    source_commit = "fake-source-commit"
    execution_head = "fake-execution-head"

    row = _authorized_row(
        "BORG-PT-S42-DYSAMPLE-E100",
        source_commit,
    )

    auth_path = (
        tmp_path
        / "SINGLE_MODULE_SCREEN_01_AUTHORIZATION.json"
    )

    auth_path.write_text(
        json.dumps(
            _authorization_payload(
                source_commit
            ),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    def fake_git(*args: str, **kwargs):
        if args == ("branch", "--show-current"):
            return sms.EXPECTED_BRANCH

        if args == ("rev-parse", "HEAD"):
            return execution_head

        if args == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return ""

        if args[:2] == (
            "diff",
            "--name-only",
        ):
            return "ultralytics/nn/tasks.py"

        raise AssertionError(args)

    monkeypatch.setattr(
        sms.br,
        "git",
        fake_git,
    )

    monkeypatch.setattr(
        sms.subprocess,
        "run",
        lambda *args, **kwargs:
            SimpleNamespace(
                returncode=0
            ),
    )

    monkeypatch.setattr(
        sms,
        "_expected_authorized_registry",
        lambda _source: None,
    )

    monkeypatch.setattr(
        sms,
        "_expected_authorized_contract",
        lambda _source: None,
    )

    monkeypatch.setattr(
        sms,
        "AUTHORIZATION",
        auth_path,
    )

    with pytest.raises(
        sms.ScreenGovernanceError,
        match="Scientific SMS-01 source changed",
    ):
        sms.verify_authorization(row)


def test_only_authorization_delta_is_allowed_after_source_freeze(
    monkeypatch,
    tmp_path,
):
    source_commit = "fake-source-commit"
    execution_head = "fake-execution-head"

    row = _authorized_row(
        "BORG-PT-S42-SCCONV-EARLY-E100",
        source_commit,
    )

    auth_path = (
        tmp_path
        / "SINGLE_MODULE_SCREEN_01_AUTHORIZATION.json"
    )

    auth_path.write_text(
        json.dumps(
            _authorization_payload(
                source_commit
            ),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    allowed = "\n".join(
        [
            "research/05_experiments/"
            "SINGLE_MODULE_SCREEN_01_EXPERIMENTS.csv",

            "research/05_experiments/"
            "SINGLE_MODULE_SCREEN_01_CONTRACT.json",

            "research/05_experiments/"
            "SINGLE_MODULE_SCREEN_01_AUTHORIZATION.json",
        ]
    )

    def fake_git(*args: str, **kwargs):
        if args == ("branch", "--show-current"):
            return sms.EXPECTED_BRANCH

        if args == ("rev-parse", "HEAD"):
            return execution_head

        if args == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return ""

        if args[:2] == (
            "diff",
            "--name-only",
        ):
            return allowed

        raise AssertionError(args)

    monkeypatch.setattr(
        sms.br,
        "git",
        fake_git,
    )

    monkeypatch.setattr(
        sms.subprocess,
        "run",
        lambda *args, **kwargs:
            SimpleNamespace(
                returncode=0
            ),
    )

    monkeypatch.setattr(
        sms,
        "_expected_authorized_registry",
        lambda _source: None,
    )

    monkeypatch.setattr(
        sms,
        "_expected_authorized_contract",
        lambda _source: None,
    )

    monkeypatch.setattr(
        sms,
        "AUTHORIZATION",
        auth_path,
    )

    observed_source, observed_head = (
        sms.verify_authorization(row)
    )

    assert observed_source == source_commit
    assert observed_head == execution_head


def test_candidate_cli_surface_contains_only_registered_four():
    assert tuple(
        sms.EXPERIMENT_IDS
    ) == (
        "BORG-PT-S42-SCCONV-EARLY-E100",
        "BORG-PT-S42-SCCONV-4STAGE-E100",
        "BORG-PT-S42-DYSAMPLE-E100",
        "BORG-PT-S42-CANONICAL-EMA-E100",
    )


def test_output_root_isolated_from_frozen_baseline_outputs():
    assert (
        sms.OUTPUT_ROOT.name
        == "ResEMA_single_module_runs"
    )

    assert (
        sms.OUTPUT_ROOT.name
        != "ResEMA_baseline_runs"
    )

    assert (
        sms.OUTPUT_ROOT.name
        != "ResEMA_epoch_calibration_runs"
    )