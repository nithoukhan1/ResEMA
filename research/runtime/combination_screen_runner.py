from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.runtime import baseline_runner as br
from ultralytics.nn.tasks import (
    DetectionModel,
    torch_safe_load,
)


SCREEN_ID = "COMBINATION_SCREEN_01"

EXPECTED_BRANCH = (
    "research/combination-screen-01"
)

EXPERIMENT_ID = (
    "BORG-PT-S42-SCCONV-EARLY-"
    "CANONICAL-EMA-E100"
)

EXPERIMENT_IDS = (
    EXPERIMENT_ID,
)


REGISTRY = (
    ROOT
    / "research/05_experiments/"
    "COMBINATION_SCREEN_01_EXPERIMENTS.csv"
)

CONTRACT = (
    ROOT
    / "research/05_experiments/"
    "COMBINATION_SCREEN_01_CONTRACT.json"
)

TECHNICAL_VERIFICATION = (
    ROOT
    / "research/05_experiments/"
    "COMBINATION_SCREEN_01_TECHNICAL_VERIFICATION.json"
)

AUTHORIZATION = (
    ROOT
    / "research/05_experiments/"
    "COMBINATION_SCREEN_01_AUTHORIZATION.json"
)

TRAINING_YAML = (
    ROOT
    / "research/05_experiments/TRAINING.yaml"
)

INIT_LOCK = (
    ROOT
    / "research/01_provenance/"
    "INIT01_INITIALIZATION_LOCK.json"
)

DATA_BINDINGS = (
    ROOT
    / "research/04_data/manifests/"
    "DATA01_ACTIVE_DATASET_BINDINGS.json"
)


BASELINE_YAML = (
    ROOT
    / "ultralytics/cfg/models/11/yolo11s.yaml"
)

MODEL_RELATIVE = (
    "ultralytics/cfg/models/11/"
    "yolo11s-scconv-early-canonical-ema-v1.yaml"
)

MODEL_YAML = (
    ROOT
    / MODEL_RELATIVE
)




TECHNICAL_VERIFICATION_SHA256 = (
    "0295a6e90bbe6856517c5e723fbe58901f87d405"
    "743a8b6ecf334f335191fe29"
)

MODEL_YAML_SHA256 = (
    "6f9547db340b77b3947698fcd911a70d3cde6332"
    "8e1190751bc135c11443df14"
)

CANDIDATES = {
    EXPERIMENT_ID: {
        "model_yaml":
            MODEL_RELATIVE,

        "expected_parameters":
            9_574_241,

        "expected_state_items":
            565,

        "expected_new_state_items":
            66,

        "technical_verification_sha256":
            TECHNICAL_VERIFICATION_SHA256,
    }
}


OUTPUT_ROOT = Path(
    "/kaggle/working/ResEMA_combination_screen_runs"
)

PREFLIGHT_ROOT_NAME = (
    "ResEMA_combination_screen_preflight"
)


BASELINE_NATIVE_NONTRANSFER_KEYS = [
    "model.23.cv3.0.2.weight",
    "model.23.cv3.0.2.bias",
    "model.23.cv3.1.2.weight",
    "model.23.cv3.1.2.bias",
    "model.23.cv3.2.2.weight",
    "model.23.cv3.2.2.bias",
]


class CombinationGovernanceError(
    br.GovernanceError
):
    pass


def _read_registry_text(
    text: str,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    reader = csv.DictReader(
        io.StringIO(
            text.lstrip("\ufeff")
        )
    )

    fields = list(
        reader.fieldnames or []
    )

    rows = list(reader)

    required = {
        "experiment_id",
        "display_name",
        "model_yaml",
        "data_binding",
        "initialization",
        "seed",
        "epochs",
        "imgsz",
        "batch",
        "optimizer",
        "primary_metric",
        "status",
        "reference_experiment",
        "promotion_threshold_map50_95",
        "parameter_hypothesis",
        "state_item_hypothesis",
        "new_state_item_hypothesis",
        "checkpoint_sha256",
        "test_access",
        "source_commit",
        "authorization_commit",
        "decision",
    }

    if not required.issubset(
        set(fields)
    ):
        raise CombinationGovernanceError(
            "COMB-01 registry schema incomplete."
        )

    if len(rows) != 1:
        raise CombinationGovernanceError(
            "COMB-01 registry must contain "
            "exactly one experiment."
        )

    if (
        rows[0].get("experiment_id")
        != EXPERIMENT_ID
    ):
        raise CombinationGovernanceError(
            "COMB-01 experiment ID drift."
        )

    return fields, rows


def read_registry():
    return _read_registry_text(
        REGISTRY.read_text(
            encoding="utf-8-sig",
            errors="strict",
        )
    )


def load_row(
    experiment_id: str,
) -> dict[str, str]:
    if experiment_id != EXPERIMENT_ID:
        raise CombinationGovernanceError(
            "Unknown COMB-01 experiment ID."
        )

    _fields, rows = read_registry()

    row = rows[0]

    fixed = {
        "model_yaml":
            MODEL_RELATIVE,
        "data_binding":
            "DATA01:B-ORG:v1",
        "initialization":
            "pretrained",
        "seed":
            "42",
        "epochs":
            "100",
        "imgsz":
            "1024",
        "batch":
            "16",
        "optimizer":
            "SGD",
        "primary_metric":
            "val_mAP50-95",
        "reference_experiment":
            "BORG-PT-S42-SCCONV-EARLY-E100",
        "promotion_threshold_map50_95":
            "0.43130",
        "parameter_hypothesis":
            "9574241",
        "state_item_hypothesis":
            "565",
        "new_state_item_hypothesis":
            "66",
        "checkpoint_sha256":
            "85a76fe86dd8afe384648546b56a7a78580c7cb7"
            "b404fc595f97969322d502d5",
        "test_access":
            "NONE",
    }

    for key, expected in fixed.items():

        if row.get(key) != expected:

            raise CombinationGovernanceError(
                "COMB-01 registry scientific "
                f"drift: {key}="
                f"{row.get(key)!r} "
                f"!= {expected!r}"
            )

    return row


def load_contract() -> dict[str, Any]:
    data = br.load_json(
        CONTRACT
    )

    if (
        data.get("experiment_family")
        != SCREEN_ID
    ):
        raise CombinationGovernanceError(
            "COMB-01 contract family drift."
        )

    if (
        data.get("branch")
        != EXPECTED_BRANCH
    ):
        raise CombinationGovernanceError(
            "COMB-01 contract branch drift."
        )

    candidate = data.get(
        "candidate",
        {},
    )

    if (
        candidate.get("experiment_id")
        != EXPERIMENT_ID
    ):
        raise CombinationGovernanceError(
            "COMB-01 candidate ID drift."
        )

    if (
        candidate.get("model_yaml")
        != MODEL_RELATIVE
    ):
        raise CombinationGovernanceError(
            "COMB-01 candidate YAML binding drift."
        )

    if (
        candidate.get("model_yaml_sha256")
        != MODEL_YAML_SHA256
    ):
        raise CombinationGovernanceError(
            "COMB-01 candidate YAML SHA drift."
        )

    condition = data.get(
        "development_condition",
        {},
    )

    required_condition = {
        "split": "B",
        "train_data": "original",
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "pretrained",
        "checkpoint": "yolo11s.pt",
        "seed": 42,
        "epochs": 100,
        "imgsz": 1024,
        "batch_global": 16,
        "optimizer": "SGD",
        "primary_metric": "validation_mAP50-95",
        "selection_split": "val",
        "test_access": "NONE",
    }

    for key, expected in (
        required_condition.items()
    ):

        if condition.get(key) != expected:

            raise CombinationGovernanceError(
                "COMB-01 development condition "
                f"drift: {key}"
            )

    firewall = data.get(
        "firewall",
        {},
    )

    if firewall.get(
        "test_access"
    ) != "NONE":
        raise CombinationGovernanceError(
            "COMB-01 test firewall drift."
        )

    return data


def load_technical_verification() -> dict[str, Any]:
    if (
        br.sha256_file(
            TECHNICAL_VERIFICATION
        )
        != TECHNICAL_VERIFICATION_SHA256
    ):
        raise CombinationGovernanceError(
            "COMB-01 technical verification "
            "record SHA drift."
        )

    data = br.load_json(
        TECHNICAL_VERIFICATION
    )

    required = {
        "schema_version":
            "COMB01-technical-verification-v1.0",
        "status":
            "PASS",
        "experiment_family":
            SCREEN_ID,
        "experiment_id":
            EXPERIMENT_ID,
        "branch":
            EXPECTED_BRANCH,
    }

    for key, expected in required.items():

        if data.get(key) != expected:

            raise CombinationGovernanceError(
                "COMB-01 technical verification "
                f"drift: {key}"
            )

    model = data.get(
        "model_identity",
        {},
    )

    expected_model = {
        "model_yaml":
            MODEL_RELATIVE,
        "model_yaml_sha256":
            MODEL_YAML_SHA256,
        "parameters":
            9_574_241,
        "state_dict_items":
            565,
        "new_state_items":
            66,
        "scconv_new_state_items":
            38,
        "canonical_ema_new_state_items":
            28,
        "zero_gates":
            6,
    }

    for key, expected in (
        expected_model.items()
    ):

        if model.get(key) != expected:

            raise CombinationGovernanceError(
                "COMB-01 verified model identity "
                f"drift: {key}"
            )

    transfer = data.get(
        "official_pretrained_transfer",
        {},
    )

    if int(
        transfer.get(
            "transferable_source_items",
            -1,
        )
    ) != 493:
        raise CombinationGovernanceError(
            "COMB-01 verified transfer "
            "count drift."
        )

    if int(
        transfer.get(
            "target_nontransfer_items",
            -1,
        )
    ) != 72:
        raise CombinationGovernanceError(
            "COMB-01 verified nontransfer "
            "count drift."
        )

    conclusion = data.get(
        "technical_conclusion",
        {},
    )

    if (
        conclusion.get(
            "technical_architecture_validation_closed"
        )
        is not True
    ):
        raise CombinationGovernanceError(
            "COMB-01 technical validation "
            "is not closed."
        )

    if (
        conclusion.get(
            "execution_layer_implementation_may_proceed"
        )
        is not True
    ):
        raise CombinationGovernanceError(
            "COMB-01 execution-layer "
            "implementation remains locked."
        )

    if (
        conclusion.get(
            "gpu_training_authorized"
        )
        is not False
    ):
        raise CombinationGovernanceError(
            "Technical verification must not "
            "self-authorize GPU training."
        )

    if (
        data.get(
            "firewall",
            {},
        ).get("test_access")
        != "NONE"
    ):
        raise CombinationGovernanceError(
            "COMB-01 technical test firewall drift."
        )

    return data


def _expected_authorized_registry(
    source_commit: str,
) -> None:
    frozen_text = br.git(
        "show",
        f"{source_commit}:"
        "research/05_experiments/"
        "COMBINATION_SCREEN_01_EXPERIMENTS.csv",
    )

    frozen_fields, frozen_rows = (
        _read_registry_text(
            frozen_text
        )
    )

    current_fields, current_rows = (
        read_registry()
    )

    if current_fields != frozen_fields:
        raise CombinationGovernanceError(
            "COMB-01 registry schema changed "
            "after source freeze."
        )

    frozen = frozen_rows[0]
    current = current_rows[0]

    if (
        frozen["status"]
        != "REGISTERED_NOT_AUTHORIZED"
    ):
        raise CombinationGovernanceError(
            "Frozen COMB-01 registry was not "
            "registration-only."
        )

    if frozen["source_commit"].strip():
        raise CombinationGovernanceError(
            "Frozen COMB-01 source_commit "
            "must be blank."
        )

    if (
        frozen["authorization_commit"].strip()
    ):
        raise CombinationGovernanceError(
            "Frozen COMB-01 authorization_commit "
            "must be blank."
        )

    if (
        current["status"]
        != "AUTHORIZED"
    ):
        raise CombinationGovernanceError(
            "COMB-01 registry is not authorized."
        )

    if (
        current["source_commit"].strip()
        != source_commit
    ):
        raise CombinationGovernanceError(
            "COMB-01 source binding mismatch."
        )

    if (
        current[
            "authorization_commit"
        ].strip()
    ):
        raise CombinationGovernanceError(
            "COMB-01 authorization_commit is "
            "non-self-referential and must "
            "remain blank."
        )

    for field in frozen_fields:

        if field in {
            "status",
            "source_commit",
        }:
            continue

        if (
            current.get(field)
            != frozen.get(field)
        ):
            raise CombinationGovernanceError(
                "COMB-01 registry changed outside "
                "authorization delta: "
                f"{field}"
            )


def _expected_authorized_contract(
    source_commit: str,
) -> None:
    frozen = json.loads(
        br.git(
            "show",
            f"{source_commit}:"
            "research/05_experiments/"
            "COMBINATION_SCREEN_01_CONTRACT.json",
        )
    )

    current = load_contract()

    expected = copy.deepcopy(
        frozen
    )

    if (
        expected.get("status")
        != "REGISTERED_NOT_AUTHORIZED"
    ):
        raise CombinationGovernanceError(
            "Frozen COMB-01 contract was not "
            "registration-only."
        )

    if (
        expected.get(
            "firewall",
            {},
        ).get(
            "training_authorized"
        )
        is not False
    ):
        raise CombinationGovernanceError(
            "Frozen COMB-01 contract was "
            "unexpectedly authorized."
        )

    if (
        expected.get(
            "provenance",
            {},
        ).get(
            "scientific_source_commit"
        )
        is not None
    ):
        raise CombinationGovernanceError(
            "Frozen COMB-01 scientific source "
            "must be null."
        )

    expected["status"] = (
        "AUTHORIZED"
    )

    expected["firewall"][
        "training_authorized"
    ] = True

    expected["provenance"][
        "scientific_source_commit"
    ] = source_commit

    if current != expected:
        raise CombinationGovernanceError(
            "COMB-01 contract changed outside "
            "the permitted authorization delta."
        )


def verify_authorization(
    row: dict[str, str],
) -> tuple[str, str]:
    branch = br.git(
        "branch",
        "--show-current",
    )

    head = br.git(
        "rev-parse",
        "HEAD",
    )

    dirty = br.git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    if branch != EXPECTED_BRANCH:
        raise CombinationGovernanceError(
            f"Wrong COMB-01 branch: {branch!r}"
        )

    if dirty:
        raise CombinationGovernanceError(
            "COMB-01 execution requires a "
            "clean Git worktree."
        )

    source_commit = (
        row.get(
            "source_commit",
            ""
        ).strip()
    )

    if not source_commit:
        raise CombinationGovernanceError(
            "COMB-01 remains locked: "
            "source_commit is blank."
        )

    if (
        row.get("status")
        != "AUTHORIZED"
    ):
        raise CombinationGovernanceError(
            "COMB-01 remains locked: "
            "registry status is not AUTHORIZED."
        )

    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            source_commit,
            head,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if ancestor.returncode != 0:
        raise CombinationGovernanceError(
            "COMB-01 frozen source is not "
            "an ancestor of execution HEAD."
        )

    _expected_authorized_registry(
        source_commit
    )

    _expected_authorized_contract(
        source_commit
    )

    if not AUTHORIZATION.is_file():
        raise CombinationGovernanceError(
            "COMB-01 authorization record "
            "is missing."
        )

    authorization = br.load_json(
        AUTHORIZATION
    )

    required_auth = {
        "schema_version":
            "COMB01-authorization-v1.0",
        "status":
            "AUTHORIZED",
        "branch":
            EXPECTED_BRANCH,
        "experiment_id":
            EXPERIMENT_ID,
        "source_commit":
            source_commit,
        "training_authorized":
            True,
        "test_access":
            "NONE",
        "technical_verification_sha256":
            TECHNICAL_VERIFICATION_SHA256,
    }

    for key, expected in (
        required_auth.items()
    ):

        if (
            authorization.get(key)
            != expected
        ):
            raise CombinationGovernanceError(
                "COMB-01 authorization record "
                f"drift: {key}"
            )

    changed = set(
        filter(
            None,
            br.git(
                "diff",
                "--name-only",
                source_commit,
                head,
            ).splitlines(),
        )
    )

    allowed_authorization_delta = {
        "research/05_experiments/"
        "COMBINATION_SCREEN_01_EXPERIMENTS.csv",

        "research/05_experiments/"
        "COMBINATION_SCREEN_01_CONTRACT.json",

        "research/05_experiments/"
        "COMBINATION_SCREEN_01_AUTHORIZATION.json",
    }

    unexpected = (
        changed
        - allowed_authorization_delta
    )

    if unexpected:
        raise CombinationGovernanceError(
            "COMB-01 scientific source changed "
            "after source freeze:\n"
            + "\n".join(
                sorted(unexpected)
            )
        )

    return (
        source_commit,
        head,
    )


def effective_training() -> dict[str, Any]:
    training = br.load_yaml(
        TRAINING_YAML
    )

    if (
        training.get("status")
        != "FROZEN_BASELINE_RECIPE_V2"
    ):
        raise CombinationGovernanceError(
            "TRAINING.yaml status drift."
        )

    effective = copy.deepcopy(
        training
    )

    t = effective["training"]

    required = {
        "epochs": 100,
        "patience": 100,
        "imgsz": 1024,
        "batch_global": 16,
        "optimizer": "SGD",
        "seed": 42,
        "cos_lr": True,
        "deterministic": True,
    }

    for key, expected in (
        required.items()
    ):

        if t.get(key) != expected:
            raise CombinationGovernanceError(
                "COMB-01 training recipe drift: "
                f"{key}"
            )

    effective["output"][
        "project_dir"
    ] = str(
        OUTPUT_ROOT
    )

    return effective


def build_model(
    path: Path,
) -> DetectionModel:
    torch.manual_seed(42)

    return DetectionModel(
        str(path),
        ch=3,
        nc=9,
        verbose=False,
    )


def candidate_model_audit(
    *,
    checkpoint: Path,
    init_lock: dict[str, Any],
    technical: dict[str, Any],
) -> dict[str, Any]:
    if (
        br.sha256_file(
            MODEL_YAML
        )
        != MODEL_YAML_SHA256
    ):
        raise CombinationGovernanceError(
            "COMB-01 model YAML hash drift."
        )

    if (
        br.sha256_file(
            TECHNICAL_VERIFICATION
        )
        != TECHNICAL_VERIFICATION_SHA256
    ):
        raise CombinationGovernanceError(
            "COMB-01 technical verification "
            "SHA drift."
        )

    baseline = build_model(
        BASELINE_YAML
    )

    target = build_model(
        MODEL_YAML
    )

    baseline_state = (
        baseline.state_dict()
    )

    target_state = (
        target.state_dict()
    )

    parameters = sum(
        int(p.numel())
        for p in target.parameters()
    )

    verified_model = technical[
        "model_identity"
    ]

    if (
        parameters
        != int(
            verified_model[
                "parameters"
            ]
        )
        or parameters
        != 9_574_241
    ):
        raise CombinationGovernanceError(
            "COMB-01 parameter-count drift."
        )

    if (
        len(target_state)
        != int(
            verified_model[
                "state_dict_items"
            ]
        )
        or len(target_state) != 565
    ):
        raise CombinationGovernanceError(
            "COMB-01 state-item drift."
        )

    missing_native = [
        key
        for key, value
        in baseline_state.items()
        if (
            key not in target_state
            or
            target_state[key].shape
            != value.shape
        )
    ]

    if missing_native:
        raise CombinationGovernanceError(
            "COMB-01 native shared-state "
            "contract failed."
        )

    new_target_keys = sorted(
        set(target_state)
        - set(baseline_state)
    )

    if len(new_target_keys) != 66:
        raise CombinationGovernanceError(
            "COMB-01 new-state count drift."
        )

    unexpected_new = [
        key
        for key in new_target_keys
        if (
            ".sc_adapters." not in key
            and
            ".ema_adapter." not in key
        )
    ]

    if unexpected_new:
        raise CombinationGovernanceError(
            "COMB-01 unexpected new state keys."
        )

    expected_checkpoint_sha = (
        init_lock[
            "official_checkpoint"
        ][
            "sha256"
        ]
    )

    if (
        br.sha256_file(
            checkpoint
        )
        != expected_checkpoint_sha
    ):
        raise CombinationGovernanceError(
            "COMB-01 official checkpoint "
            "SHA drift."
        )

    ckpt, _ = torch_safe_load(
        str(checkpoint)
    )

    source_model = (
        ckpt.get("ema")
        or ckpt["model"]
    )

    source_state = (
        source_model
        .float()
        .state_dict()
    )

    transferable = {
        key: value
        for key, value
        in source_state.items()
        if (
            key in target_state
            and
            target_state[key].shape
            == value.shape
        )
    }

    if len(transferable) != 493:
        raise CombinationGovernanceError(
            "COMB-01 official checkpoint "
            "transfer-count drift."
        )

    native_nontransfer = set(
        init_lock[
            "nine_class_initialization_contract"
        ][
            "nontransferable_target_keys"
        ]
    )

    if (
        native_nontransfer
        != set(
            BASELINE_NATIVE_NONTRANSFER_KEYS
        )
    ):
        raise CombinationGovernanceError(
            "INIT-01 native nontransfer "
            "contract drift."
        )

    nontransferred_target = (
        set(target_state)
        - set(transferable)
    )

    expected_nontransferred = (
        native_nontransfer
        | set(new_target_keys)
    )

    if (
        nontransferred_target
        != expected_nontransferred
        or
        len(nontransferred_target)
        != 72
    ):
        raise CombinationGovernanceError(
            "COMB-01 target nontransfer "
            "partition drift."
        )

    initial_target = {
        key:
            value.detach()
            .cpu()
            .clone()
        for key, value
        in target_state.items()
    }

    result = target.load_state_dict(
        transferable,
        strict=False,
    )

    if result.unexpected_keys:
        raise CombinationGovernanceError(
            "COMB-01 unexpected checkpoint "
            "load keys."
        )

    loaded = target.state_dict()

    if not all(
        torch.equal(
            loaded[key],
            value,
        )
        for key, value
        in transferable.items()
    ):
        raise CombinationGovernanceError(
            "COMB-01 transferred tensor drift."
        )

    if not all(
        torch.equal(
            loaded[key].detach().cpu(),
            initial_target[key],
        )
        for key in expected_nontransferred
    ):
        raise CombinationGovernanceError(
            "COMB-01 unmatched seeded "
            "initialization was not preserved."
        )

    gates = [
        parameter
        for name, parameter
        in target.named_parameters()
        if (
            name.endswith(".alpha")
            and (
                ".sc_adapters." in name
                or ".ema_adapter." in name
            )
        )
    ]

    if (
        len(gates) != 6
        or not all(
            torch.equal(
                parameter.detach(),
                torch.zeros_like(
                    parameter.detach()
                ),
            )
            for parameter in gates
        )
    ):
        raise CombinationGovernanceError(
            "COMB-01 zero-gate contract drift."
        )

    return {
        "model_yaml":
            MODEL_RELATIVE,

        "model_yaml_sha256":
            MODEL_YAML_SHA256,

        "technical_verification_sha256":
            TECHNICAL_VERIFICATION_SHA256,

        "expected_parameters":
            parameters,

        "expected_state_items":
            len(target_state),

        "expected_transferable_source_items":
            len(transferable),

        "baseline_native_nontransfer_keys":
            sorted(native_nontransfer),

        "new_target_state_keys":
            new_target_keys,

        "expected_nontransferable_target_keys":
            sorted(
                expected_nontransferred
            ),

        "new_target_state_items":
            len(new_target_keys),

        "official_checkpoint_transfer_exact":
            True,

        "unmatched_seeded_initialization_preserved":
            True,

        "zero_gate_count":
            6,

        "zero_gate_initialization_exact":
            True,
    }


def perform_preflight(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
):
    row = load_row(
        experiment_id
    )

    load_contract()

    technical = (
        load_technical_verification()
    )

    training = (
        effective_training()
    )

    # ------------------------------------------------------------
    # Authorization MUST be verified before runtime environment,
    # dataset membership, checkpoint discovery, or model access.
    # ------------------------------------------------------------
    source_commit, execution_commit = (
        verify_authorization(
            row
        )
    )

    init_lock = br.load_json(
        INIT_LOCK
    )

    data_manifest = br.load_json(
        DATA_BINDINGS
    )

    runtime = (
        br.verify_runtime_environment(
            init_lock
        )
    )

    expectation = (
        br.binding_expectation(
            row,
            data_manifest,
        )
    )

    if (
        expectation["binding_id"]
        != "DATA01:B-ORG:v1"
    ):
        raise CombinationGovernanceError(
            "COMB-01 must use B-ORG."
        )

    train_images = (
        br.discover_membership_directory(
            input_root,
            kind="images",
            expected_count=
                expectation[
                    "train_count"
                ],
            expected_hash=
                expectation[
                    "train_hash"
                ],
        )
    )

    train_labels = (
        br.discover_membership_directory(
            input_root,
            kind="labels",
            expected_count=
                expectation[
                    "train_count"
                ],
            expected_hash=
                expectation[
                    "train_hash"
                ],
        )
    )

    val_images = (
        br.discover_membership_directory(
            input_root,
            kind="images",
            expected_count=
                expectation[
                    "validation_count"
                ],
            expected_hash=
                expectation[
                    "validation_hash"
                ],
        )
    )

    val_labels = (
        br.discover_membership_directory(
            input_root,
            kind="labels",
            expected_count=
                expectation[
                    "validation_count"
                ],
            expected_hash=
                expectation[
                    "validation_hash"
                ],
        )
    )

    br.verify_image_label_pair(
        train_images,
        train_labels,
    )

    br.verify_image_label_pair(
        val_images,
        val_labels,
    )

    checkpoint = (
        br.discover_checkpoint(
            input_root,
            init_lock,
        )
    )

    model_contract = (
        candidate_model_audit(
            checkpoint=checkpoint,
            init_lock=init_lock,
            technical=technical,
        )
    )

    preflight_dir = (
        work_root
        / PREFLIGHT_ROOT_NAME
        / experiment_id
    )

    preflight_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_yaml = (
        preflight_dir
        / "runtime_data_train_val_only.yaml"
    )

    runtime_yaml_sha = (
        br.write_runtime_data_yaml(
            runtime_yaml,
            train_images=train_images,
            validation_images=val_images,
        )
    )

    run_dir = (
        Path(
            training[
                "output"
            ][
                "project_dir"
            ]
        )
        / experiment_id
    )

    if run_dir.exists():
        raise CombinationGovernanceError(
            "COMB-01 fresh launch refuses "
            f"existing run directory: {run_dir}"
        )

    preflight = {
        "schema_version":
            "COMB01-preflight-v1.0",

        "experiment_family":
            SCREEN_ID,

        "experiment_id":
            experiment_id,

        "training_source_commit":
            source_commit,

        "execution_commit":
            execution_commit,

        "data_binding":
            row["data_binding"],

        "initialization":
            "pretrained",

        "model_contract":
            model_contract,

        "technical_verification": {
            "path":
                str(
                    TECHNICAL_VERIFICATION
                ),

            "sha256":
                TECHNICAL_VERIFICATION_SHA256,

            "status":
                "PASS",
        },

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

        "membership": {
            "train_images": {
                "path":
                    str(train_images),

                "count":
                    expectation[
                        "train_count"
                    ],

                "sha256_final_newline":
                    expectation[
                        "train_hash"
                    ],
            },

            "train_labels": {
                "path":
                    str(train_labels),

                "count":
                    expectation[
                        "train_count"
                    ],

                "sha256_final_newline":
                    expectation[
                        "train_hash"
                    ],
            },

            "validation_images": {
                "path":
                    str(val_images),

                "count":
                    expectation[
                        "validation_count"
                    ],

                "sha256_final_newline":
                    expectation[
                        "validation_hash"
                    ],
            },

            "validation_labels": {
                "path":
                    str(val_labels),

                "count":
                    expectation[
                        "validation_count"
                    ],

                "sha256_final_newline":
                    expectation[
                        "validation_hash"
                    ],
            },
        },

        "operational_expected": {
            "train_images":
                expectation[
                    "operational_train_images"
                ],

            "validation_images":
                expectation[
                    "operational_validation_images"
                ],
        },

        "runtime_data_yaml": {
            "path":
                str(runtime_yaml),

            "sha256":
                runtime_yaml_sha,

            "contains_test_key":
                False,
        },

        "initial_checkpoint": {
            "path":
                str(checkpoint),

            "sha256":
                init_lock[
                    "official_checkpoint"
                ][
                    "sha256"
                ],

            "bytes":
                init_lock[
                    "official_checkpoint"
                ][
                    "bytes"
                ],
        },

        "frozen_inputs": {
            "registry_sha256":
                br.sha256_file(
                    REGISTRY
                ),

            "contract_sha256":
                br.sha256_file(
                    CONTRACT
                ),

            "technical_verification_sha256":
                br.sha256_file(
                    TECHNICAL_VERIFICATION
                ),

            "training_yaml_sha256":
                br.sha256_file(
                    TRAINING_YAML
                ),

            "init_lock_sha256":
                br.sha256_file(
                    INIT_LOCK
                ),

            "data_bindings_sha256":
                br.sha256_file(
                    DATA_BINDINGS
                ),
        },

        "runtime":
            runtime,

        "output": {
            "run_dir":
                str(run_dir),

            "run_dir_exists_before_launch":
                False,
        },
    }

    preflight_path = (
        preflight_dir
        / "PRETRAIN_PREFLIGHT.json"
    )

    preflight_path.write_text(
        json.dumps(
            preflight,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return (
        preflight,
        preflight_path,
        checkpoint,
    )


def execute(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
) -> None:
    (
        preflight,
        preflight_path,
        checkpoint,
    ) = perform_preflight(
        experiment_id,
        input_root=input_root,
        work_root=work_root,
    )

    training = effective_training()

    os.environ[
        "YOLO_OFFLINE"
    ] = "true"

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        "/kaggle/working/"
        ".ultralytics_config",
    )

    os.environ[
        "RESEMA_EXPERIMENT_ID"
    ] = experiment_id

    os.environ[
        "RESEMA_INITIALIZATION"
    ] = "pretrained"

    os.environ[
        "RESEMA_PREFLIGHT_MANIFEST"
    ] = str(
        preflight_path
    )

    os.environ[
        "RESEMA_PREFLIGHT_SHA256"
    ] = br.sha256_file(
        preflight_path
    )

    os.environ[
        "RESEMA_REPO_ROOT"
    ] = str(
        ROOT.resolve()
    )

    os.environ[
        "RESEMA_SCREEN_FAMILY"
    ] = SCREEN_ID

    from ultralytics.research.combination_screen_trainer import (
        GovernedCombinationScreenTrainer,
    )

    args = br.build_training_args(
        training,
        experiment_id=experiment_id,
        runtime_data_yaml=Path(
            preflight[
                "runtime_data_yaml"
            ][
                "path"
            ]
        ),
    )

    args["model"] = str(
        MODEL_YAML
    )

    args["pretrained"] = str(
        checkpoint
    )

    args["task"] = "detect"

    trainer = (
        GovernedCombinationScreenTrainer(
            overrides=args
        )
    )

    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Governed COMBINATION-SCREEN-01 "
            "fresh-run launcher."
        )
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
        choices=EXPERIMENT_IDS,
    )

    parser.add_argument(
        "--mode",
        choices=(
            "preflight-only",
            "execute",
        ),
        default="preflight-only",
    )

    parser.add_argument(
        "--input-root",
        default="/kaggle/input",
    )

    parser.add_argument(
        "--work-root",
        default="/kaggle/working",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "preflight-only":

        (
            preflight,
            preflight_path,
            _checkpoint,
        ) = perform_preflight(
            args.experiment_id,
            input_root=Path(
                args.input_root
            ),
            work_root=Path(
                args.work_root
            ),
        )

        print("=" * 100)
        print(
            "COMBINATION-SCREEN-01 "
            "GOVERNED PREFLIGHT"
        )
        print("=" * 100)

        print(
            "EXPERIMENT_ID="
            f"{preflight['experiment_id']}"
        )

        print(
            "TRAINING_SOURCE_COMMIT="
            f"{preflight['training_source_commit']}"
        )

        print(
            "EXECUTION_COMMIT="
            f"{preflight['execution_commit']}"
        )

        print(
            "MODEL_YAML="
            f"{preflight['model_contract']['model_yaml']}"
        )

        print(
            "PARAMETERS="
            f"{preflight['model_contract']['expected_parameters']}"
        )

        print(
            "STATE_ITEMS="
            f"{preflight['model_contract']['expected_state_items']}"
        )

        print(
            "TRANSFERABLE_ITEMS="
            f"{preflight['model_contract']['expected_transferable_source_items']}"
        )

        print(
            "NONTRANSFERABLE_TARGET_ITEMS="
            f"{len(preflight['model_contract']['expected_nontransferable_target_keys'])}"
        )

        print(
            "TECHNICAL_VERIFICATION_SHA256="
            f"{preflight['model_contract']['technical_verification_sha256']}"
        )

        print(
            "DATA_BINDING="
            f"{preflight['data_binding']}"
        )

        print("INITIALIZATION=pretrained")
        print("SEED=42")
        print("EPOCHS=100")
        print("TEST_ACCESS=NONE")

        print(
            "PREFLIGHT_MANIFEST="
            f"{preflight_path}"
        )

        print(
            "PREFLIGHT_SHA256="
            f"{br.sha256_file(preflight_path)}"
        )

        print("TRAINING_STARTED=FALSE")
        print("COMB01_PREFLIGHT=PASS")
        print("=" * 100)

        return

    execute(
        args.experiment_id,
        input_root=Path(
            args.input_root
        ),
        work_root=Path(
            args.work_root
        ),
    )


if __name__ == "__main__":
    main()