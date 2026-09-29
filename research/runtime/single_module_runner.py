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
from ultralytics.nn.tasks import DetectionModel, torch_safe_load


SCREEN_ID = "SINGLE_MODULE_SCREEN_01"

EXPECTED_BRANCH = "research/single-module-screen-01"

REGISTRY = (
    ROOT
    / "research/05_experiments/"
    "SINGLE_MODULE_SCREEN_01_EXPERIMENTS.csv"
)

CONTRACT = (
    ROOT
    / "research/05_experiments/"
    "SINGLE_MODULE_SCREEN_01_CONTRACT.json"
)

AUTHORIZATION = (
    ROOT
    / "research/05_experiments/"
    "SINGLE_MODULE_SCREEN_01_AUTHORIZATION.json"
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

OUTPUT_ROOT = Path(
    "/kaggle/working/ResEMA_single_module_runs"
)

PREFLIGHT_ROOT_NAME = (
    "ResEMA_single_module_preflight"
)

EXPERIMENT_IDS = (
    "BORG-PT-S42-SCCONV-EARLY-E100",
    "BORG-PT-S42-SCCONV-4STAGE-E100",
    "BORG-PT-S42-DYSAMPLE-E100",
    "BORG-PT-S42-CANONICAL-EMA-E100",
)

BASELINE_NATIVE_NONTRANSFER_KEYS = [
    "model.23.cv3.0.2.weight",
    "model.23.cv3.0.2.bias",
    "model.23.cv3.1.2.weight",
    "model.23.cv3.1.2.bias",
    "model.23.cv3.2.2.weight",
    "model.23.cv3.2.2.bias",
]

CANDIDATES = {
    "BORG-PT-S42-SCCONV-EARLY-E100": {
        "model_yaml":
            "ultralytics/cfg/models/11/"
            "yolo11s-tpsc-early-v1.yaml",
        "expected_parameters": 9_570_093,
        "expected_state_items": 537,
        "expected_new_state_items": 38,
        "allowed_new_state_markers": (
            ".sc_adapters.",
        ),
    },
    "BORG-PT-S42-SCCONV-4STAGE-E100": {
        "model_yaml":
            "ultralytics/cfg/models/11/"
            "yolo11s-tpsc-g4-v1.yaml",
        "expected_parameters": 10_021_679,
        "expected_state_items": 575,
        "expected_new_state_items": 76,
        "allowed_new_state_markers": (
            ".sc_adapters.",
        ),
    },
    "BORG-PT-S42-DYSAMPLE-E100": {
        "model_yaml":
            "ultralytics/cfg/models/11/"
            "yolo11s-dysample-v2.yaml",
        "expected_parameters": 9_455_915,
        "expected_state_items": 505,
        "expected_new_state_items": 6,
        "allowed_new_state_markers": (
            ".offset.",
            ".init_pos",
        ),
    },
    "BORG-PT-S42-CANONICAL-EMA-E100": {
        "model_yaml":
            "ultralytics/cfg/models/11/"
            "yolo11s-tpema-head-v1.yaml",
        "expected_parameters": 9_435_423,
        "expected_state_items": 527,
        "expected_new_state_items": 28,
        "allowed_new_state_markers": (
            ".ema_adapter.",
        ),
    },
}


class ScreenGovernanceError(br.GovernanceError):
    pass


def _read_registry_text(
    text: str,
) -> tuple[list[str], list[dict[str, str]]]:
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
        "status",
        "source_commit",
        "authorization_commit",
        "expected_parameters",
        "checkpoint_sha256",
        "test_access",
    }

    if not required.issubset(fields):
        raise ScreenGovernanceError(
            "SMS-01 registry schema incomplete."
        )

    ids = [
        row["experiment_id"]
        for row in rows
    ]

    if len(rows) != 4:
        raise ScreenGovernanceError(
            "SMS-01 registry must contain exactly four rows."
        )

    if len(ids) != len(set(ids)):
        raise ScreenGovernanceError(
            "SMS-01 duplicate experiment ID."
        )

    if set(ids) != set(EXPERIMENT_IDS):
        raise ScreenGovernanceError(
            "SMS-01 experiment-ID set drift."
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
    _fields, rows = read_registry()

    matches = [
        row
        for row in rows
        if row["experiment_id"] == experiment_id
    ]

    if len(matches) != 1:
        raise ScreenGovernanceError(
            "SMS-01 experiment must resolve once."
        )

    row = matches[0]

    fixed = {
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "pretrained",
        "seed": "42",
        "epochs": "100",
        "imgsz": "1024",
        "batch": "16",
        "optimizer": "SGD",
        "test_access": "NONE",
    }

    for key, expected in fixed.items():
        if row.get(key) != expected:
            raise ScreenGovernanceError(
                f"SMS-01 registry drift: "
                f"{key}={row.get(key)!r} "
                f"!= {expected!r}"
            )

    candidate = CANDIDATES[experiment_id]

    if (
        row["model_yaml"]
        != candidate["model_yaml"]
    ):
        raise ScreenGovernanceError(
            "SMS-01 model-YAML binding drift."
        )

    if int(
        row["expected_parameters"]
    ) != int(
        candidate["expected_parameters"]
    ):
        raise ScreenGovernanceError(
            "SMS-01 parameter registry drift."
        )

    return row


def load_contract() -> dict:
    data = br.load_json(CONTRACT)

    if (
        data.get("experiment_family")
        != SCREEN_ID
    ):
        raise ScreenGovernanceError(
            "SMS-01 contract family drift."
        )

    condition = data[
        "development_condition"
    ]

    required = {
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "pretrained",
        "seed": 42,
        "epochs": 100,
        "imgsz": 1024,
        "batch_global": 16,
        "optimizer": "SGD",
        "selection_split": "val",
        "test_access": "NONE",
    }

    for key, expected in required.items():
        if condition.get(key) != expected:
            raise ScreenGovernanceError(
                "SMS-01 contract condition drift: "
                f"{key}"
            )

    if (
        data["advancement_rules"][
            "combination_training_authorized"
        ]
        is not False
    ):
        raise ScreenGovernanceError(
            "SMS-01 combinations unexpectedly authorized."
        )

    return data


def _expected_authorized_registry(
    source_commit: str,
):
    frozen_text = br.git(
        "show",
        f"{source_commit}:"
        "research/05_experiments/"
        "SINGLE_MODULE_SCREEN_01_EXPERIMENTS.csv",
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
        raise ScreenGovernanceError(
            "SMS-01 registry schema changed "
            "after source freeze."
        )

    frozen_by_id = {
        row["experiment_id"]: row
        for row in frozen_rows
    }

    current_by_id = {
        row["experiment_id"]: row
        for row in current_rows
    }

    for experiment_id in EXPERIMENT_IDS:
        frozen = frozen_by_id[
            experiment_id
        ]

        current = current_by_id[
            experiment_id
        ]

        if (
            frozen["status"]
            != "REGISTERED_NOT_AUTHORIZED"
        ):
            raise ScreenGovernanceError(
                "Frozen SMS-01 source did not "
                "contain registration state."
            )

        if frozen["source_commit"].strip():
            raise ScreenGovernanceError(
                "Frozen SMS-01 source_commit "
                "must be blank."
            )

        if (
            current["status"]
            != "AUTHORIZED"
        ):
            raise ScreenGovernanceError(
                "SMS-01 current registry is "
                "not authorized."
            )

        if (
            current["source_commit"].strip()
            != source_commit
        ):
            raise ScreenGovernanceError(
                "SMS-01 source binding mismatch."
            )

        if current[
            "authorization_commit"
        ].strip():
            raise ScreenGovernanceError(
                "SMS-01 authorization_commit "
                "field is intentionally "
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
                raise ScreenGovernanceError(
                    "SMS-01 registry changed "
                    "outside authorized binding: "
                    f"{experiment_id} {field}"
                )


def _expected_authorized_contract(
    source_commit: str,
):
    frozen = json.loads(
        br.git(
            "show",
            f"{source_commit}:"
            "research/05_experiments/"
            "SINGLE_MODULE_SCREEN_01_CONTRACT.json",
        )
    )

    current = load_contract()

    expected = copy.deepcopy(frozen)

    if (
        expected["status"]
        != "REGISTERED_NOT_AUTHORIZED"
    ):
        raise ScreenGovernanceError(
            "Frozen SMS-01 contract status drift."
        )

    if (
        expected["firewall"][
            "training_authorized"
        ]
        is not False
    ):
        raise ScreenGovernanceError(
            "Frozen SMS-01 contract was "
            "unexpectedly authorized."
        )

    if (
        expected["provenance"][
            "source_commit"
        ]
        is not None
    ):
        raise ScreenGovernanceError(
            "Frozen SMS-01 contract source "
            "must be null."
        )

    expected["status"] = "AUTHORIZED"
    expected["firewall"][
        "training_authorized"
    ] = True

    expected["provenance"][
        "source_commit"
    ] = source_commit

    if current != expected:
        raise ScreenGovernanceError(
            "SMS-01 contract changed outside "
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
        raise ScreenGovernanceError(
            f"Wrong SMS-01 branch: {branch!r}"
        )

    if dirty:
        raise ScreenGovernanceError(
            "SMS-01 execution requires a "
            "clean Git worktree."
        )

    source_commit = (
        row["source_commit"].strip()
    )

    if not source_commit:
        raise ScreenGovernanceError(
            "SMS-01 remains locked: "
            "source_commit is blank."
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
        raise ScreenGovernanceError(
            "SMS-01 frozen source is not "
            "an ancestor of execution HEAD."
        )

    _expected_authorized_registry(
        source_commit
    )

    _expected_authorized_contract(
        source_commit
    )

    if not AUTHORIZATION.is_file():
        raise ScreenGovernanceError(
            "SMS-01 authorization record "
            "is missing."
        )

    authorization = br.load_json(
        AUTHORIZATION
    )

    required_auth = {
        "schema_version":
            "SMS01-authorization-v1.0",
        "status": "AUTHORIZED",
        "branch": EXPECTED_BRANCH,
        "source_commit": source_commit,
        "training_authorized": True,
        "test_access": "NONE",
    }

    for key, expected in required_auth.items():
        if (
            authorization.get(key)
            != expected
        ):
            raise ScreenGovernanceError(
                "SMS-01 authorization "
                f"record drift: {key}"
            )

    if set(
        authorization.get(
            "experiment_ids",
            [],
        )
    ) != set(EXPERIMENT_IDS):
        raise ScreenGovernanceError(
            "SMS-01 authorization "
            "experiment set drift."
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
        "SINGLE_MODULE_SCREEN_01_EXPERIMENTS.csv",
        "research/05_experiments/"
        "SINGLE_MODULE_SCREEN_01_CONTRACT.json",
        "research/05_experiments/"
        "SINGLE_MODULE_SCREEN_01_AUTHORIZATION.json",
    }

    unexpected = (
        changed
        - allowed_authorization_delta
    )

    if unexpected:
        raise ScreenGovernanceError(
            "Scientific SMS-01 source changed "
            "after source freeze:\n"
            + "\n".join(
                sorted(unexpected)
            )
        )

    return source_commit, head


def effective_training() -> dict:
    training = br.load_yaml(
        TRAINING_YAML
    )

    if (
        training.get("status")
        != "FROZEN_BASELINE_RECIPE_V2"
    ):
        raise ScreenGovernanceError(
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

    for key, expected in required.items():
        if t.get(key) != expected:
            raise ScreenGovernanceError(
                f"SMS-01 training drift: {key}"
            )

    effective["output"][
        "project_dir"
    ] = str(OUTPUT_ROOT)

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
    experiment_id: str,
    checkpoint: Path,
    init_lock: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    candidate = CANDIDATES[
        experiment_id
    ]

    model_yaml = (
        ROOT
        / candidate["model_yaml"]
    )

    registry_row = load_row(
        experiment_id
    )

    if (
        br.sha256_file(model_yaml)
        != next(
            item["model_yaml_sha256"]
            for item in contract[
                "candidates"
            ]
            if item[
                "experiment_id"
            ] == experiment_id
        )
    ):
        raise ScreenGovernanceError(
            "SMS-01 candidate YAML hash drift."
        )

    baseline = build_model(
        BASELINE_YAML
    )

    target = build_model(
        model_yaml
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

    if (
        parameters
        != candidate[
            "expected_parameters"
        ]
    ):
        raise ScreenGovernanceError(
            "SMS-01 candidate parameter "
            f"count drift: {parameters}"
        )

    if (
        len(target_state)
        != candidate[
            "expected_state_items"
        ]
    ):
        raise ScreenGovernanceError(
            "SMS-01 candidate state-item "
            f"count drift: {len(target_state)}"
        )

    missing_native = [
        key
        for key, value
        in baseline_state.items()
        if (
            key not in target_state
            or target_state[key].shape
            != value.shape
        )
    ]

    if missing_native:
        raise ScreenGovernanceError(
            "SMS-01 native shared-state "
            f"contract failed: "
            f"{missing_native[:20]}"
        )

    new_target_keys = sorted(
        set(target_state)
        - set(baseline_state)
    )

    if (
        len(new_target_keys)
        != candidate[
            "expected_new_state_items"
        ]
    ):
        raise ScreenGovernanceError(
            "SMS-01 new-state-item "
            "count drift."
        )

    markers = candidate[
        "allowed_new_state_markers"
    ]

    unexpected_new = [
        key
        for key in new_target_keys
        if not any(
            marker in key
            for marker in markers
        )
    ]

    if unexpected_new:
        raise ScreenGovernanceError(
            "SMS-01 unexpected candidate "
            f"state keys: {unexpected_new}"
        )

    expected_checkpoint_sha = (
        init_lock[
            "official_checkpoint"
        ]["sha256"]
    )

    observed_checkpoint_sha = (
        br.sha256_file(
            checkpoint
        )
    )

    if (
        observed_checkpoint_sha
        != expected_checkpoint_sha
    ):
        raise ScreenGovernanceError(
            "SMS-01 official checkpoint "
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
            and target_state[key].shape
            == value.shape
        )
    }

    expected_transferable = int(
        init_lock[
            "nine_class_initialization_contract"
        ][
            "transferable_state_items"
        ]
    )

    if (
        len(transferable)
        != expected_transferable
    ):
        raise ScreenGovernanceError(
            "SMS-01 official checkpoint "
            "transfer count drift: "
            f"{len(transferable)} "
            f"!= {expected_transferable}"
        )

    native_nontransfer = set(
        init_lock[
            "nine_class_initialization_contract"
        ][
            "nontransferable_target_keys"
        ]
    )

    if native_nontransfer != set(
        BASELINE_NATIVE_NONTRANSFER_KEYS
    ):
        raise ScreenGovernanceError(
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
    ):
        raise ScreenGovernanceError(
            "SMS-01 target nontransfer "
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
        raise ScreenGovernanceError(
            "SMS-01 unexpected official "
            "checkpoint load keys."
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
        raise ScreenGovernanceError(
            "SMS-01 transferred official "
            "checkpoint tensor differs."
        )

    if not all(
        torch.equal(
            loaded[key].detach().cpu(),
            initial_target[key],
        )
        for key
        in expected_nontransferred
    ):
        raise ScreenGovernanceError(
            "SMS-01 unmatched target state "
            "did not preserve seeded init."
        )

    if (
        registry_row[
            "checkpoint_sha256"
        ]
        != expected_checkpoint_sha
    ):
        raise ScreenGovernanceError(
            "SMS-01 registry checkpoint "
            "binding drift."
        )

    return {
        "model_yaml":
            candidate["model_yaml"],
        "model_yaml_sha256":
            br.sha256_file(
                model_yaml
            ),
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

    contract = load_contract()

    training = effective_training()

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

    runtime = br.verify_runtime_environment(
        init_lock
    )

    expectation = br.binding_expectation(
        row,
        data_manifest,
    )

    if (
        expectation["binding_id"]
        != "DATA01:B-ORG:v1"
    ):
        raise ScreenGovernanceError(
            "SMS-01 must use B-ORG."
        )

    train_images = (
        br.discover_membership_directory(
            input_root,
            kind="images",
            expected_count=
                expectation["train_count"],
            expected_hash=
                expectation["train_hash"],
        )
    )

    train_labels = (
        br.discover_membership_directory(
            input_root,
            kind="labels",
            expected_count=
                expectation["train_count"],
            expected_hash=
                expectation["train_hash"],
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
            experiment_id=experiment_id,
            checkpoint=checkpoint,
            init_lock=init_lock,
            contract=contract,
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
        raise ScreenGovernanceError(
            "SMS-01 fresh launch refuses "
            f"existing run directory: {run_dir}"
        )

    preflight = {
        "schema_version":
            "SMS01-preflight-v1.0",
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
                ]["sha256"],
            "bytes":
                init_lock[
                    "official_checkpoint"
                ]["bytes"],
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
        ) + "\n",
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
):
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

    os.environ["YOLO_OFFLINE"] = "true"

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
    ] = str(preflight_path)

    os.environ[
        "RESEMA_PREFLIGHT_SHA256"
    ] = br.sha256_file(
        preflight_path
    )

    os.environ[
        "RESEMA_REPO_ROOT"
    ] = str(ROOT.resolve())

    os.environ[
        "RESEMA_SCREEN_FAMILY"
    ] = SCREEN_ID

    from ultralytics.research.single_module_trainer import (
        GovernedSingleModuleTrainer,
    )

    model_yaml = (
        ROOT
        / preflight[
            "model_contract"
        ][
            "model_yaml"
        ]
    )

    args = br.build_training_args(
        training,
        experiment_id=experiment_id,
        runtime_data_yaml=Path(
            preflight[
                "runtime_data_yaml"
            ]["path"]
        ),
    )

    # Direct trainer construction is deliberate:
    # cfg = candidate YAML;
    # pretrained = locked official YOLO11s checkpoint.
    # This avoids instantiating the baseline architecture
    # from the checkpoint itself.
    args["model"] = str(
        model_yaml
    )

    args["pretrained"] = str(
        checkpoint
    )

    args["task"] = "detect"

    trainer = (
        GovernedSingleModuleTrainer(
            overrides=args
        )
    )

    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Governed SINGLE-MODULE-SCREEN-01 "
            "launcher."
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
            "SINGLE-MODULE-SCREEN-01 "
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
            "TRANSFERABLE_ITEMS="
            f"{preflight['model_contract']['expected_transferable_source_items']}"
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
        print("SMS01_PREFLIGHT=PASS")
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