from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from research.runtime import baseline_runner as br
from research.runtime import resume_runner as base_resume
from research.runtime import combination_screen_runner as comb

ROOT = comb.ROOT
RESUME_CONTRACT = (
    ROOT
    / "research/05_experiments/"
    "COMBINATION_SCREEN_01_RESUME_CONTRACT.json"
)

REQUIRED_PRIOR_RUN_FILES = [
    "args.yaml",
    "results.csv",
    "weights/last.pt",
    "weights/best.pt",
    "governance/PRETRAIN_PREFLIGHT.json",
    "governance/RUNTIME_MANIFEST.json",
    "governance/COMB01_MODEL_RUNTIME.json",
]


def _path_is_test_like(
    path: Path,
    *,
    root: Path,
) -> bool:
    try:
        relative = (
            path.resolve()
            .relative_to(
                root.resolve()
            )
        )
    except ValueError as exc:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume path escaped "
            "the supplied input root: "
            f"{path}"
        ) from exc

    return any(
        part.lower().startswith("test")
        for part in relative.parts
    )


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    )
    if not isinstance(data, dict):
        raise comb.CombinationGovernanceError(
            f"Expected JSON object: {path}"
        )
    return data


def load_resume_contract() -> dict[str, Any]:
    data = br.load_json(RESUME_CONTRACT)

    required = {
        "schema_version": "COMB01-resume-contract-v1.0",
        "status": "IMPLEMENTED",
        "experiment_family": comb.SCREEN_ID,
        "branch": comb.EXPECTED_BRANCH,
        "test_access": "NONE",
    }

    for key, expected in required.items():
        if data.get(key) != expected:
            raise comb.CombinationGovernanceError(
                "COMB-01 resume contract drift: "
                f"{key}"
            )

    if data.get("experiment_ids") != list(comb.EXPERIMENT_IDS):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume experiment set drift."
        )

    if data.get("allowed_modes") != [
        "preflight-only",
        "execute",
    ]:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume modes drift."
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
        if data.get(key) is not True:
            raise comb.CombinationGovernanceError(
                "COMB-01 resume policy drift: "
                f"{key}"
            )

    if data.get("scientific_overrides_allowed") is not False:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume scientific override policy drift."
        )

    if (
        data.get(
            "training_authorization_delegated_to_registry_and_authorization_record"
        )
        is not True
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume authorization-delegation "
            "policy drift."
        )

    if (
        data.get(
            "technical_verification_sha256"
        )
        != comb.TECHNICAL_VERIFICATION_SHA256
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume technical-verification "
            "SHA drift."
        )

    if int(
        data.get(
            "expected_parameters",
            -1,
        )
    ) != 9_574_241:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume parameter authority drift."
        )

    if int(
        data.get(
            "expected_state_items",
            -1,
        )
    ) != 565:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume state-item authority drift."
        )

    if (
        data.get(
            "expected_original_initialization"
        )
        != "pretrained"
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume original-initialization "
            "authority drift."
        )

    return data


def _discover_prior_run(
    input_root: Path,
    experiment_id: str,
) -> Path:
    candidates: list[Path] = []

    for current, dirs, _files in os.walk(input_root):
        current_path = Path(current)

        dirs[:] = [
            d
            for d in dirs
            if (
                not d.lower().startswith("test")
                and d not in {".git", "__pycache__"}
            )
        ]

        if _path_is_test_like(
            current_path,
            root=input_root,
        ):
            continue

        if current_path.name != experiment_id:
            continue

        if all(
            (current_path / rel).is_file()
            for rel in REQUIRED_PRIOR_RUN_FILES
        ):
            candidates.append(current_path.resolve())

    unique = sorted(set(candidates))

    if len(unique) != 1:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume expected exactly one complete "
            f"prior candidate run for {experiment_id}; "
            f"observed {unique}"
        )

    return unique[0]


def _existing_comb_resume_indices(
    run_dir: Path,
) -> list[int]:
    sessions_dir = (
        run_dir
        / "governance/resume_sessions"
    )

    base_indices = (
        base_resume._existing_resume_indices(
            run_dir
        )
    )

    if not sessions_dir.exists():
        if base_indices:
            raise comb.CombinationGovernanceError(
                "COMB-01 resume session provenance "
                "is incomplete: "
                f"baseline={base_indices}, "
                "sms=[]"
            )
        return []

    comb_indices: set[int] = set()

    for path in sessions_dir.glob(
        "COMB01_RESUME_MODEL_RUNTIME_*.json"
    ):
        suffix = path.stem.rsplit("_", 1)[-1]
        try:
            comb_indices.add(int(suffix))
        except ValueError as exc:
            raise comb.CombinationGovernanceError(
                "Malformed COMB-01 resume model runtime "
                f"filename: {path}"
            ) from exc

    observed = sorted(comb_indices)

    if observed != base_indices:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume session provenance is incomplete: "
            f"baseline={base_indices}, sms={observed}"
        )

    return observed


def _verify_prior_resume_sessions(
    run_dir: Path,
    *,
    experiment_id: str,
    source_commit: str,
    execution_commit: str,
    data_binding: str,
    model_contract: dict[str, Any],
    original_runtime_manifest_sha256: str,
    original_pretrain_preflight_sha256: str,
    original_comb_model_runtime_sha256: str,
) -> list[int]:
    indices = _existing_comb_resume_indices(run_dir)

    if not indices:
        return []

    sessions_dir = (
        run_dir
        / "governance/resume_sessions"
    )

    for index in indices:
        preflight_path = (
            sessions_dir
            / f"RESUME_PREFLIGHT_{index:03d}.json"
        )
        runtime_path = (
            sessions_dir
            / f"RESUME_RUNTIME_{index:03d}.json"
        )
        comb_runtime_path = (
            sessions_dir
            / f"COMB01_RESUME_MODEL_RUNTIME_{index:03d}.json"
        )
        args_path = (
            sessions_dir
            / f"ARGS_BEFORE_RESUME_{index:03d}.yaml"
        )

        if not args_path.is_file():
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume args archive is missing: "
                f"{args_path}"
            )

        preflight = _json(preflight_path)
        runtime = _json(runtime_path)
        comb_runtime = _json(comb_runtime_path)

        preflight_required = {
            "schema_version":
                "COMB01-resume-preflight-v1.0",
            "experiment_family":
                comb.SCREEN_ID,
            "experiment_id":
                experiment_id,
            "resume_session_index":
                index,
            "training_source_commit":
                source_commit,
            "execution_commit":
                execution_commit,
            "data_binding":
                data_binding,
            "initialization":
                "pretrained",
            "original_runtime_manifest_sha256":
                original_runtime_manifest_sha256,
            "original_pretrain_preflight_sha256":
                original_pretrain_preflight_sha256,
            "original_comb_model_runtime_sha256":
                original_comb_model_runtime_sha256,
        }

        for key, expected in preflight_required.items():
            if preflight.get(key) != expected:
                raise comb.CombinationGovernanceError(
                    "COMB-01 prior resume preflight drift: "
                    f"session={index}, key={key}"
                )

        if preflight.get("model_contract") != model_contract:
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume model contract drift: "
                f"session={index}"
            )

        prior_technical = preflight.get(
            "technical_verification",
            {},
        )

        if (
            prior_technical.get("sha256")
            != comb.TECHNICAL_VERIFICATION_SHA256
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume technical "
                "verification drift: "
                f"session={index}"
            )

        if (
            prior_technical.get("status")
            != "PASS"
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume technical "
                "verification status drift: "
                f"session={index}"
            )

        prior_test_access = preflight.get(
            "test_access",
            {},
        )

        for key in (
            "runtime_yaml_contains_test",
            "test_directory_scan_pruned",
            "test_predictions",
            "test_metrics",
            "test_error_analysis",
        ):
            expected = (
                True
                if key == "test_directory_scan_pruned"
                else False
            )

            if prior_test_access.get(key) is not expected:
                raise comb.CombinationGovernanceError(
                    "COMB-01 prior resume test firewall drift: "
                    f"session={index}, key={key}"
                )

        copy_verification = preflight.get(
            "copy_verification",
            {},
        )

        if (
            copy_verification.get(
                "exact_file_inventory_match"
            )
            is not True
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume copy verification drift: "
                f"session={index}"
            )

        archived_args = preflight.get(
            "archived_args_before_resume",
            {},
        )

        if (
            archived_args.get("sha256")
            != br.sha256_file(args_path)
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume args archive hash drift: "
                f"session={index}"
            )

        runtime_required = {
            "schema_version":
                "RESUME01-runtime-session-v1.0",
            "experiment_id":
                experiment_id,
            "resume_session_index":
                index,
            "training_source_commit":
                source_commit,
            "execution_commit":
                execution_commit,
            "data_binding":
                data_binding,
            "original_initialization":
                "pretrained",
            "test_split_present":
                False,
        }

        for key, expected in runtime_required.items():
            if runtime.get(key) != expected:
                raise comb.CombinationGovernanceError(
                    "COMB-01 prior resume runtime drift: "
                    f"session={index}, key={key}"
                )

        if (
            runtime.get(
                "resume_preflight_manifest",
                {},
            ).get("sha256")
            != br.sha256_file(preflight_path)
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume preflight hash drift: "
                f"session={index}"
            )

        if (
            runtime.get(
                "original_runtime_manifest",
                {},
            ).get("sha256")
            != original_runtime_manifest_sha256
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior original runtime hash drift: "
                f"session={index}"
            )

        if (
            runtime.get("parent_checkpoint")
            != preflight.get("parent_checkpoint")
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior parent-checkpoint provenance drift: "
                f"session={index}"
            )

        parent_checkpoint = preflight.get(
            "parent_checkpoint",
            {},
        )

        expected_start_epoch = (
            int(
                parent_checkpoint[
                    "epoch_zero_based"
                ]
            )
            + 1
        )

        if (
            int(
                runtime.get(
                    "restored_start_epoch_zero_based",
                    -1,
                )
            )
            != expected_start_epoch
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior restored start epoch drift: "
                f"session={index}"
            )

        if (
            int(
                runtime.get(
                    "next_epoch_one_based",
                    -1,
                )
            )
            != int(
                parent_checkpoint[
                    "next_epoch_one_based"
                ]
            )
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior next-epoch provenance drift: "
                f"session={index}"
            )

        if (
            int(
                runtime.get(
                    "total_epochs",
                    -1,
                )
            )
            != int(
                preflight.get(
                    "total_epochs",
                    -2,
                )
            )
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior epoch-budget provenance drift: "
                f"session={index}"
            )

        sms_required = {
            "schema_version":
                "COMB01-model-runtime-v1.0",
            "experiment_family":
                comb.SCREEN_ID,
            "experiment_id":
                experiment_id,
            "resume_session_index":
                index,
            "training_source_commit":
                source_commit,
            "execution_commit":
                execution_commit,
            "test_access":
                "NONE",
        }

        for key, expected in sms_required.items():
            if comb_runtime.get(key) != expected:
                raise comb.CombinationGovernanceError(
                    "COMB-01 prior resume model runtime drift: "
                    f"session={index}, key={key}"
                )

        if comb_runtime.get("model_contract") != model_contract:
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume model-contract audit drift: "
                f"session={index}"
            )

        prior_runtime_technical = comb_runtime.get(
            "technical_verification",
            {},
        )

        if (
            prior_runtime_technical.get("sha256")
            != comb.TECHNICAL_VERIFICATION_SHA256
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume model-runtime "
                "technical-verification drift: "
                f"session={index}"
            )

        if (
            prior_runtime_technical.get("status")
            != "PASS"
        ):
            raise comb.CombinationGovernanceError(
                "COMB-01 prior resume model-runtime "
                "technical status drift: "
                f"session={index}"
            )

        audit = comb_runtime.get(
            "initialization_audit",
            {},
        )

        audit_required = {
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

        for key, expected in audit_required.items():
            if audit.get(key) != expected:
                raise comb.CombinationGovernanceError(
                    "COMB-01 prior resume model audit drift: "
                    f"session={index}, key={key}"
                )

        if runtime.get("resume_model_audit") != audit:
            raise comb.CombinationGovernanceError(
                "COMB-01 prior runtime/model audit mismatch: "
                f"session={index}"
            )

    return indices


def _verify_original_comb_runtime(
    *,
    experiment_id: str,
    path: Path,
    original_preflight: dict[str, Any],
    source_commit: str,
    execution_commit: str,
) -> dict[str, Any]:
    data = _json(path)

    required = {
        "schema_version": "COMB01-model-runtime-v1.0",
        "experiment_family": comb.SCREEN_ID,
        "experiment_id": experiment_id,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "test_access": "NONE",
    }

    for key, expected in required.items():
        if data.get(key) != expected:
            raise comb.CombinationGovernanceError(
                "COMB-01 original model runtime drift: "
                f"{key}"
            )

    if (
        data.get("model_contract")
        != original_preflight.get("model_contract")
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 model runtime/preflight "
            "contract mismatch."
        )

    audit = data.get("initialization_audit", {})

    if audit.get("initialization") != "pretrained":
        raise comb.CombinationGovernanceError(
            "COMB-01 original initialization audit drift."
        )

    if audit.get("official_checkpoint_loaded") is not True:
        raise comb.CombinationGovernanceError(
            "COMB-01 original checkpoint-load audit drift."
        )

    if int(audit.get("transferable_state_items", -1)) != 493:
        raise comb.CombinationGovernanceError(
            "COMB-01 original transfer-count audit drift."
        )

    technical = data.get(
        "technical_verification",
        {},
    )

    if (
        technical.get("sha256")
        != comb.TECHNICAL_VERIFICATION_SHA256
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original model runtime "
            "technical-verification drift."
        )

    if (
        technical.get("status")
        != "PASS"
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original technical verification "
            "is not PASS."
        )

    return data


def _verify_model_contract(
    *,
    experiment_id: str,
    model_contract: dict[str, Any],
) -> None:
    candidate = comb.CANDIDATES[experiment_id]

    expected = {
        "model_yaml": candidate["model_yaml"],
        "expected_parameters":
            candidate["expected_parameters"],
        "expected_state_items":
            candidate["expected_state_items"],
        "new_target_state_items":
            candidate["expected_new_state_items"],
        "expected_transferable_source_items": 493,
        "technical_verification_sha256":
            candidate[
                "technical_verification_sha256"
            ],
    }

    for key, value in expected.items():
        if model_contract.get(key) != value:
            raise comb.CombinationGovernanceError(
                "COMB-01 original model contract drift: "
                f"{key}"
            )

    model_yaml = ROOT / candidate["model_yaml"]

    if (
        br.sha256_file(model_yaml)
        != model_contract.get("model_yaml_sha256")
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume model YAML hash drift."
        )


def _verify_checkpoint_model_identity(
    *,
    experiment_id: str,
    ckpt: dict[str, Any],
    checkpoint: dict[str, Any],
    model_contract: dict[str, Any],
) -> None:
    expected_parameters = int(
        model_contract["expected_parameters"]
    )
    expected_state_items = int(
        model_contract["expected_state_items"]
    )

    if checkpoint["parameter_count"] != expected_parameters:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume checkpoint parameter-count "
            f"mismatch: {checkpoint['parameter_count']} "
            f"!= {expected_parameters}"
        )

    if checkpoint["state_items"] != expected_state_items:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume checkpoint state-item "
            f"mismatch: {checkpoint['state_items']} "
            f"!= {expected_state_items}"
        )

    target = comb.build_model(
        ROOT
        / comb.CANDIDATES[experiment_id]["model_yaml"]
    )

    expected_state = target.state_dict()

    ema = ckpt.get("ema")
    if ema is None:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume checkpoint EMA model is missing."
        )

    observed_state = ema.state_dict()

    if list(observed_state) != list(expected_state):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume checkpoint state-key "
            "contract mismatch."
        )

    bad_shapes = [
        key
        for key in expected_state
        if (
            observed_state[key].shape
            != expected_state[key].shape
        )
    ]

    if bad_shapes:
        raise comb.CombinationGovernanceError(
            "COMB-01 resume checkpoint tensor-shape "
            f"mismatch: {bad_shapes[:20]}"
        )


def analyze_resume(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    row = comb.load_row(experiment_id)
    comb.load_contract()
    load_resume_contract()

    # Must fail closed before runtime/dataset/checkpoint access.
    source_commit, execution_commit = (
        comb.verify_authorization(row)
    )

    training = comb.effective_training()
    init_lock = br.load_json(comb.INIT_LOCK)
    data_manifest = br.load_json(comb.DATA_BINDINGS)
    runtime = br.verify_runtime_environment(init_lock)

    expectation = br.binding_expectation(
        row,
        data_manifest,
    )

    if expectation["binding_id"] != "DATA01:B-ORG:v1":
        raise comb.CombinationGovernanceError(
            "COMB-01 resume must use B-ORG."
        )

    destination_run = (
        Path(training["output"]["project_dir"])
        / experiment_id
    )

    if destination_run.exists():
        raise comb.CombinationGovernanceError(
            "COMB-01 resume refuses existing destination "
            f"run directory: {destination_run}"
        )

    source_run = _discover_prior_run(
        input_root,
        experiment_id,
    )

    original_manifest_path = (
        source_run
        / "governance/RUNTIME_MANIFEST.json"
    )
    original_preflight_path = (
        source_run
        / "governance/PRETRAIN_PREFLIGHT.json"
    )
    original_comb_runtime_path = (
        source_run
        / "governance/COMB01_MODEL_RUNTIME.json"
    )

    original_manifest = _json(
        original_manifest_path
    )
    original_preflight = _json(
        original_preflight_path
    )

    if (
        original_manifest.get("schema_version")
        != "TRAIN01-runtime-manifest-v1.0"
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original runtime-manifest schema drift."
        )

    manifest_required = {
        "experiment_id": experiment_id,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": row["data_binding"],
        "initialization": row["initialization"],
        "test_split_present": False,
    }

    for key, expected in manifest_required.items():
        if original_manifest.get(key) != expected:
            raise comb.CombinationGovernanceError(
                "COMB-01 original runtime manifest drift: "
                f"{key}"
            )

    preflight_required = {
        "schema_version": "COMB01-preflight-v1.0",
        "experiment_family": comb.SCREEN_ID,
        "experiment_id": experiment_id,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": row["data_binding"],
        "initialization": "pretrained",
    }

    for key, expected in preflight_required.items():
        if original_preflight.get(key) != expected:
            raise comb.CombinationGovernanceError(
                "COMB-01 original preflight drift: "
                f"{key}"
            )

    test_access = original_preflight.get(
        "test_access",
        {},
    )

    for key in (
        "runtime_yaml_contains_test",
        "test_predictions",
        "test_metrics",
        "test_error_analysis",
    ):
        if test_access.get(key) is not False:
            raise comb.CombinationGovernanceError(
                "COMB-01 original preflight test "
                f"firewall drift: {key}"
            )

    manifest_preflight_sha = (
        original_manifest.get(
            "preflight_manifest",
            {},
        ).get("sha256")
    )

    if (
        manifest_preflight_sha
        != br.sha256_file(original_preflight_path)
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original PRETRAIN_PREFLIGHT "
            "hash mismatch."
        )

    model_contract = original_preflight[
        "model_contract"
    ]

    original_technical = (
        original_preflight.get(
            "technical_verification",
            {},
        )
    )

    if (
        original_technical.get("sha256")
        != comb.TECHNICAL_VERIFICATION_SHA256
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original preflight "
            "technical-verification SHA drift."
        )

    if (
        original_technical.get("status")
        != "PASS"
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original preflight technical "
            "verification is not PASS."
        )

    _verify_model_contract(
        experiment_id=experiment_id,
        model_contract=model_contract,
    )

    _verify_original_comb_runtime(
        experiment_id=experiment_id,
        path=original_comb_runtime_path,
        original_preflight=original_preflight,
        source_commit=source_commit,
        execution_commit=execution_commit,
    )

    train_images = br.discover_membership_directory(
        input_root,
        kind="images",
        expected_count=expectation["train_count"],
        expected_hash=expectation["train_hash"],
    )
    train_labels = br.discover_membership_directory(
        input_root,
        kind="labels",
        expected_count=expectation["train_count"],
        expected_hash=expectation["train_hash"],
    )
    val_images = br.discover_membership_directory(
        input_root,
        kind="images",
        expected_count=expectation["validation_count"],
        expected_hash=expectation["validation_hash"],
    )
    val_labels = br.discover_membership_directory(
        input_root,
        kind="labels",
        expected_count=expectation["validation_count"],
        expected_hash=expectation["validation_hash"],
    )

    br.verify_image_label_pair(
        train_images,
        train_labels,
    )
    br.verify_image_label_pair(
        val_images,
        val_labels,
    )

    canonical_preflight_dir = (
        work_root
        / comb.PREFLIGHT_ROOT_NAME
        / experiment_id
    )
    canonical_preflight_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_yaml = (
        canonical_preflight_dir
        / "runtime_data_train_val_only.yaml"
    )

    runtime_yaml_sha = br.write_runtime_data_yaml(
        runtime_yaml,
        train_images=train_images,
        validation_images=val_images,
    )

    original_runtime_yaml = original_preflight.get(
        "runtime_data_yaml",
        {},
    )

    if (
        Path(
            str(
                original_runtime_yaml.get(
                    "path",
                    "",
                )
            )
        ).resolve()
        != runtime_yaml.resolve()
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 canonical runtime data YAML path drift."
        )

    if (
        original_runtime_yaml.get("sha256")
        != runtime_yaml_sha
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 regenerated runtime data YAML "
            "hash differs from original."
        )

    if (
        original_runtime_yaml.get("contains_test_key")
        is not False
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 original runtime YAML "
            "test-key firewall drift."
        )

    last_pt = source_run / "weights/last.pt"
    best_pt = source_run / "weights/best.pt"
    args_yaml = source_run / "args.yaml"
    results_csv = source_run / "results.csv"

    ckpt, checkpoint = (
        base_resume._load_checkpoint_metadata(
            last_pt
        )
    )

    total_epochs = int(
        training["training"]["epochs"]
    )

    if checkpoint["completed_epochs"] >= total_epochs:
        raise comb.CombinationGovernanceError(
            "COMB-01 prior run already reached "
            "the frozen epoch budget."
        )

    if (
        checkpoint["git"].get("commit")
        != execution_commit
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 last.pt Git commit does not "
            "equal original execution."
        )

    if (
        checkpoint["git"].get("branch")
        != comb.EXPECTED_BRANCH
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 last.pt Git branch drift."
        )

    if (
        checkpoint["version"]
        != init_lock[
            "exact_fork_runtime"
        ][
            "ultralytics_version"
        ]
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 last.pt Ultralytics version drift."
        )

    _verify_checkpoint_model_identity(
        experiment_id=experiment_id,
        ckpt=ckpt,
        checkpoint=checkpoint,
        model_contract=model_contract,
    )

    base_resume._compare_frozen_training_args(
        checkpoint["train_args"],
        training,
        experiment_id=experiment_id,
        runtime_yaml=runtime_yaml,
    )

    base_resume._verify_args_yaml_matches_checkpoint(
        args_yaml,
        checkpoint["train_args"],
    )

    checkpoint_results = (
        base_resume._verify_results_against_checkpoint(
            results_csv,
            ckpt["train_results"],
        )
    )

    best_ckpt, best_checkpoint = (
        base_resume._load_checkpoint_metadata(
            best_pt
        )
    )

    if (
        best_checkpoint["git"].get("commit")
        != execution_commit
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 best.pt Git commit drift."
        )

    if (
        best_checkpoint["git"].get("branch")
        != comb.EXPECTED_BRANCH
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 best.pt Git branch drift."
        )

    if (
        best_checkpoint["version"]
        != init_lock[
            "exact_fork_runtime"
        ][
            "ultralytics_version"
        ]
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 best.pt Ultralytics version drift."
        )

    _verify_checkpoint_model_identity(
        experiment_id=experiment_id,
        ckpt=best_ckpt,
        checkpoint=best_checkpoint,
        model_contract=model_contract,
    )

    base_resume._compare_frozen_training_args(
        best_checkpoint["train_args"],
        training,
        experiment_id=experiment_id,
        runtime_yaml=runtime_yaml,
    )

    if (
        best_checkpoint["completed_epochs"]
        > checkpoint["completed_epochs"]
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 best.pt epoch is later than last.pt."
        )

    results_state = base_resume._read_results_state(
        results_csv
    )

    if (
        results_state["rows"]
        != checkpoint["completed_epochs"]
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 results row count does not "
            "match last.pt."
        )

    if (
        results_state["last_epoch_one_based"]
        != checkpoint["completed_epochs"]
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 results last epoch does not "
            "match last.pt."
        )

    prior_indices = _verify_prior_resume_sessions(
        source_run,
        experiment_id=experiment_id,
        source_commit=source_commit,
        execution_commit=execution_commit,
        data_binding=row["data_binding"],
        model_contract=model_contract,
        original_runtime_manifest_sha256=
            br.sha256_file(
                original_manifest_path
            ),
        original_pretrain_preflight_sha256=
            br.sha256_file(
                original_preflight_path
            ),
        original_comb_model_runtime_sha256=
            br.sha256_file(
                original_comb_runtime_path
            ),
    )

    session_index = len(prior_indices) + 1

    source_inventory, source_tree_sha = (
        base_resume._tree_inventory(
            source_run
        )
    )

    plan = {
        "schema_version":
            "COMB01-resume-preflight-v1.0",
        "experiment_family":
            comb.SCREEN_ID,
        "experiment_id":
            experiment_id,
        "resume_session_index":
            session_index,
        "training_source_commit":
            source_commit,
        "execution_commit":
            execution_commit,
        "data_binding":
            row["data_binding"],
        "initialization":
            row["initialization"],
        "model_contract":
            model_contract,
        "technical_verification": {
            "path":
                str(
                    comb.TECHNICAL_VERIFICATION
                ),
            "sha256":
                comb.TECHNICAL_VERIFICATION_SHA256,
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
        "membership": {
            "train_images": {
                "path": str(train_images),
                "count": expectation["train_count"],
                "sha256_final_newline":
                    expectation["train_hash"],
            },
            "train_labels": {
                "path": str(train_labels),
                "count": expectation["train_count"],
                "sha256_final_newline":
                    expectation["train_hash"],
            },
            "validation_images": {
                "path": str(val_images),
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
                "path": str(val_labels),
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
        "runtime_data_yaml": {
            "path": str(runtime_yaml),
            "sha256": runtime_yaml_sha,
            "matches_original_session_hash": True,
            "contains_test_key": False,
        },
        "source_run": {
            "path": str(source_run),
            "tree_sha256": source_tree_sha,
            "file_count": len(source_inventory),
        },
        "destination_run": str(destination_run),
        "original_runtime_manifest_sha256":
            br.sha256_file(
                original_manifest_path
            ),
        "original_pretrain_preflight_sha256":
            br.sha256_file(
                original_preflight_path
            ),
        "original_comb_model_runtime_sha256":
            br.sha256_file(
                original_comb_runtime_path
            ),
        "parent_checkpoint": {
            "path_before_copy": str(last_pt),
            "sha256": br.sha256_file(last_pt),
            "bytes": last_pt.stat().st_size,
            "epoch_zero_based":
                checkpoint["epoch_zero_based"],
            "completed_epochs":
                checkpoint["completed_epochs"],
            "next_epoch_one_based":
                checkpoint["next_epoch_one_based"],
        },
        "prior_artifacts": {
            "best_pt_sha256":
                br.sha256_file(best_pt),
            "args_yaml_sha256":
                br.sha256_file(args_yaml),
            "results_csv_sha256":
                br.sha256_file(results_csv),
            "results_rows":
                results_state["rows"],
            "results_match_checkpoint_train_results":
                checkpoint_results[
                    "matches_checkpoint_train_results"
                ],
            "best_pt_epoch_zero_based":
                best_checkpoint[
                    "epoch_zero_based"
                ],
        },
        "total_epochs": total_epochs,
        "runtime": runtime,
        "source_tree_inventory":
            source_inventory,
    }

    del ckpt
    del best_ckpt

    return plan


def materialize_resume(
    plan: dict[str, Any],
):
    return base_resume.materialize_resume(plan)


def write_preview(
    plan: dict[str, Any],
    *,
    work_root: Path,
) -> Path:
    preview_dir = (
        work_root
        / "ResEMA_combination_screen_resume_preflight"
        / plan["experiment_id"]
    )
    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        preview_dir
        / "COMB01_RESUME_PREFLIGHT_PREVIEW.json"
    )

    path.write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return path


def execute(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
) -> None:
    plan = analyze_resume(
        experiment_id,
        input_root=input_root,
        work_root=work_root,
    )

    (
        materialized,
        preflight_path,
        last_pt,
    ) = materialize_resume(plan)

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        "/kaggle/working/.ultralytics_config",
    )

    runtime_yaml = Path(
        materialized[
            "runtime_data_yaml"
        ][
            "path"
        ]
    )

    if not runtime_yaml.is_file():
        raise comb.CombinationGovernanceError(
            "COMB-01 resume runtime YAML "
            "disappeared before launch."
        )

    if (
        br.sha256_file(runtime_yaml)
        != materialized[
            "runtime_data_yaml"
        ][
            "sha256"
        ]
    ):
        raise comb.CombinationGovernanceError(
            "COMB-01 resume runtime YAML "
            "changed before launch."
        )

    preflight_sha = br.sha256_file(
        preflight_path
    )

    os.environ["RESEMA_EXPERIMENT_ID"] = (
        experiment_id
    )
    os.environ["RESEMA_INITIALIZATION"] = (
        materialized["initialization"]
    )
    os.environ[
        "RESEMA_RESUME_PREFLIGHT_MANIFEST"
    ] = str(preflight_path)
    os.environ[
        "RESEMA_RESUME_PREFLIGHT_SHA256"
    ] = preflight_sha
    os.environ["RESEMA_REPO_ROOT"] = str(
        ROOT.resolve()
    )
    os.environ["RESEMA_SCREEN_FAMILY"] = (
        comb.SCREEN_ID
    )

    from ultralytics import YOLO
    from ultralytics.research.combination_screen_trainer import (
        GovernedCombinationScreenTrainer,
    )

    model = YOLO(
        str(last_pt),
        task="detect",
    )

    model.train(
        trainer=GovernedCombinationScreenTrainer,
        resume=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Governed COMB-01 multi-session resume "
            "launcher. Scientific hyperparameter "
            "overrides are not accepted."
        )
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
        choices=comb.EXPERIMENT_IDS,
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

    input_root = Path(args.input_root)
    work_root = Path(args.work_root)

    if args.mode == "preflight-only":
        plan = analyze_resume(
            args.experiment_id,
            input_root=input_root,
            work_root=work_root,
        )

        preview_path = write_preview(
            plan,
            work_root=work_root,
        )

        print("=" * 100)
        print("COMB-01 GOVERNED RESUME PREFLIGHT")
        print("=" * 100)
        print(
            f"EXPERIMENT_ID={plan['experiment_id']}"
        )
        print(
            "RESUME_SESSION_INDEX="
            f"{plan['resume_session_index']}"
        )
        print(
            "TRAINING_SOURCE_COMMIT="
            f"{plan['training_source_commit']}"
        )
        print(
            "EXECUTION_COMMIT="
            f"{plan['execution_commit']}"
        )
        print(
            "PARENT_LAST_PT_SHA256="
            f"{plan['parent_checkpoint']['sha256']}"
        )
        print(
            "COMPLETED_EPOCHS="
            f"{plan['parent_checkpoint']['completed_epochs']}"
        )
        print(
            "NEXT_EPOCH_ONE_BASED="
            f"{plan['parent_checkpoint']['next_epoch_one_based']}"
        )
        print(
            f"TOTAL_EPOCHS={plan['total_epochs']}"
        )
        print(
            "EXPECTED_PARAMETERS="
            f"{plan['model_contract']['expected_parameters']}"
        )
        print(
            "EXPECTED_STATE_ITEMS="
            f"{plan['model_contract']['expected_state_items']}"
        )
        print(
            f"PREVIEW_MANIFEST={preview_path}"
        )
        print(
            "PREVIEW_SHA256="
            f"{br.sha256_file(preview_path)}"
        )
        print(
            "DESTINATION_RUN_MATERIALIZED=FALSE"
        )
        print("TEST_ACCESS=NONE")
        print("COMB01_RESUME_PREFLIGHT=PASS")
        print("=" * 100)
        return

    execute(
        args.experiment_id,
        input_root=input_root,
        work_root=work_root,
    )


if __name__ == "__main__":
    main()
