from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_HINT))

from research.runtime.baseline_runner import (
    DATA_BINDINGS_JSON,
    INIT_LOCK_JSON,
    EXPECTED_BRANCH,
    ROOT,
    TRAINING_YAML,
    GovernanceError,
    binding_expectation,
    build_training_args,
    discover_membership_directory,
    load_experiment,
    load_json,
    load_yaml,
    sha256_file,
    verify_git_provenance,
    verify_image_label_pair,
    verify_runtime_environment,
    write_runtime_data_yaml,
)


RESUME_CONTRACT_JSON = (
    ROOT / "research/05_experiments/RESUME01_RESUME_CONTRACT.json"
)

REQUIRED_PRIOR_RUN_FILES = [
    "args.yaml",
    "results.csv",
    "weights/last.pt",
    "weights/best.pt",
    "governance/PRETRAIN_PREFLIGHT.json",
    "governance/RUNTIME_MANIFEST.json",
]

SCIENTIFIC_ARG_KEYS = [
    "epochs",
    "patience",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "nbs",
    "cos_lr",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "seed",
    "deterministic",
    "amp",
    "multi_scale",
    "freeze",
    "cache",
    "single_cls",
    "rect",
    "fraction",
    "profile",
    "compile",
    "time",
    "box",
    "cls",
    "dfl",
    "fl_gamma",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "close_mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "val",
    "split",
    "conf",
    "iou",
    "max_det",
    "save_json",
    "plots",
    "workers",
    "save",
    "save_period",
]


def _path_is_test_like(path: Path) -> bool:
    return any(part.lower().startswith("test") for part in path.parts)


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(data, dict):
        raise GovernanceError(f"Expected JSON object: {path}")
    return data


def _tree_inventory(root: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GovernanceError(f"RESUME-01 rejects symlink inside run directory: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = "".join(
        f"{row['path']}|{row['bytes']}|{row['sha256']}\n"
        for row in rows
    ).encode("utf-8")
    import hashlib

    return rows, hashlib.sha256(payload).hexdigest()


def _discover_prior_run(input_root: Path, experiment_id: str) -> Path:
    candidates: list[Path] = []
    for current, dirs, _files in os.walk(input_root):
        current_path = Path(current)
        dirs[:] = [
            d
            for d in dirs
            if not d.lower().startswith("test") and d not in {".git", "__pycache__"}
        ]
        if _path_is_test_like(current_path):
            continue
        if current_path.name != experiment_id:
            continue
        if all((current_path / rel).is_file() for rel in REQUIRED_PRIOR_RUN_FILES):
            candidates.append(current_path.resolve())

    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise GovernanceError(
            "RESUME-01 expected exactly one complete prior run directory for "
            f"{experiment_id}; observed {unique}"
        )
    return unique[0]


def _existing_resume_indices(run_dir: Path) -> list[int]:
    sessions_dir = run_dir / "governance/resume_sessions"
    if not sessions_dir.exists():
        return []

    preflight: set[int] = set()
    runtime: set[int] = set()
    for path in sessions_dir.glob("RESUME_PREFLIGHT_*.json"):
        try:
            preflight.add(int(path.stem.rsplit("_", 1)[1]))
        except ValueError as exc:
            raise GovernanceError(f"Malformed RESUME-01 preflight filename: {path}") from exc
    for path in sessions_dir.glob("RESUME_RUNTIME_*.json"):
        try:
            runtime.add(int(path.stem.rsplit("_", 1)[1]))
        except ValueError as exc:
            raise GovernanceError(f"Malformed RESUME-01 runtime filename: {path}") from exc

    if preflight != runtime:
        raise GovernanceError(
            "RESUME-01 prior session provenance is incomplete: "
            f"preflight={sorted(preflight)}, runtime={sorted(runtime)}"
        )
    expected = list(range(1, len(preflight) + 1))
    observed = sorted(preflight)
    if observed != expected:
        raise GovernanceError(
            f"RESUME-01 prior session indices are not contiguous: {observed}"
        )
    return observed


def _read_results_state(results_csv: Path) -> dict:
    with results_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise GovernanceError("RESUME-01 results.csv contains no completed epoch.")
    if "epoch" not in rows[-1]:
        raise GovernanceError("RESUME-01 results.csv has no epoch column.")
    last_epoch_one_based = int(float(rows[-1]["epoch"]))
    return {
        "rows": len(rows),
        "last_epoch_one_based": last_epoch_one_based,
    }



def _verify_args_yaml_matches_checkpoint(
    args_yaml: Path,
    checkpoint_args: dict,
) -> None:
    observed = load_yaml(args_yaml)
    keys = [
        *SCIENTIFIC_ARG_KEYS,
        "project",
        "name",
        "save_dir",
        "data",
        "task",
        "single_cls",
    ]
    mismatches = {
        key: {
            "args_yaml": observed.get(key),
            "checkpoint": checkpoint_args.get(key),
        }
        for key in keys
        if observed.get(key) != checkpoint_args.get(key)
    }
    if mismatches:
        raise GovernanceError(
            "RESUME-01 args.yaml does not match last.pt train_args: "
            f"{mismatches}"
        )


def _verify_results_against_checkpoint(
    results_csv: Path,
    checkpoint_train_results: dict,
) -> dict:
    with results_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames or not rows:
        raise GovernanceError("RESUME-01 results.csv is empty or malformed.")
    if set(fieldnames) != set(checkpoint_train_results):
        raise GovernanceError(
            "RESUME-01 results.csv columns differ from last.pt train_results."
        )

    for column in fieldnames:
        expected_values = checkpoint_train_results.get(column)
        if not isinstance(expected_values, list):
            raise GovernanceError(
                f"RESUME-01 checkpoint train_results column is not a list: {column}"
            )
        if len(expected_values) != len(rows):
            raise GovernanceError(
                "RESUME-01 checkpoint train_results length mismatch for "
                f"{column}: {len(expected_values)} != {len(rows)}"
            )
        for index, row in enumerate(rows):
            try:
                observed = float(row[column])
                expected = float(expected_values[index])
            except (TypeError, ValueError) as exc:
                raise GovernanceError(
                    f"RESUME-01 non-numeric results evidence at {column}[{index}]."
                ) from exc
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise GovernanceError(
                    "RESUME-01 results.csv differs from checkpoint train_results at "
                    f"{column}[{index}]: {observed} != {expected}"
                )

    return {
        "rows": len(rows),
        "columns": fieldnames,
        "matches_checkpoint_train_results": True,
    }


def _load_checkpoint_metadata(last_pt: Path) -> tuple[dict, dict]:
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, resolved = torch_safe_load(str(last_pt))
    if Path(resolved).resolve() != last_pt.resolve():
        raise GovernanceError("RESUME-01 last.pt resolved to an unexpected file.")
    if not isinstance(ckpt, dict):
        raise GovernanceError("RESUME-01 last.pt is not a checkpoint dictionary.")

    required = [
        "epoch", "ema", "optimizer", "scaler", "train_args", "train_results",
        "git", "version",
    ]
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise GovernanceError(f"RESUME-01 checkpoint missing fields: {missing}")
    if ckpt["ema"] is None:
        raise GovernanceError("RESUME-01 checkpoint EMA model is missing.")
    if ckpt["optimizer"] is None:
        raise GovernanceError(
            "RESUME-01 checkpoint optimizer state is absent; checkpoint may be final/stripped."
        )
    if ckpt["scaler"] is None:
        raise GovernanceError("RESUME-01 checkpoint AMP scaler state is missing.")
    if not isinstance(ckpt["train_args"], dict):
        raise GovernanceError("RESUME-01 checkpoint train_args is not a dictionary.")
    if not isinstance(ckpt["train_results"], dict):
        raise GovernanceError("RESUME-01 checkpoint train_results is not a dictionary.")
    if not isinstance(ckpt["git"], dict):
        raise GovernanceError("RESUME-01 checkpoint git metadata is not a dictionary.")

    epoch = int(ckpt["epoch"])
    if epoch < 0:
        raise GovernanceError(
            "RESUME-01 checkpoint epoch is negative; completed/stripped checkpoints cannot resume."
        )

    ema = ckpt["ema"]
    state = ema.state_dict()
    parameter_count = sum(int(p.numel()) for p in ema.parameters())
    metadata = {
        "epoch_zero_based": epoch,
        "completed_epochs": epoch + 1,
        "next_epoch_one_based": epoch + 2,
        "state_items": len(state),
        "parameter_count": parameter_count,
        "git": ckpt.get("git") or {},
        "version": str(ckpt.get("version", "")),
        "train_args": ckpt["train_args"],
    }
    return ckpt, metadata


def _compare_frozen_training_args(
    checkpoint_args: dict,
    training: dict,
    *,
    experiment_id: str,
    runtime_yaml: Path,
) -> None:
    expected = build_training_args(
        training,
        experiment_id=experiment_id,
        runtime_data_yaml=runtime_yaml,
    )
    mismatches = {}
    for key in SCIENTIFIC_ARG_KEYS:
        if checkpoint_args.get(key) != expected.get(key):
            mismatches[key] = {
                "checkpoint": checkpoint_args.get(key),
                "expected": expected.get(key),
            }
    if mismatches:
        raise GovernanceError(
            "RESUME-01 checkpoint scientific training arguments drifted: "
            f"{mismatches}"
        )

    expected_run_dir = Path(training["output"]["project_dir"]) / experiment_id
    if checkpoint_args.get("project") != str(training["output"]["project_dir"]):
        raise GovernanceError("RESUME-01 checkpoint project directory drift.")
    if checkpoint_args.get("name") != experiment_id:
        raise GovernanceError("RESUME-01 checkpoint experiment name drift.")
    if Path(str(checkpoint_args.get("save_dir", ""))).resolve() != expected_run_dir.resolve():
        raise GovernanceError("RESUME-01 checkpoint save_dir drift.")
    if Path(str(checkpoint_args.get("data", ""))).resolve() != runtime_yaml.resolve():
        raise GovernanceError("RESUME-01 checkpoint runtime data-YAML path drift.")


def analyze_resume(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
) -> dict:
    training = load_yaml(TRAINING_YAML)
    data_manifest = load_json(DATA_BINDINGS_JSON)
    init_lock = load_json(INIT_LOCK_JSON)
    resume_contract = load_json(RESUME_CONTRACT_JSON)
    row = load_experiment(experiment_id)

    if training.get("status") != "FROZEN_BASELINE_RECIPE_V2":
        raise GovernanceError("RESUME-01 TRAINING.yaml status drift.")
    if resume_contract.get("status") != "IMPLEMENTED":
        raise GovernanceError("RESUME-01 contract status drift.")

    source_commit, execution_commit = verify_git_provenance(row)
    runtime = verify_runtime_environment(init_lock)
    expectation = binding_expectation(row, data_manifest)

    destination_run = Path(training["output"]["project_dir"]) / experiment_id
    if destination_run.exists():
        raise GovernanceError(
            f"RESUME-01 refuses existing destination run directory: {destination_run}"
        )

    source_run = _discover_prior_run(input_root, experiment_id)
    original_manifest_path = source_run / "governance/RUNTIME_MANIFEST.json"
    original_manifest = _json(original_manifest_path)
    original_preflight_path = source_run / "governance/PRETRAIN_PREFLIGHT.json"
    original_preflight = _json(original_preflight_path)

    if original_manifest.get("schema_version") != "TRAIN01-runtime-manifest-v1.0":
        raise GovernanceError("RESUME-01 original runtime-manifest schema drift.")
    if original_manifest.get("experiment_id") != experiment_id:
        raise GovernanceError("RESUME-01 original runtime experiment mismatch.")
    if original_manifest.get("training_source_commit") != source_commit:
        raise GovernanceError("RESUME-01 training-source commit mismatch.")
    if original_manifest.get("execution_commit") != execution_commit:
        raise GovernanceError(
            "RESUME-01 requires the exact same execution commit used by the original session."
        )
    if original_manifest.get("data_binding") != row["data_binding"]:
        raise GovernanceError("RESUME-01 data-binding mismatch.")
    if original_manifest.get("initialization") != row["initialization"]:
        raise GovernanceError("RESUME-01 initialization identity mismatch.")
    if original_manifest.get("test_split_present") is not False:
        raise GovernanceError("RESUME-01 original runtime manifest indicates test access.")

    if original_preflight.get("experiment_id") != experiment_id:
        raise GovernanceError("RESUME-01 original preflight experiment mismatch.")
    if original_preflight.get("training_source_commit") != source_commit:
        raise GovernanceError("RESUME-01 original preflight source-commit mismatch.")
    if original_preflight.get("execution_commit") != execution_commit:
        raise GovernanceError("RESUME-01 original preflight execution-commit mismatch.")
    if original_preflight.get("test_access", {}).get("test_predictions") is not False:
        raise GovernanceError("RESUME-01 original preflight test-prediction firewall drift.")
    if original_preflight.get("test_access", {}).get("test_metrics") is not False:
        raise GovernanceError("RESUME-01 original preflight test-metric firewall drift.")

    manifest_preflight_sha = original_manifest.get("preflight_manifest", {}).get("sha256")
    if manifest_preflight_sha != sha256_file(original_preflight_path):
        raise GovernanceError("RESUME-01 original PRETRAIN_PREFLIGHT hash mismatch.")

    train_images = discover_membership_directory(
        input_root,
        kind="images",
        expected_count=expectation["train_count"],
        expected_hash=expectation["train_hash"],
    )
    train_labels = discover_membership_directory(
        input_root,
        kind="labels",
        expected_count=expectation["train_count"],
        expected_hash=expectation["train_hash"],
    )
    validation_images = discover_membership_directory(
        input_root,
        kind="images",
        expected_count=expectation["validation_count"],
        expected_hash=expectation["validation_hash"],
    )
    validation_labels = discover_membership_directory(
        input_root,
        kind="labels",
        expected_count=expectation["validation_count"],
        expected_hash=expectation["validation_hash"],
    )
    verify_image_label_pair(train_images, train_labels)
    verify_image_label_pair(validation_images, validation_labels)

    canonical_preflight_dir = work_root / "ResEMA_preflight" / experiment_id
    canonical_preflight_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = canonical_preflight_dir / "runtime_data_train_val_only.yaml"
    runtime_yaml_sha = write_runtime_data_yaml(
        runtime_yaml,
        train_images=train_images,
        validation_images=validation_images,
    )

    original_runtime_yaml = original_preflight.get("runtime_data_yaml", {})
    if Path(str(original_runtime_yaml.get("path", ""))).resolve() != runtime_yaml.resolve():
        raise GovernanceError("RESUME-01 canonical runtime data-YAML path drift.")
    if original_runtime_yaml.get("sha256") != runtime_yaml_sha:
        raise GovernanceError(
            "RESUME-01 regenerated runtime data-YAML hash differs from the original session."
        )
    if original_runtime_yaml.get("contains_test_key") is not False:
        raise GovernanceError("RESUME-01 original runtime YAML test-key firewall drift.")

    last_pt = source_run / "weights/last.pt"
    best_pt = source_run / "weights/best.pt"
    args_yaml = source_run / "args.yaml"
    results_csv = source_run / "results.csv"

    ckpt, checkpoint = _load_checkpoint_metadata(last_pt)
    total_epochs = int(training["training"]["epochs"])
    if checkpoint["completed_epochs"] >= total_epochs:
        raise GovernanceError(
            "RESUME-01 prior run has already reached the frozen epoch budget."
        )
    if checkpoint["state_items"] != 499:
        raise GovernanceError(
            f"RESUME-01 checkpoint state-item mismatch: {checkpoint['state_items']} != 499"
        )
    if checkpoint["parameter_count"] != 9_431_275:
        raise GovernanceError(
            "RESUME-01 checkpoint parameter-count mismatch: "
            f"{checkpoint['parameter_count']} != 9431275"
        )

    checkpoint_git = checkpoint["git"]
    if checkpoint_git.get("commit") != execution_commit:
        raise GovernanceError(
            "RESUME-01 checkpoint Git commit does not equal the frozen execution commit."
        )
    if checkpoint_git.get("branch") != EXPECTED_BRANCH:
        raise GovernanceError(
            "RESUME-01 checkpoint Git branch does not match the governed branch."
        )
    if checkpoint["version"] != init_lock["exact_fork_runtime"]["ultralytics_version"]:
        raise GovernanceError(
            "RESUME-01 checkpoint Ultralytics version differs from the frozen runtime."
        )

    _compare_frozen_training_args(
        checkpoint["train_args"],
        training,
        experiment_id=experiment_id,
        runtime_yaml=runtime_yaml,
    )
    _verify_args_yaml_matches_checkpoint(args_yaml, checkpoint["train_args"])

    checkpoint_results = _verify_results_against_checkpoint(
        results_csv,
        ckpt["train_results"],
    )

    best_ckpt, best_checkpoint = _load_checkpoint_metadata(best_pt)
    if best_checkpoint["state_items"] != 499 or best_checkpoint["parameter_count"] != 9_431_275:
        raise GovernanceError("RESUME-01 best.pt model identity mismatch.")
    if best_checkpoint["git"].get("commit") != execution_commit:
        raise GovernanceError("RESUME-01 best.pt Git commit mismatch.")
    if best_checkpoint["git"].get("branch") != EXPECTED_BRANCH:
        raise GovernanceError("RESUME-01 best.pt Git branch mismatch.")
    if best_checkpoint["version"] != init_lock["exact_fork_runtime"]["ultralytics_version"]:
        raise GovernanceError("RESUME-01 best.pt Ultralytics version mismatch.")
    _compare_frozen_training_args(
        best_checkpoint["train_args"],
        training,
        experiment_id=experiment_id,
        runtime_yaml=runtime_yaml,
    )
    if best_checkpoint["completed_epochs"] > checkpoint["completed_epochs"]:
        raise GovernanceError("RESUME-01 best.pt epoch is later than last.pt.")
    del best_ckpt

    results_state = _read_results_state(results_csv)
    if results_state["rows"] != checkpoint["completed_epochs"]:
        raise GovernanceError(
            "RESUME-01 results.csv row count does not match completed checkpoint epochs."
        )
    if results_state["last_epoch_one_based"] != checkpoint["completed_epochs"]:
        raise GovernanceError(
            "RESUME-01 results.csv last epoch does not match last.pt epoch."
        )

    prior_indices = _existing_resume_indices(source_run)
    session_index = len(prior_indices) + 1
    source_inventory, source_tree_sha = _tree_inventory(source_run)

    plan = {
        "schema_version": "RESUME01-preflight-v1.0",
        "experiment_id": experiment_id,
        "resume_session_index": session_index,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": row["data_binding"],
        "initialization": row["initialization"],
        "test_access": {
            "runtime_yaml_contains_test": False,
            "test_directory_scan_pruned": True,
            "test_predictions": False,
            "test_metrics": False,
            "test_error_analysis": False,
        },
        "operational_expected": {
            "train_images": expectation["operational_train_images"],
            "validation_images": expectation["operational_validation_images"],
        },
        "membership": {
            "train_images": {
                "path": str(train_images),
                "count": expectation["train_count"],
                "sha256_final_newline": expectation["train_hash"],
            },
            "train_labels": {
                "path": str(train_labels),
                "count": expectation["train_count"],
                "sha256_final_newline": expectation["train_hash"],
            },
            "validation_images": {
                "path": str(validation_images),
                "count": expectation["validation_count"],
                "sha256_final_newline": expectation["validation_hash"],
            },
            "validation_labels": {
                "path": str(validation_labels),
                "count": expectation["validation_count"],
                "sha256_final_newline": expectation["validation_hash"],
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
        "original_runtime_manifest_sha256": sha256_file(original_manifest_path),
        "original_pretrain_preflight_sha256": sha256_file(original_preflight_path),
        "parent_checkpoint": {
            "path_before_copy": str(last_pt),
            "sha256": sha256_file(last_pt),
            "bytes": last_pt.stat().st_size,
            "epoch_zero_based": checkpoint["epoch_zero_based"],
            "completed_epochs": checkpoint["completed_epochs"],
            "next_epoch_one_based": checkpoint["next_epoch_one_based"],
        },
        "prior_artifacts": {
            "best_pt_sha256": sha256_file(best_pt),
            "args_yaml_sha256": sha256_file(args_yaml),
            "results_csv_sha256": sha256_file(results_csv),
            "results_rows": results_state["rows"],
            "results_match_checkpoint_train_results": checkpoint_results[
                "matches_checkpoint_train_results"
            ],
            "best_pt_epoch_zero_based": best_checkpoint["epoch_zero_based"],
        },
        "total_epochs": total_epochs,
        "runtime": runtime,
        "source_tree_inventory": source_inventory,
    }
    del ckpt
    return plan


def materialize_resume(plan: dict) -> tuple[dict, Path, Path]:
    source_run = Path(plan["source_run"]["path"])
    destination_run = Path(plan["destination_run"])
    if destination_run.exists():
        raise GovernanceError(
            f"RESUME-01 refuses existing destination: {destination_run}"
        )

    source_inventory_before, source_tree_sha_before = _tree_inventory(source_run)
    if source_tree_sha_before != plan["source_run"]["tree_sha256"]:
        raise GovernanceError("RESUME-01 source run changed after preflight analysis.")

    shutil.copytree(source_run, destination_run, copy_function=shutil.copy2)
    destination_inventory, destination_tree_sha = _tree_inventory(destination_run)
    if destination_inventory != source_inventory_before:
        shutil.rmtree(destination_run, ignore_errors=True)
        raise GovernanceError("RESUME-01 full-run copy verification failed.")
    if destination_tree_sha != source_tree_sha_before:
        shutil.rmtree(destination_run, ignore_errors=True)
        raise GovernanceError("RESUME-01 copied run tree digest mismatch.")

    last_pt = destination_run / "weights/last.pt"
    if sha256_file(last_pt) != plan["parent_checkpoint"]["sha256"]:
        shutil.rmtree(destination_run, ignore_errors=True)
        raise GovernanceError("RESUME-01 copied last.pt hash mismatch.")

    session_index = int(plan["resume_session_index"])
    sessions_dir = destination_run / "governance/resume_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    args_source = destination_run / "args.yaml"
    archived_args = sessions_dir / f"ARGS_BEFORE_RESUME_{session_index:03d}.yaml"
    if archived_args.exists():
        shutil.rmtree(destination_run, ignore_errors=True)
        raise GovernanceError("RESUME-01 args archive collision.")
    shutil.copy2(args_source, archived_args)

    materialized = dict(plan)
    materialized["copy_verification"] = {
        "source_tree_sha256": source_tree_sha_before,
        "destination_tree_sha256_before_session_append": destination_tree_sha,
        "exact_file_inventory_match": True,
    }
    materialized["parent_checkpoint"] = dict(plan["parent_checkpoint"])
    materialized["parent_checkpoint"]["path_after_copy"] = str(last_pt)
    materialized["archived_args_before_resume"] = {
        "path": str(archived_args),
        "sha256": sha256_file(archived_args),
        "bytes": archived_args.stat().st_size,
    }

    preflight_path = sessions_dir / f"RESUME_PREFLIGHT_{session_index:03d}.json"
    if preflight_path.exists():
        shutil.rmtree(destination_run, ignore_errors=True)
        raise GovernanceError("RESUME-01 preflight session collision.")
    preflight_path.write_text(
        json.dumps(materialized, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return materialized, preflight_path, last_pt


def write_preview(plan: dict, *, work_root: Path) -> Path:
    preview_dir = work_root / "ResEMA_resume_preflight" / plan["experiment_id"]
    preview_dir.mkdir(parents=True, exist_ok=True)
    path = preview_dir / "RESUME_PREFLIGHT_PREVIEW.json"
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
    materialized, preflight_path, last_pt = materialize_resume(plan)

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics_config"
    )
    runtime_yaml = Path(materialized["runtime_data_yaml"]["path"])
    if not runtime_yaml.is_file():
        raise GovernanceError("RESUME-01 runtime data YAML disappeared before launch.")
    if sha256_file(runtime_yaml) != materialized["runtime_data_yaml"]["sha256"]:
        raise GovernanceError("RESUME-01 runtime data YAML changed before launch.")
    preflight_sha = sha256_file(preflight_path)

    os.environ["RESEMA_EXPERIMENT_ID"] = experiment_id
    os.environ["RESEMA_INITIALIZATION"] = materialized["initialization"]
    os.environ["RESEMA_RESUME_PREFLIGHT_MANIFEST"] = str(preflight_path)
    os.environ["RESEMA_RESUME_PREFLIGHT_SHA256"] = preflight_sha
    os.environ["RESEMA_REPO_ROOT"] = str(ROOT.resolve())

    from ultralytics import YOLO
    from ultralytics.research import GovernedDetectionTrainer

    model = YOLO(str(last_pt), task="detect")
    model.train(
        trainer=GovernedDetectionTrainer,
        resume=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Governed RESUME-01 continuation launcher. "
            "Scientific hyperparameter overrides are not accepted."
        )
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        choices=[
            "BASE-B-ORG-PT-S42",
            "BASE-B-ORG-SCR-S42",
            "BASE-B-AUG-PT-S42",
            "BASE-B-AUG-SCR-S42",
            "BASE-A-AUG-PT-S42",
            "BASE-A-AUG-SCR-S42",
        ],
    )
    parser.add_argument(
        "--mode",
        choices=["preflight-only", "execute"],
        default="preflight-only",
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work-root", default="/kaggle/working")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    work_root = Path(args.work_root)

    if args.mode == "preflight-only":
        plan = analyze_resume(
            args.experiment_id,
            input_root=input_root,
            work_root=work_root,
        )
        preview_path = write_preview(plan, work_root=work_root)
        print("=" * 100)
        print("RESUME-01 GOVERNED PREFLIGHT")
        print("=" * 100)
        print(f"EXPERIMENT_ID={plan['experiment_id']}")
        print(f"RESUME_SESSION_INDEX={plan['resume_session_index']}")
        print(f"TRAINING_SOURCE_COMMIT={plan['training_source_commit']}")
        print(f"EXECUTION_COMMIT={plan['execution_commit']}")
        print(f"PARENT_LAST_PT_SHA256={plan['parent_checkpoint']['sha256']}")
        print(f"COMPLETED_EPOCHS={plan['parent_checkpoint']['completed_epochs']}")
        print(f"NEXT_EPOCH_ONE_BASED={plan['parent_checkpoint']['next_epoch_one_based']}")
        print(f"TOTAL_EPOCHS={plan['total_epochs']}")
        print(f"PREVIEW_MANIFEST={preview_path}")
        print(f"PREVIEW_SHA256={sha256_file(preview_path)}")
        print("DESTINATION_RUN_MATERIALIZED=FALSE")
        print("TEST_ACCESS=NONE")
        print("RESUME01_PREFLIGHT=PASS")
        print("=" * 100)
        return

    execute(
        args.experiment_id,
        input_root=input_root,
        work_root=work_root,
    )


if __name__ == "__main__":
    main()
