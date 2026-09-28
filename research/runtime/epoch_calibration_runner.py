from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# When this file is executed directly (python research/runtime/epoch_calibration_runner.py),
# Python places research/runtime on sys.path rather than the repository root. Bootstrap
# the repository root before importing the project-local research namespace.
_REPO_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_HINT))

from research.runtime import baseline_runner as br
from research.runtime import resume_runner as rr


ROOT = Path(__file__).resolve().parents[2]
CAL_YAML = ROOT / "research/05_experiments/EPOCH_CALIBRATION.yaml"
CAL_MATRIX = ROOT / "research/05_experiments/EPOCH_CALIBRATION_EXPERIMENTS.csv"
CAL_CONTRACT = ROOT / "research/05_experiments/EPOCH_CALIBRATION_CONTRACT.json"
TRAINING_YAML = ROOT / "research/05_experiments/TRAINING.yaml"
DATA_BINDINGS_JSON = ROOT / "research/04_data/manifests/DATA01_ACTIVE_DATASET_BINDINGS.json"
INIT_LOCK_JSON = ROOT / "research/01_provenance/INIT01_INITIALIZATION_LOCK.json"

EXPECTED_BRANCH = "research/baseline-refresh"
EXPERIMENT_ID = "BASE-B-ORG-SCR-S42-E200-CAL"

SOURCE_GUARD_PATHS = [
    "ultralytics",
    "research/runtime/baseline_runner.py",
    "research/runtime/resume_runner.py",
    "research/runtime/epoch_calibration_runner.py",
    "research/05_experiments/TRAINING.yaml",
    "research/05_experiments/EPOCH_CALIBRATION.yaml",
    "research/05_experiments/EPOCH_CALIBRATION_CONTRACT.json",
    "research/01_provenance/INIT01_INITIALIZATION_LOCK.json",
    "research/04_data/manifests/DATA01_ACTIVE_DATASET_BINDINGS.json",
    "ultralytics/research/baseline_trainer.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/models/yolo/detect/train.py",
    "ultralytics/cfg/models/11/yolo11s.yaml",
]


class CalibrationError(br.GovernanceError):
    pass


def _read_matrix_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    fields = list(reader.fieldnames or [])
    rows = list(reader)
    if not fields or len(rows) != 1:
        raise CalibrationError("EPOCH-CAL matrix must contain exactly one row.")
    if rows[0].get("experiment_id") != EXPERIMENT_ID:
        raise CalibrationError("EPOCH-CAL experiment identity drift.")
    return fields, rows


def _read_matrix() -> tuple[list[str], list[dict[str, str]]]:
    return _read_matrix_text(CAL_MATRIX.read_text(encoding="utf-8-sig", errors="strict"))


def load_calibration_row() -> dict[str, str]:
    _fields, rows = _read_matrix()
    row = rows[0]
    required = {
        "status": "NOT_STARTED",
        "comparison_parent": "BASE-B-ORG-SCR-S42",
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "scratch",
        "seed": "42",
        "epochs": "200",
        "patience": "200",
        "imgsz": "1024",
        "batch": "16",
        "optimizer": "SGD",
        "primary_metric": "val_mAP50-95",
        "test_access": "NONE",
    }
    for key, expected in required.items():
        if row.get(key) != expected:
            raise CalibrationError(f"EPOCH-CAL matrix drift: {key}={row.get(key)!r} != {expected!r}")
    return row


def verify_atomic_source_binding(source_commit: str) -> None:
    frozen_text = br.git("show", f"{source_commit}:research/05_experiments/EPOCH_CALIBRATION_EXPERIMENTS.csv")
    frozen_fields, frozen_rows = _read_matrix_text(frozen_text)
    current_fields, current_rows = _read_matrix()
    if frozen_fields != current_fields:
        raise CalibrationError("EPOCH-CAL matrix schema changed after source freeze.")
    frozen = frozen_rows[0]
    current = current_rows[0]
    if frozen["source_commit"].strip():
        raise CalibrationError("Frozen EPOCH-CAL source must contain blank source_commit.")
    if current["source_commit"].strip() != source_commit:
        raise CalibrationError("EPOCH-CAL authorization does not bind the frozen source commit.")
    for field in frozen_fields:
        if field == "source_commit":
            continue
        if frozen.get(field) != current.get(field):
            raise CalibrationError(
                f"EPOCH-CAL matrix changed outside source_commit: {field}: "
                f"{frozen.get(field)!r} != {current.get(field)!r}"
            )


def verify_git_provenance(row: dict[str, str]) -> tuple[str, str]:
    branch = br.git("branch", "--show-current")
    head = br.git("rev-parse", "HEAD")
    dirty = br.git("status", "--porcelain=v1", "--untracked-files=all")
    if branch != EXPECTED_BRANCH:
        raise CalibrationError(f"Wrong branch: {branch!r}")
    if dirty:
        raise CalibrationError("EPOCH-CAL requires a clean Git worktree.")

    source_commit = row["source_commit"].strip()
    if not source_commit:
        raise CalibrationError("EPOCH-CAL remains locked: source_commit is blank.")

    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", source_commit, head],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise CalibrationError("Frozen EPOCH-CAL source is not an ancestor of execution commit.")

    verify_atomic_source_binding(source_commit)

    diff = br.git("diff", "--name-only", source_commit, head, "--", *SOURCE_GUARD_PATHS)
    if diff:
        raise CalibrationError("Scientific EPOCH-CAL source changed after freeze:\n" + diff)

    return source_commit, head


def effective_training() -> tuple[dict, dict, dict]:
    baseline = br.load_yaml(TRAINING_YAML)
    cal = br.load_yaml(CAL_YAML)
    contract = br.load_json(CAL_CONTRACT)

    if baseline.get("status") != "FROZEN_BASELINE_RECIPE_V2":
        raise CalibrationError("Parent baseline recipe status drift.")
    if cal.get("status") != "FROZEN_EPOCH_CALIBRATION_RECIPE_V1":
        raise CalibrationError("EPOCH-CAL recipe status drift.")
    if contract.get("status") != "IMPLEMENTED_RESUME_AWARE":
        raise CalibrationError("EPOCH-CAL contract status drift.")

    effective = copy.deepcopy(baseline)
    effective["training"]["epochs"] = 200
    effective["training"]["patience"] = 200
    effective["output"]["project_dir"] = "/kaggle/working/ResEMA_epoch_calibration_runs"

    if effective["training"]["seed"] != 42:
        raise CalibrationError("Seed drift.")
    if effective["training"]["imgsz"] != 1024 or effective["training"]["batch_global"] != 16:
        raise CalibrationError("Image-size/batch drift.")
    if effective["training"]["optimizer"] != "SGD":
        raise CalibrationError("Optimizer drift.")
    if effective["training"]["cos_lr"] is not True:
        raise CalibrationError("Cosine-scheduler contract drift.")
    if effective["augmentation"]["close_mosaic"] != 10:
        raise CalibrationError("close_mosaic drift.")
    if effective["validation"]["split"] != "val":
        raise CalibrationError("Validation split drift.")

    return effective, cal, contract


def expectation_and_membership(input_root: Path) -> tuple[dict, Path, Path, Path, Path]:
    manifest = br.load_json(DATA_BINDINGS_JSON)
    expectation = br.binding_expectation({"data_binding": "DATA01:B-ORG:v1"}, manifest)
    train_images = br.discover_membership_directory(
        input_root, kind="images",
        expected_count=expectation["train_count"], expected_hash=expectation["train_hash"],
    )
    train_labels = br.discover_membership_directory(
        input_root, kind="labels",
        expected_count=expectation["train_count"], expected_hash=expectation["train_hash"],
    )
    val_images = br.discover_membership_directory(
        input_root, kind="images",
        expected_count=expectation["validation_count"], expected_hash=expectation["validation_hash"],
    )
    val_labels = br.discover_membership_directory(
        input_root, kind="labels",
        expected_count=expectation["validation_count"], expected_hash=expectation["validation_hash"],
    )
    br.verify_image_label_pair(train_images, train_labels)
    br.verify_image_label_pair(val_images, val_labels)
    return expectation, train_images, train_labels, val_images, val_labels


def fresh_preflight(*, input_root: Path, work_root: Path) -> tuple[dict, Path]:
    row = load_calibration_row()
    training, _cal, _contract = effective_training()
    init_lock = br.load_json(INIT_LOCK_JSON)
    source_commit, execution_commit = verify_git_provenance(row)
    runtime = br.verify_runtime_environment(init_lock)
    expectation, train_images, train_labels, val_images, val_labels = expectation_and_membership(input_root)

    preflight_dir = work_root / "ResEMA_epoch_calibration_preflight" / EXPERIMENT_ID
    preflight_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = preflight_dir / "runtime_data_train_val_only.yaml"
    runtime_yaml_sha = br.write_runtime_data_yaml(
        runtime_yaml, train_images=train_images, validation_images=val_images
    )

    run_dir = Path(training["output"]["project_dir"]) / EXPERIMENT_ID
    if run_dir.exists():
        raise CalibrationError(f"Fresh EPOCH-CAL refuses existing run directory: {run_dir}")

    preflight = {
        "schema_version": "EPOCHCAL01-preflight-v1.0",
        "experiment_id": EXPERIMENT_ID,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "scratch",
        "comparison_parent": "BASE-B-ORG-SCR-S42",
        "budget": {
            "fresh_training": True,
            "resume_from_parent_baseline": False,
            "epochs": 200,
            "patience": 200,
            "cos_lr": True,
            "close_mosaic": 10,
            "interpretation": "fresh 200-epoch recipe budget; not a pure continuation of the 100-epoch trajectory",
        },
        "test_access": {
            "runtime_yaml_contains_test": False,
            "test_directory_scan_pruned": True,
            "test_predictions": False,
            "test_metrics": False,
        },
        "membership": {
            "train_images": {"path": str(train_images), "count": expectation["train_count"], "sha256_final_newline": expectation["train_hash"]},
            "train_labels": {"path": str(train_labels), "count": expectation["train_count"], "sha256_final_newline": expectation["train_hash"]},
            "validation_images": {"path": str(val_images), "count": expectation["validation_count"], "sha256_final_newline": expectation["validation_hash"]},
            "validation_labels": {"path": str(val_labels), "count": expectation["validation_count"], "sha256_final_newline": expectation["validation_hash"]},
        },
        "operational_expected": {
            "train_images": expectation["operational_train_images"],
            "validation_images": expectation["operational_validation_images"],
        },
        "runtime_data_yaml": {"path": str(runtime_yaml), "sha256": runtime_yaml_sha, "contains_test_key": False},
        "initial_checkpoint": None,
        "frozen_inputs": {
            "TRAINING_yaml_sha256": br.sha256_file(TRAINING_YAML),
            "EPOCH_CALIBRATION_yaml_sha256": br.sha256_file(CAL_YAML),
            "EPOCH_CALIBRATION_contract_sha256": br.sha256_file(CAL_CONTRACT),
            "INIT01_lock_sha256": br.sha256_file(INIT_LOCK_JSON),
            "DATA01_bindings_sha256": br.sha256_file(DATA_BINDINGS_JSON),
        },
        "runtime": runtime,
        "output": {"run_dir": str(run_dir), "run_dir_exists_before_launch": False},
    }
    path = preflight_dir / "PRETRAIN_PREFLIGHT.json"
    path.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8", newline="\n")
    return preflight, path


def fresh_execute(*, input_root: Path, work_root: Path) -> None:
    preflight, preflight_path = fresh_preflight(input_root=input_root, work_root=work_root)
    training, _cal, _contract = effective_training()

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics_config")
    os.environ["RESEMA_EXPERIMENT_ID"] = EXPERIMENT_ID
    os.environ["RESEMA_INITIALIZATION"] = "scratch"
    os.environ["RESEMA_PREFLIGHT_MANIFEST"] = str(preflight_path)
    os.environ["RESEMA_PREFLIGHT_SHA256"] = br.sha256_file(preflight_path)
    os.environ["RESEMA_REPO_ROOT"] = str(ROOT.resolve())

    from ultralytics import YOLO
    from ultralytics.research import GovernedDetectionTrainer

    model = YOLO(str(ROOT / "ultralytics/cfg/models/11/yolo11s.yaml"), task="detect")
    args = br.build_training_args(
        training,
        experiment_id=EXPERIMENT_ID,
        runtime_data_yaml=Path(preflight["runtime_data_yaml"]["path"]),
    )
    if args["epochs"] != 200 or args["patience"] != 200:
        raise CalibrationError("Effective 200-epoch budget drift.")
    model.train(trainer=GovernedDetectionTrainer, **args)


def _compare_checkpoint_args(checkpoint_args: dict, training: dict, runtime_yaml: Path) -> None:
    expected = br.build_training_args(
        training, experiment_id=EXPERIMENT_ID, runtime_data_yaml=runtime_yaml
    )
    mismatches = {}
    for key in rr.SCIENTIFIC_ARG_KEYS:
        if checkpoint_args.get(key) != expected.get(key):
            mismatches[key] = {"checkpoint": checkpoint_args.get(key), "expected": expected.get(key)}
    if mismatches:
        raise CalibrationError(f"EPOCH-CAL resume scientific args drift: {mismatches}")

    expected_run_dir = Path(training["output"]["project_dir"]) / EXPERIMENT_ID
    if checkpoint_args.get("project") != str(training["output"]["project_dir"]):
        raise CalibrationError("EPOCH-CAL resume project drift.")
    if checkpoint_args.get("name") != EXPERIMENT_ID:
        raise CalibrationError("EPOCH-CAL resume experiment-name drift.")
    if Path(str(checkpoint_args.get("save_dir", ""))).resolve() != expected_run_dir.resolve():
        raise CalibrationError("EPOCH-CAL resume save_dir drift.")
    if Path(str(checkpoint_args.get("data", ""))).resolve() != runtime_yaml.resolve():
        raise CalibrationError("EPOCH-CAL resume runtime-data YAML drift.")


def resume_plan(*, input_root: Path, work_root: Path) -> dict:
    row = load_calibration_row()
    training, _cal, _contract = effective_training()
    init_lock = br.load_json(INIT_LOCK_JSON)
    source_commit, execution_commit = verify_git_provenance(row)
    runtime = br.verify_runtime_environment(init_lock)
    expectation, train_images, train_labels, val_images, val_labels = expectation_and_membership(input_root)

    destination_run = Path(training["output"]["project_dir"]) / EXPERIMENT_ID
    if destination_run.exists():
        raise CalibrationError(f"EPOCH-CAL resume refuses existing destination: {destination_run}")

    source_run = rr._discover_prior_run(input_root, EXPERIMENT_ID)
    original_manifest_path = source_run / "governance/RUNTIME_MANIFEST.json"
    original_preflight_path = source_run / "governance/PRETRAIN_PREFLIGHT.json"
    original_manifest = rr._json(original_manifest_path)
    original_preflight = rr._json(original_preflight_path)

    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "scratch",
    }.items():
        if original_manifest.get(key) != expected:
            raise CalibrationError(f"EPOCH-CAL original runtime manifest mismatch: {key}")
        if original_preflight.get(key) != expected:
            raise CalibrationError(f"EPOCH-CAL original preflight mismatch: {key}")

    if original_manifest.get("test_split_present") is not False:
        raise CalibrationError("EPOCH-CAL original runtime indicates test access.")
    test_access = original_preflight.get("test_access", {})
    if test_access.get("test_predictions") is not False or test_access.get("test_metrics") is not False:
        raise CalibrationError("EPOCH-CAL original preflight test firewall drift.")
    if original_manifest.get("preflight_manifest", {}).get("sha256") != br.sha256_file(original_preflight_path):
        raise CalibrationError("EPOCH-CAL original preflight hash mismatch.")

    preflight_dir = work_root / "ResEMA_epoch_calibration_preflight" / EXPERIMENT_ID
    preflight_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = preflight_dir / "runtime_data_train_val_only.yaml"
    runtime_yaml_sha = br.write_runtime_data_yaml(
        runtime_yaml, train_images=train_images, validation_images=val_images
    )
    original_runtime_yaml = original_preflight.get("runtime_data_yaml", {})
    if Path(str(original_runtime_yaml.get("path", ""))).resolve() != runtime_yaml.resolve():
        raise CalibrationError("EPOCH-CAL resume runtime YAML path drift.")
    if original_runtime_yaml.get("sha256") != runtime_yaml_sha:
        raise CalibrationError("EPOCH-CAL resume runtime YAML hash drift.")

    last_pt = source_run / "weights/last.pt"
    best_pt = source_run / "weights/best.pt"
    args_yaml = source_run / "args.yaml"
    results_csv = source_run / "results.csv"

    ckpt, checkpoint = rr._load_checkpoint_metadata(last_pt)
    if checkpoint["completed_epochs"] >= 200:
        raise CalibrationError("EPOCH-CAL prior run already reached 200 epochs.")
    if checkpoint["state_items"] != 499 or checkpoint["parameter_count"] != 9_431_275:
        raise CalibrationError("EPOCH-CAL checkpoint model identity drift.")
    if checkpoint["git"].get("commit") != execution_commit:
        raise CalibrationError("EPOCH-CAL checkpoint Git commit drift.")
    if checkpoint["git"].get("branch") != EXPECTED_BRANCH:
        raise CalibrationError("EPOCH-CAL checkpoint branch drift.")
    if checkpoint["version"] != init_lock["exact_fork_runtime"]["ultralytics_version"]:
        raise CalibrationError("EPOCH-CAL checkpoint Ultralytics version drift.")

    _compare_checkpoint_args(checkpoint["train_args"], training, runtime_yaml)
    rr._verify_args_yaml_matches_checkpoint(args_yaml, checkpoint["train_args"])
    checkpoint_results = rr._verify_results_against_checkpoint(results_csv, ckpt["train_results"])

    best_ckpt, best_meta = rr._load_checkpoint_metadata(best_pt)
    if best_meta["state_items"] != 499 or best_meta["parameter_count"] != 9_431_275:
        raise CalibrationError("EPOCH-CAL best.pt model identity drift.")
    if best_meta["git"].get("commit") != execution_commit:
        raise CalibrationError("EPOCH-CAL best.pt Git commit drift.")
    _compare_checkpoint_args(best_meta["train_args"], training, runtime_yaml)
    if best_meta["completed_epochs"] > checkpoint["completed_epochs"]:
        raise CalibrationError("EPOCH-CAL best.pt later than last.pt.")
    del best_ckpt

    results_state = rr._read_results_state(results_csv)
    if results_state["rows"] != checkpoint["completed_epochs"]:
        raise CalibrationError("EPOCH-CAL results row count/checkpoint epoch mismatch.")
    if results_state["last_epoch_one_based"] != checkpoint["completed_epochs"]:
        raise CalibrationError("EPOCH-CAL results final epoch/checkpoint mismatch.")

    prior_indices = rr._existing_resume_indices(source_run)
    inventory, tree_sha = rr._tree_inventory(source_run)
    plan = {
        "schema_version": "EPOCHCAL01-resume-preflight-v1.0",
        "experiment_id": EXPERIMENT_ID,
        "resume_session_index": len(prior_indices) + 1,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": "DATA01:B-ORG:v1",
        "initialization": "scratch",
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
            "train_images": {"path": str(train_images), "count": expectation["train_count"], "sha256_final_newline": expectation["train_hash"]},
            "train_labels": {"path": str(train_labels), "count": expectation["train_count"], "sha256_final_newline": expectation["train_hash"]},
            "validation_images": {"path": str(val_images), "count": expectation["validation_count"], "sha256_final_newline": expectation["validation_hash"]},
            "validation_labels": {"path": str(val_labels), "count": expectation["validation_count"], "sha256_final_newline": expectation["validation_hash"]},
        },
        "runtime_data_yaml": {
            "path": str(runtime_yaml), "sha256": runtime_yaml_sha,
            "matches_original_session_hash": True, "contains_test_key": False,
        },
        "source_run": {"path": str(source_run), "tree_sha256": tree_sha, "file_count": len(inventory)},
        "destination_run": str(destination_run),
        "original_runtime_manifest_sha256": br.sha256_file(original_manifest_path),
        "original_pretrain_preflight_sha256": br.sha256_file(original_preflight_path),
        "parent_checkpoint": {
            "path_before_copy": str(last_pt),
            "sha256": br.sha256_file(last_pt),
            "bytes": last_pt.stat().st_size,
            "epoch_zero_based": checkpoint["epoch_zero_based"],
            "completed_epochs": checkpoint["completed_epochs"],
            "next_epoch_one_based": checkpoint["next_epoch_one_based"],
        },
        "prior_artifacts": {
            "best_pt_sha256": br.sha256_file(best_pt),
            "args_yaml_sha256": br.sha256_file(args_yaml),
            "results_csv_sha256": br.sha256_file(results_csv),
            "results_rows": results_state["rows"],
            "results_match_checkpoint_train_results": checkpoint_results["matches_checkpoint_train_results"],
            "best_pt_epoch_zero_based": best_meta["epoch_zero_based"],
        },
        "total_epochs": 200,
        "runtime": runtime,
        "source_tree_inventory": inventory,
    }
    del ckpt
    return plan


def resume_execute(*, input_root: Path, work_root: Path) -> None:
    plan = resume_plan(input_root=input_root, work_root=work_root)
    materialized, preflight_path, last_pt = rr.materialize_resume(plan)

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics_config")
    os.environ["RESEMA_EXPERIMENT_ID"] = EXPERIMENT_ID
    os.environ["RESEMA_INITIALIZATION"] = "scratch"
    os.environ["RESEMA_RESUME_PREFLIGHT_MANIFEST"] = str(preflight_path)
    os.environ["RESEMA_RESUME_PREFLIGHT_SHA256"] = br.sha256_file(preflight_path)
    os.environ["RESEMA_REPO_ROOT"] = str(ROOT.resolve())

    from ultralytics import YOLO
    from ultralytics.research import GovernedDetectionTrainer

    model = YOLO(str(last_pt), task="detect")
    model.train(trainer=GovernedDetectionTrainer, resume=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Governed EPOCH-CAL-01 launcher.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["fresh-preflight", "fresh-execute", "resume-preflight", "resume-execute"],
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work-root", default="/kaggle/working")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    work_root = Path(args.work_root)

    if args.mode == "fresh-preflight":
        preflight, path = fresh_preflight(input_root=input_root, work_root=work_root)
        print("=" * 100)
        print("EPOCH-CAL-01 FRESH PREFLIGHT")
        print("=" * 100)
        print(f"EXPERIMENT_ID={EXPERIMENT_ID}")
        print(f"TRAINING_SOURCE_COMMIT={preflight['training_source_commit']}")
        print(f"EXECUTION_COMMIT={preflight['execution_commit']}")
        print("EPOCHS=200")
        print("PATIENCE=200")
        print("INITIALIZATION=scratch")
        print("DATA_BINDING=DATA01:B-ORG:v1")
        print(f"PREFLIGHT_MANIFEST={path}")
        print(f"PREFLIGHT_SHA256={br.sha256_file(path)}")
        print("TRAINING_STARTED=FALSE")
        print("TEST_ACCESS=NONE")
        print("EPOCH_CALIBRATION_PREFLIGHT=PASS")
        print("=" * 100)
        return

    if args.mode == "fresh-execute":
        fresh_execute(input_root=input_root, work_root=work_root)
        return

    if args.mode == "resume-preflight":
        plan = resume_plan(input_root=input_root, work_root=work_root)
        preview = rr.write_preview(plan, work_root=work_root)
        print("=" * 100)
        print("EPOCH-CAL-01 RESUME PREFLIGHT")
        print("=" * 100)
        print(f"EXPERIMENT_ID={EXPERIMENT_ID}")
        print(f"RESUME_SESSION_INDEX={plan['resume_session_index']}")
        print(f"TRAINING_SOURCE_COMMIT={plan['training_source_commit']}")
        print(f"EXECUTION_COMMIT={plan['execution_commit']}")
        print(f"PARENT_LAST_PT_SHA256={plan['parent_checkpoint']['sha256']}")
        print(f"COMPLETED_EPOCHS={plan['parent_checkpoint']['completed_epochs']}")
        print(f"NEXT_EPOCH_ONE_BASED={plan['parent_checkpoint']['next_epoch_one_based']}")
        print("TOTAL_EPOCHS=200")
        print(f"PREVIEW_MANIFEST={preview}")
        print(f"PREVIEW_SHA256={br.sha256_file(preview)}")
        print("DESTINATION_RUN_MATERIALIZED=FALSE")
        print("TEST_ACCESS=NONE")
        print("EPOCH_CALIBRATION_RESUME_PREFLIGHT=PASS")
        print("=" * 100)
        return

    resume_execute(input_root=input_root, work_root=work_root)


if __name__ == "__main__":
    main()