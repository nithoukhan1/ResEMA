from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAINING_YAML = ROOT / "research/05_experiments/TRAINING.yaml"
EXPERIMENTS_CSV = ROOT / "research/05_experiments/EXPERIMENTS.csv"
DATA_BINDINGS_JSON = (
    ROOT / "research/04_data/manifests/DATA01_ACTIVE_DATASET_BINDINGS.json"
)
INIT_LOCK_JSON = ROOT / "research/01_provenance/INIT01_INITIALIZATION_LOCK.json"
TRAIN_CONTRACT_JSON = (
    ROOT / "research/05_experiments/TRAIN01_TRAINER_CONTRACT.json"
)

EXPECTED_BRANCH = "research/baseline-refresh"
CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
    "periostealreaction", "pronatorsign", "softtissue", "text",
]
IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
}
SOURCE_GUARD_PATHS = [
    "ultralytics",
    "research/runtime",
    "ultralytics/research/baseline_trainer.py",
    "research/runtime/baseline_runner.py",
    "research/runtime/resume_runner.py",
    "research/05_experiments/TRAINING.yaml",
    "research/05_experiments/TRAIN01_TRAINER_CONTRACT.json",
    "research/05_experiments/RESUME01_RESUME_CONTRACT.json",
    "research/01_provenance/INIT01_INITIALIZATION_LOCK.json",
    "research/04_data/manifests/DATA01_ACTIVE_DATASET_BINDINGS.json",
    "ultralytics/engine/model.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/dist.py",
    "ultralytics/models/yolo/detect/train.py",
    "ultralytics/cfg/models/11/yolo11.yaml",
]


class GovernanceError(RuntimeError):
    pass


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and proc.returncode != 0:
        raise GovernanceError(
            f"Git command failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout}"
        )
    return proc.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(data, dict):
        raise GovernanceError(f"Expected YAML mapping: {path}")
    return data


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="strict"))


def _read_experiment_matrix_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    if not fieldnames or "experiment_id" not in fieldnames or "source_commit" not in fieldnames:
        raise GovernanceError("Experiment matrix schema is incomplete.")
    ids = [row["experiment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise GovernanceError("Experiment matrix contains duplicate experiment IDs.")
    return fieldnames, rows


def _read_current_experiment_matrix() -> tuple[list[str], list[dict[str, str]]]:
    return _read_experiment_matrix_text(
        EXPERIMENTS_CSV.read_text(encoding="utf-8-sig", errors="strict")
    )


def load_experiment(experiment_id: str) -> dict[str, str]:
    _fieldnames, rows = _read_current_experiment_matrix()

    matches = [r for r in rows if r["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise GovernanceError(
            f"Experiment ID must resolve exactly once: {experiment_id!r}"
        )

    row = matches[0]
    if row["status"] != "NOT_STARTED":
        raise GovernanceError(
            "Governed baseline execution requires NOT_STARTED until the "
            f"experiment finishes; observed {row['status']!r}"
        )
    if row["seed"] != "42":
        raise GovernanceError(f"TRAIN-01 seed drift: {row['seed']!r}")
    return row


def binding_expectation(row: dict[str, str], manifest: dict) -> dict:
    binding_id = row["data_binding"]

    if binding_id == "DATA01:B-ORG:v1":
        b = manifest["bindings"]["B_ORG"]
        return {
            "binding_id": binding_id,
            "train_count": b["runtime_counts"]["train_images"],
            "train_hash": b["membership_sha256_final_newline"]["train"],
            "validation_count": b["runtime_counts"]["validation_images"],
            "validation_hash": b["membership_sha256_final_newline"]["validation"],
            "operational_train_images": 14227,
            "operational_validation_images": 3049,
        }

    if binding_id == "DATA01:B-AUG-HIST:v1":
        b_aug = manifest["bindings"]["B_AUG"]
        b_org = manifest["bindings"]["B_ORG"]
        return {
            "binding_id": binding_id,
            "train_count": b_aug["runtime_counts"]["train_aug_historical_images"],
            "train_hash": b_aug["membership_sha256_final_newline"]["train_aug_historical"],
            "validation_count": b_aug["runtime_counts"]["validation_images"],
            "validation_hash": b_org["membership_sha256_final_newline"]["validation"],
            "operational_train_images": 28454,
            "operational_validation_images": 3049,
        }

    if binding_id == "DATA01:A-AUG:v1":
        b = manifest["bindings"]["A_AUG"]
        return {
            "binding_id": binding_id,
            "train_count": b["runtime_counts"]["train_aug_images"],
            "train_hash": b["membership_sha256_final_newline"]["runtime_train_aug_raw"],
            "validation_count": b["runtime_counts"]["validation_images"],
            "validation_hash": b["membership_sha256_final_newline"]["validation"],
            "operational_train_images": 28408,
            "operational_validation_images": 4094,
        }

    raise GovernanceError(f"Unsupported DATA-01 binding: {binding_id!r}")


def membership_hash(directory: Path, *, kind: str) -> tuple[int, str]:
    if kind == "images":
        files = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
    elif kind == "labels":
        files = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        ]
    else:
        raise ValueError(kind)

    stems = sorted(p.stem for p in files)
    if len(stems) != len(set(stems)):
        raise GovernanceError(f"Duplicate stems in {directory}")

    payload = (("\n".join(stems) + "\n") if stems else "").encode("utf-8")
    return len(stems), hashlib.sha256(payload).hexdigest()


def _path_is_test_like(path: Path) -> bool:
    return any(part.lower().startswith("test") for part in path.parts)


def discover_membership_directory(
    input_root: Path,
    *,
    kind: str,
    expected_count: int,
    expected_hash: str,
) -> Path:
    matches: list[Path] = []

    for current, dirs, _files in os.walk(input_root):
        current_path = Path(current)
        dirs[:] = [d for d in dirs if not d.lower().startswith("test")]

        if _path_is_test_like(current_path):
            continue

        try:
            count, digest = membership_hash(current_path, kind=kind)
        except OSError:
            continue

        if count == expected_count and digest == expected_hash:
            matches.append(current_path.resolve())

    unique = sorted(set(matches))
    if len(unique) != 1:
        raise GovernanceError(
            f"Expected exactly one {kind} directory for count={expected_count}, "
            f"hash={expected_hash}; observed {unique}"
        )
    return unique[0]


def verify_image_label_pair(image_dir: Path, label_dir: Path) -> None:
    image_count, image_hash = membership_hash(image_dir, kind="images")
    label_count, label_hash = membership_hash(label_dir, kind="labels")
    if image_count != label_count or image_hash != label_hash:
        raise GovernanceError(
            "Image/label membership mismatch:\n"
            f"images={image_dir} {image_count} {image_hash}\n"
            f"labels={label_dir} {label_count} {label_hash}"
        )



def verify_experiment_matrix_binding(source_commit: str) -> None:
    """Allow only the atomic source_commit binding to differ from source commit S."""
    frozen_text = git(
        "show",
        f"{source_commit}:research/05_experiments/EXPERIMENTS.csv",
    )
    frozen_fields, frozen_rows = _read_experiment_matrix_text(frozen_text)
    current_fields, current_rows = _read_current_experiment_matrix()

    if current_fields != frozen_fields:
        raise GovernanceError(
            "EXPERIMENTS.csv field order/schema changed after the frozen source commit."
        )
    if len(current_rows) != 6 or len(frozen_rows) != 6:
        raise GovernanceError(
            "The governed baseline matrix must contain exactly six experiments."
        )

    frozen_by_id = {row["experiment_id"]: row for row in frozen_rows}
    current_by_id = {row["experiment_id"]: row for row in current_rows}
    if set(current_by_id) != set(frozen_by_id):
        raise GovernanceError(
            "EXPERIMENTS.csv experiment IDs changed after the frozen source commit."
        )

    for experiment_id in sorted(frozen_by_id):
        frozen = frozen_by_id[experiment_id]
        current = current_by_id[experiment_id]

        if frozen["source_commit"].strip():
            raise GovernanceError(
                "Frozen source commit S must contain blank source_commit bindings."
            )
        if current["source_commit"].strip() != source_commit:
            raise GovernanceError(
                "Authorization binding drift: every experiment source_commit must "
                f"equal frozen source commit {source_commit}."
            )

        for field in frozen_fields:
            if field == "source_commit":
                continue
            if current.get(field) != frozen.get(field):
                raise GovernanceError(
                    "EXPERIMENTS.csv changed outside the permitted source_commit "
                    f"binding: experiment={experiment_id}, field={field}, "
                    f"frozen={frozen.get(field)!r}, current={current.get(field)!r}"
                )


def verify_git_provenance(row: dict[str, str]) -> tuple[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")

    if branch != EXPECTED_BRANCH:
        raise GovernanceError(f"Wrong branch: {branch!r}")
    if dirty:
        raise GovernanceError("Training requires a clean Git worktree.")

    source_commit = row["source_commit"].strip()
    if not source_commit:
        raise GovernanceError(
            "Training remains locked: EXPERIMENTS.csv source_commit is blank. "
            "RESUME-01 closure must bind the frozen source commit before any launch."
        )

    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", source_commit, head],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise GovernanceError(
            "Frozen training source commit is not an ancestor of the execution commit."
        )

    verify_experiment_matrix_binding(source_commit)

    diff = git(
        "diff", "--name-only", source_commit, head, "--", *SOURCE_GUARD_PATHS
    )
    if diff:
        raise GovernanceError(
            "Scientific training source changed after the frozen source commit:\n"
            f"{diff}"
        )

    return source_commit, head


def verify_runtime_environment(init_lock: dict) -> dict:
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics_config"
    )

    import torch
    import ultralytics

    expected = init_lock["exact_fork_runtime"]
    python_version = platform.python_version()
    ultralytics_version = ultralytics.__version__
    ultralytics_source = Path(ultralytics.__file__).resolve()

    if python_version != expected["python"]:
        raise GovernanceError(
            f"Python drift: {python_version} != {expected['python']}"
        )
    if torch.__version__ != expected["torch"]:
        raise GovernanceError(
            f"PyTorch drift: {torch.__version__} != {expected['torch']}"
        )
    if ultralytics_version != expected["ultralytics_version"]:
        raise GovernanceError(
            f"Ultralytics drift: {ultralytics_version} != "
            f"{expected['ultralytics_version']}"
        )
    if ROOT not in ultralytics_source.parents:
        raise GovernanceError(
            "Imported Ultralytics is not from the governed repository checkout."
        )
    if not torch.cuda.is_available():
        raise GovernanceError("TRAIN-01 requires CUDA.")

    gpu_names = [
        torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())
    ]
    if gpu_names != ["Tesla T4", "Tesla T4"]:
        raise GovernanceError(f"GPU contract drift: {gpu_names}")

    return {
        "python": python_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics_version,
        "ultralytics_source": str(ultralytics_source),
        "cuda_available": True,
        "gpu_names": gpu_names,
    }


def discover_checkpoint(input_root: Path, init_lock: dict) -> Path:
    official = init_lock["official_checkpoint"]
    expected_bytes = int(official["bytes"])
    expected_sha = official["sha256"]
    exact: list[Path] = []

    for path in input_root.rglob("yolo11s.pt"):
        if not path.is_file():
            continue
        if path.stat().st_size != expected_bytes:
            continue
        if sha256_file(path) == expected_sha:
            exact.append(path.resolve())

    exact = sorted(set(exact))
    if len(exact) != 1:
        raise GovernanceError(
            f"Expected exactly one official yolo11s.pt candidate; observed {exact}"
        )
    return exact[0]


def write_runtime_data_yaml(
    path: Path,
    *,
    train_images: Path,
    validation_images: Path,
) -> str:
    payload = {
        "train": str(train_images),
        "val": str(validation_images),
        "nc": 9,
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    if "test" in payload:
        raise GovernanceError("TRAIN-01 runtime YAML must not contain test.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return sha256_file(path)


def build_training_args(
    training: dict,
    *,
    experiment_id: str,
    runtime_data_yaml: Path,
) -> dict:
    t = training["training"]
    loss = training["loss"]
    aug = training["augmentation"]
    val = training["validation"]
    runtime = training["runtime"]
    output = training["output"]

    args = {
        "data": str(runtime_data_yaml),
        "epochs": int(t["epochs"]),
        "patience": int(t["patience"]),
        "imgsz": int(t["imgsz"]),
        "batch": int(t["batch_global"]),
        "optimizer": str(t["optimizer"]),
        "lr0": float(t["lr0"]),
        "lrf": float(t["lrf"]),
        "momentum": float(t["momentum"]),
        "weight_decay": float(t["weight_decay"]),
        "nbs": int(t["nbs"]),
        "cos_lr": bool(t["cos_lr"]),
        "warmup_epochs": float(t["warmup_epochs"]),
        "warmup_momentum": float(t["warmup_momentum"]),
        "warmup_bias_lr": float(t["warmup_bias_lr"]),
        "seed": int(t["seed"]),
        "deterministic": bool(t["deterministic"]),
        "amp": bool(t["amp"]),
        "multi_scale": float(t["multi_scale"]),
        "freeze": int(t["freeze"]),
        "cache": t["cache"],
        "single_cls": bool(t["single_cls"]),
        "rect": bool(t["rect"]),
        "fraction": float(t["fraction"]),
        "profile": bool(t["profile"]),
        "compile": t["compile"],
        "time": t["time"],
        "box": float(loss["box"]),
        "cls": float(loss["cls"]),
        "dfl": float(loss["dfl"]),
        "fl_gamma": float(loss["fl_gamma"]),
        "hsv_h": float(aug["hsv_h"]),
        "hsv_s": float(aug["hsv_s"]),
        "hsv_v": float(aug["hsv_v"]),
        "degrees": float(aug["degrees"]),
        "translate": float(aug["translate"]),
        "scale": float(aug["scale"]),
        "shear": float(aug["shear"]),
        "perspective": float(aug["perspective"]),
        "flipud": float(aug["flipud"]),
        "fliplr": float(aug["fliplr"]),
        "bgr": float(aug["bgr"]),
        "mosaic": float(aug["mosaic"]),
        "close_mosaic": int(aug["close_mosaic"]),
        "mixup": float(aug["mixup"]),
        "cutmix": float(aug["cutmix"]),
        "copy_paste": float(aug["copy_paste"]),
        "val": bool(val["enabled"]),
        "split": str(val["split"]),
        "conf": val["conf"],
        "iou": float(val["iou"]),
        "max_det": int(val["max_det"]),
        "save_json": bool(val["save_json"]),
        "plots": bool(val["plots"]),
        "device": list(runtime["device"]),
        "workers": int(runtime["workers_per_rank"]),
        "project": str(output["project_dir"]),
        "name": experiment_id,
        "exist_ok": bool(output["exist_ok"]),
        "save": bool(output["save"]),
        "save_period": int(output["save_period"]),
    }

    if args["split"] != "val":
        raise GovernanceError("TRAIN-01 validation split must be 'val'.")
    return args


def perform_preflight(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
):
    training = load_yaml(TRAINING_YAML)
    data_manifest = load_json(DATA_BINDINGS_JSON)
    init_lock = load_json(INIT_LOCK_JSON)
    train_contract = load_json(TRAIN_CONTRACT_JSON)
    row = load_experiment(experiment_id)

    if train_contract["status"] != "IMPLEMENTED_RESUME_AWARE":
        raise GovernanceError("TRAIN-01 contract status drift.")
    if training["status"] != "FROZEN_BASELINE_RECIPE_V2":
        raise GovernanceError("TRAINING.yaml status drift.")

    source_commit, execution_commit = verify_git_provenance(row)
    runtime = verify_runtime_environment(init_lock)
    expectation = binding_expectation(row, data_manifest)

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

    preflight_dir = work_root / "ResEMA_preflight" / experiment_id
    preflight_dir.mkdir(parents=True, exist_ok=True)

    runtime_yaml = preflight_dir / "runtime_data_train_val_only.yaml"
    runtime_yaml_sha = write_runtime_data_yaml(
        runtime_yaml,
        train_images=train_images,
        validation_images=validation_images,
    )

    checkpoint = None
    initialization = row["initialization"]

    if initialization == "pretrained":
        checkpoint = discover_checkpoint(input_root, init_lock)
        if (
            row["init_checkpoint_sha256"]
            != init_lock["official_checkpoint"]["sha256"]
        ):
            raise GovernanceError(
                "Pretrained experiment checkpoint SHA does not match INIT-01."
            )
    elif initialization == "scratch":
        if row["init_checkpoint_sha256"]:
            raise GovernanceError(
                "Scratch experiment unexpectedly binds a checkpoint SHA."
            )
    else:
        raise GovernanceError(f"Unknown initialization: {initialization!r}")

    run_dir = Path(training["output"]["project_dir"]) / experiment_id
    if run_dir.exists():
        raise GovernanceError(
            f"Fresh TRAIN-01 launch refuses existing run directory: {run_dir}"
        )

    preflight = {
        "schema_version": "TRAIN01-preflight-v1.0",
        "experiment_id": experiment_id,
        "training_source_commit": source_commit,
        "execution_commit": execution_commit,
        "data_binding": row["data_binding"],
        "initialization": initialization,
        "test_access": {
            "runtime_yaml_contains_test": False,
            "test_directory_scan_pruned": True,
            "test_predictions": False,
            "test_metrics": False,
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
        "operational_expected": {
            "train_images": expectation["operational_train_images"],
            "validation_images": expectation["operational_validation_images"],
        },
        "runtime_data_yaml": {
            "path": str(runtime_yaml),
            "sha256": runtime_yaml_sha,
            "contains_test_key": False,
        },
        "initial_checkpoint": (
            {
                "path": str(checkpoint),
                "sha256": init_lock["official_checkpoint"]["sha256"],
                "bytes": init_lock["official_checkpoint"]["bytes"],
            }
            if checkpoint is not None
            else None
        ),
        "frozen_inputs": {
            "TRAINING_yaml_sha256": sha256_file(TRAINING_YAML),
            "TRAIN01_contract_sha256": sha256_file(TRAIN_CONTRACT_JSON),
            "INIT01_lock_sha256": sha256_file(INIT_LOCK_JSON),
            "DATA01_bindings_sha256": sha256_file(DATA_BINDINGS_JSON),
        },
        "runtime": runtime,
        "output": {
            "run_dir": str(run_dir),
            "run_dir_exists_before_launch": False,
        },
    }

    preflight_path = preflight_dir / "PRETRAIN_PREFLIGHT.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return preflight, preflight_path, checkpoint


def execute(
    experiment_id: str,
    *,
    input_root: Path,
    work_root: Path,
) -> None:
    preflight, preflight_path, checkpoint = perform_preflight(
        experiment_id,
        input_root=input_root,
        work_root=work_root,
    )
    training = load_yaml(TRAINING_YAML)

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", "/kaggle/working/.ultralytics_config"
    )
    os.environ["RESEMA_EXPERIMENT_ID"] = experiment_id
    os.environ["RESEMA_INITIALIZATION"] = preflight["initialization"]
    os.environ["RESEMA_PREFLIGHT_MANIFEST"] = str(preflight_path)
    os.environ["RESEMA_PREFLIGHT_SHA256"] = sha256_file(preflight_path)
    os.environ["RESEMA_REPO_ROOT"] = str(ROOT.resolve())

    from ultralytics import YOLO
    from ultralytics.research import GovernedDetectionTrainer

    if checkpoint is not None:
        model_source = str(checkpoint)
    else:
        model_source = str(
            ROOT / "ultralytics/cfg/models/11/yolo11s.yaml"
        )

    model = YOLO(model_source, task="detect")
    args = build_training_args(
        training,
        experiment_id=experiment_id,
        runtime_data_yaml=Path(preflight["runtime_data_yaml"]["path"]),
    )
    model.train(trainer=GovernedDetectionTrainer, **args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Governed fresh baseline launcher. "
            "No scientific hyperparameter overrides are accepted."
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

    if args.mode == "preflight-only":
        preflight, preflight_path, _checkpoint = perform_preflight(
            args.experiment_id,
            input_root=Path(args.input_root),
            work_root=Path(args.work_root),
        )
        print("=" * 100)
        print("TRAIN-01 GOVERNED PREFLIGHT")
        print("=" * 100)
        print(f"EXPERIMENT_ID={preflight['experiment_id']}")
        print(f"TRAINING_SOURCE_COMMIT={preflight['training_source_commit']}")
        print(f"EXECUTION_COMMIT={preflight['execution_commit']}")
        print(f"DATA_BINDING={preflight['data_binding']}")
        print(f"INITIALIZATION={preflight['initialization']}")
        print(f"PREFLIGHT_MANIFEST={preflight_path}")
        print(f"PREFLIGHT_SHA256={sha256_file(preflight_path)}")
        print("TEST_ACCESS=NONE")
        print("TRAIN01_PREFLIGHT=PASS")
        print("=" * 100)
        return

    execute(
        args.experiment_id,
        input_root=Path(args.input_root),
        work_root=Path(args.work_root),
    )


if __name__ == "__main__":
    main()
