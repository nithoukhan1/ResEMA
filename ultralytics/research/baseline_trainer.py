from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

from ultralytics import __version__
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK


EXPECTED_TARGET_PARAMETERS = 9_431_275
EXPECTED_TARGET_STATE_ITEMS = 499
EXPECTED_TRANSFERABLE_STATE_ITEMS = 493
EXPECTED_NONTRANSFERABLE_KEYS = [
    "model.23.cv3.0.2.weight",
    "model.23.cv3.0.2.bias",
    "model.23.cv3.1.2.weight",
    "model.23.cv3.1.2.bias",
    "model.23.cv3.2.2.weight",
    "model.23.cv3.2.2.bias",
]


def architecture_fingerprint(state_dict: dict) -> str:
    rows = [
        f"{key}|{tuple(tensor.shape)}|{str(tensor.dtype)}"
        for key, tensor in state_dict.items()
    ]
    return hashlib.sha256(
        ("\n".join(rows) + "\n").encode("utf-8")
    ).hexdigest()


@contextmanager
def isolated_model_init_seed(seed: int):
    """Seed model construction only, then restore the outer DDP RNG states."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _state_clone(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _normalize_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return [str(x) for x in names]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class GovernedDetectionTrainer(DetectionTrainer):
    """Enforce frozen INIT-01/TRAIN-01/RESUME-01 contracts in the actual DDP trainer."""

    def _verify_preflight_before_dataset_access(self) -> dict:
        """Fail closed before BaseTrainer opens any train/validation dataset."""
        is_resume = bool(getattr(self, "resume", False))
        preflight_env = (
            "RESEMA_RESUME_PREFLIGHT_MANIFEST"
            if is_resume
            else "RESEMA_PREFLIGHT_MANIFEST"
        )
        preflight_sha_env = (
            "RESEMA_RESUME_PREFLIGHT_SHA256"
            if is_resume
            else "RESEMA_PREFLIGHT_SHA256"
        )

        preflight_path = Path(os.environ.get(preflight_env, ""))
        if not preflight_path.is_file():
            raise RuntimeError(
                f"{'RESUME-01' if is_resume else 'TRAIN-01'} preflight manifest "
                "is missing before dataset access."
            )
        expected_preflight_sha = os.environ.get(preflight_sha_env, "")
        if not expected_preflight_sha:
            raise RuntimeError(
                "Governed preflight SHA256 environment binding is missing before "
                "dataset access."
            )
        if _sha256_file(preflight_path) != expected_preflight_sha:
            raise RuntimeError(
                "Governed preflight manifest changed before dataset access."
            )

        preflight = json.loads(
            preflight_path.read_text(encoding="utf-8", errors="strict")
        )
        experiment_id = os.environ.get("RESEMA_EXPERIMENT_ID", "")
        if preflight.get("experiment_id") != experiment_id:
            raise RuntimeError(
                "Governed experiment/preflight identity mismatch before dataset access."
            )

        test_access = preflight.get("test_access", {})
        if test_access.get("runtime_yaml_contains_test") is not False:
            raise RuntimeError("Governed preflight does not certify a test-free runtime YAML.")
        if test_access.get("test_predictions") is not False:
            raise RuntimeError("Governed preflight test-prediction firewall drift.")
        if test_access.get("test_metrics") is not False:
            raise RuntimeError("Governed preflight test-metric firewall drift.")

        runtime_yaml_meta = preflight.get("runtime_data_yaml", {})
        runtime_yaml_path = Path(str(runtime_yaml_meta.get("path", ""))).resolve()
        if not runtime_yaml_path.is_file():
            raise RuntimeError(
                "Governed runtime data YAML is missing before dataset access."
            )
        if _sha256_file(runtime_yaml_path) != runtime_yaml_meta.get("sha256"):
            raise RuntimeError(
                "Governed runtime data YAML hash mismatch before dataset access."
            )
        if runtime_yaml_meta.get("contains_test_key") is not False:
            raise RuntimeError("Governed runtime YAML metadata indicates a test key.")
        if Path(str(self.args.data)).resolve() != runtime_yaml_path:
            raise RuntimeError(
                "Governed trainer data path differs from preflight before dataset access."
            )

        runtime_yaml = yaml.safe_load(
            runtime_yaml_path.read_text(encoding="utf-8", errors="strict")
        )
        if not isinstance(runtime_yaml, dict):
            raise RuntimeError("Governed runtime data YAML is not a mapping.")
        if "test" in runtime_yaml:
            raise RuntimeError(
                "Governed runtime data YAML contains forbidden test key before dataset access."
            )
        if "train" not in runtime_yaml or "val" not in runtime_yaml:
            raise RuntimeError("Governed runtime data YAML must contain train and val.")

        expected_train_path = Path(
            str(preflight["membership"]["train_images"]["path"])
        ).resolve()
        expected_val_path = Path(
            str(preflight["membership"]["validation_images"]["path"])
        ).resolve()
        if Path(str(runtime_yaml["train"])).resolve() != expected_train_path:
            raise RuntimeError(
                "Governed runtime train path differs from preflight before dataset access."
            )
        if Path(str(runtime_yaml["val"])).resolve() != expected_val_path:
            raise RuntimeError(
                "Governed runtime validation path differs from preflight before dataset access."
            )

        self.governed_preflight_before_dataset_access = preflight
        return preflight

    def get_dataset(self):
        """Verify the dataset firewall before delegating to Ultralytics dataset loading."""
        self._verify_preflight_before_dataset_access()
        data = super().get_dataset()
        if "test" in data:
            raise RuntimeError(
                "Governed dataset dictionary contains a forbidden test split."
            )
        return data

    def get_model(self, cfg=None, weights=None, verbose=True):
        expected_initialization = os.environ.get("RESEMA_INITIALIZATION", "")
        configured_seed = int(self.args.seed)
        is_resume = bool(getattr(self, "resume", False))

        with isolated_model_init_seed(configured_seed):
            model = DetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
            initial_state = _state_clone(model)

        expected_names = [
            "boneanomaly",
            "bonelesion",
            "foreignbody",
            "fracture",
            "metal",
            "periostealreaction",
            "pronatorsign",
            "softtissue",
            "text",
        ]

        if int(model.nc) != 9:
            raise RuntimeError(f"TRAIN-01 expected nc=9, observed nc={model.nc}")

        if _normalize_names(self.data["names"]) != expected_names:
            raise RuntimeError("TRAIN-01 class-name/order contract mismatch.")

        parameter_count = sum(int(p.numel()) for p in model.parameters())
        if parameter_count != EXPECTED_TARGET_PARAMETERS:
            raise RuntimeError(
                "TRAIN-01 target parameter-count mismatch: "
                f"{parameter_count} != {EXPECTED_TARGET_PARAMETERS}"
            )

        if len(initial_state) != EXPECTED_TARGET_STATE_ITEMS:
            raise RuntimeError(
                "TRAIN-01 target state-item mismatch: "
                f"{len(initial_state)} != {EXPECTED_TARGET_STATE_ITEMS}"
            )

        audit = {
            "configured_seed": configured_seed,
            "model_construction_seed": configured_seed,
            "outer_ultralytics_rank_seed": configured_seed + 1 + RANK,
            "outer_rank_rng_restored_after_model_construction": True,
            "target_parameters": parameter_count,
            "target_state_items": len(initial_state),
            "target_architecture_fingerprint_sha256": architecture_fingerprint(initial_state),
            "expected_initialization": expected_initialization,
            "resume": is_resume,
        }

        if is_resume:
            if weights is None:
                raise RuntimeError(
                    "RESUME-01 expected last.pt model weights but trainer received weights=None."
                )
            if expected_initialization not in {"pretrained", "scratch"}:
                raise RuntimeError(
                    "RESUME-01 original initialization identity is missing or invalid."
                )

            source_state = {
                key: value.detach().cpu()
                for key, value in weights.state_dict().items()
            }
            source_keys = list(source_state)
            target_keys = list(initial_state)
            if source_keys != target_keys:
                missing = [key for key in target_keys if key not in source_state]
                extra = [key for key in source_keys if key not in initial_state]
                raise RuntimeError(
                    "RESUME-01 checkpoint state-key contract mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            shape_mismatches = [
                key
                for key in target_keys
                if source_state[key].shape != initial_state[key].shape
            ]
            if shape_mismatches:
                raise RuntimeError(
                    "RESUME-01 checkpoint tensor-shape mismatch: "
                    f"{shape_mismatches}"
                )

            if len(source_state) != EXPECTED_TARGET_STATE_ITEMS:
                raise RuntimeError(
                    "RESUME-01 checkpoint state-item mismatch: "
                    f"{len(source_state)} != {EXPECTED_TARGET_STATE_ITEMS}"
                )

            source_parameter_count = sum(int(p.numel()) for p in weights.parameters())
            if source_parameter_count != EXPECTED_TARGET_PARAMETERS:
                raise RuntimeError(
                    "RESUME-01 checkpoint parameter-count mismatch: "
                    f"{source_parameter_count} != {EXPECTED_TARGET_PARAMETERS}"
                )

            model.load_state_dict(weights.state_dict(), strict=True)
            loaded_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            if not all(
                torch.equal(loaded_state[key], source_state[key])
                for key in target_keys
            ):
                raise RuntimeError(
                    "RESUME-01 trained checkpoint tensors were not restored exactly."
                )

            audit.update(
                {
                    "initialization": "resume",
                    "original_initialization": expected_initialization,
                    "resume_checkpoint_loaded": True,
                    "resume_checkpoint_state_items": len(source_state),
                    "resume_checkpoint_parameter_count": source_parameter_count,
                    "resume_checkpoint_all_tensors_loaded_exactly": True,
                    "fresh_init_transfer_contract_reapplied": False,
                }
            )

        elif weights is None:
            if expected_initialization != "scratch":
                raise RuntimeError(
                    "TRAIN-01 expected pretrained weights but trainer received weights=None."
                )
            audit.update(
                {
                    "initialization": "scratch",
                    "official_checkpoint_loaded": False,
                    "transferable_state_items": 0,
                    "nontransferable_target_state_items": 0,
                }
            )
        else:
            if expected_initialization != "pretrained":
                raise RuntimeError(
                    "TRAIN-01 scratch experiment unexpectedly received pretrained weights."
                )

            source_state = {
                key: value.detach().cpu()
                for key, value in weights.state_dict().items()
            }

            transferable_keys = [
                key
                for key, tensor in source_state.items()
                if key in initial_state and tensor.shape == initial_state[key].shape
            ]
            nontransferable_target_keys = [
                key
                for key, tensor in initial_state.items()
                if key not in source_state or source_state[key].shape != tensor.shape
            ]

            if len(transferable_keys) != EXPECTED_TRANSFERABLE_STATE_ITEMS:
                raise RuntimeError(
                    "TRAIN-01 transferable-state count mismatch: "
                    f"{len(transferable_keys)} != {EXPECTED_TRANSFERABLE_STATE_ITEMS}"
                )

            if nontransferable_target_keys != EXPECTED_NONTRANSFERABLE_KEYS:
                raise RuntimeError(
                    "TRAIN-01 nontransferable-key contract mismatch: "
                    f"{nontransferable_target_keys}"
                )

            model.load(weights, verbose=False)
            loaded_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

            transferred_exact = all(
                torch.equal(loaded_state[key], source_state[key])
                for key in transferable_keys
            )
            unmatched_preserved = all(
                torch.equal(loaded_state[key], initial_state[key])
                for key in nontransferable_target_keys
            )

            if not transferred_exact:
                raise RuntimeError("TRAIN-01 pretrained transfer was not exact.")
            if not unmatched_preserved:
                raise RuntimeError(
                    "TRAIN-01 unmatched target tensors did not preserve "
                    "the isolated seeded initialization."
                )

            audit.update(
                {
                    "initialization": "pretrained",
                    "official_checkpoint_loaded": True,
                    "transferable_state_items": len(transferable_keys),
                    "nontransferable_target_state_items": len(nontransferable_target_keys),
                    "nontransferable_target_keys": nontransferable_target_keys,
                    "transferred_tensors_exact": True,
                    "unmatched_tensors_preserved_from_seeded_initialization": True,
                }
            )

        self.train01_initialization_audit = audit
        return model

    def _setup_train(self):
        super()._setup_train()

        if RANK not in {-1, 0}:
            return

        if "test" in self.data:
            raise RuntimeError(
                "Governed runtime dataset dictionary contains a forbidden test split."
            )

        is_resume = bool(getattr(self, "resume", False))
        preflight_env = (
            "RESEMA_RESUME_PREFLIGHT_MANIFEST"
            if is_resume
            else "RESEMA_PREFLIGHT_MANIFEST"
        )
        preflight_path = Path(os.environ.get(preflight_env, ""))
        if not preflight_path.is_file():
            raise RuntimeError(
                f"{'RESUME-01' if is_resume else 'TRAIN-01'} preflight manifest is missing."
            )

        expected_preflight_sha_env = (
            "RESEMA_RESUME_PREFLIGHT_SHA256"
            if is_resume
            else "RESEMA_PREFLIGHT_SHA256"
        )
        expected_preflight_sha = os.environ.get(expected_preflight_sha_env, "")
        if not expected_preflight_sha:
            raise RuntimeError("Governed preflight SHA256 environment binding is missing.")
        if _sha256_file(preflight_path) != expected_preflight_sha:
            raise RuntimeError("Governed preflight manifest changed before trainer setup.")

        preflight = json.loads(
            preflight_path.read_text(encoding="utf-8", errors="strict")
        )
        experiment_id = os.environ.get("RESEMA_EXPERIMENT_ID", "")

        if preflight.get("experiment_id") != experiment_id:
            raise RuntimeError("Governed experiment/preflight identity mismatch.")

        runtime_yaml_meta = preflight.get("runtime_data_yaml", {})
        runtime_yaml_path = Path(str(runtime_yaml_meta.get("path", ""))).resolve()
        if not runtime_yaml_path.is_file():
            raise RuntimeError("Governed runtime data YAML is missing.")
        if _sha256_file(runtime_yaml_path) != runtime_yaml_meta.get("sha256"):
            raise RuntimeError("Governed runtime data YAML hash mismatch.")
        if Path(str(self.args.data)).resolve() != runtime_yaml_path:
            raise RuntimeError("Governed trainer data path differs from the preflight YAML.")

        expected_train_path = Path(
            str(preflight["membership"]["train_images"]["path"])
        ).resolve()
        expected_val_path = Path(
            str(preflight["membership"]["validation_images"]["path"])
        ).resolve()
        if Path(str(self.data["train"])).resolve() != expected_train_path:
            raise RuntimeError("Governed train dataset path differs from preflight.")
        if Path(str(self.data["val"])).resolve() != expected_val_path:
            raise RuntimeError("Governed validation dataset path differs from preflight.")

        actual_train = len(self.train_loader.dataset)
        actual_val = len(self.test_loader.dataset)
        expected_train = int(preflight["operational_expected"]["train_images"])
        expected_val = int(preflight["operational_expected"]["validation_images"])

        if actual_train != expected_train:
            raise RuntimeError(
                "Governed operational train count mismatch: "
                f"{actual_train} != {expected_train}"
            )
        if actual_val != expected_val:
            raise RuntimeError(
                "Governed operational validation count mismatch: "
                f"{actual_val} != {expected_val}"
            )

        repo_root = Path(os.environ["RESEMA_REPO_ROOT"]).resolve()
        ultralytics_source = Path(sys.modules["ultralytics"].__file__).resolve()
        if repo_root not in ultralytics_source.parents:
            raise RuntimeError(
                "Governed run imported Ultralytics outside the governed checkout."
            )

        governance_dir = self.save_dir / "governance"
        governance_dir.mkdir(parents=True, exist_ok=True)

        selected_arg_keys = [
            "epochs", "patience", "imgsz", "batch", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "nbs", "cos_lr", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "seed", "deterministic", "amp",
            "multi_scale", "freeze", "cache", "single_cls", "rect", "fraction",
            "profile", "compile", "box", "cls", "dfl", "fl_gamma", "hsv_h",
            "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
            "perspective", "flipud", "fliplr", "bgr", "mosaic", "close_mosaic",
            "mixup", "cutmix", "copy_paste", "val", "split", "iou", "max_det",
            "save_json", "plots", "workers", "device", "project", "name",
            "exist_ok", "save", "save_period",
        ]

        runtime_block = {
            "python": platform.python_version(),
            "ultralytics": __version__,
            "ultralytics_source": str(ultralytics_source),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
            "rank": RANK,
        }
        dataset_block = {
            "train_path": self.data["train"],
            "validation_path": self.data["val"],
            "operational_train_images": actual_train,
            "operational_validation_images": actual_val,
        }
        effective_args = {
            key: getattr(self.args, key)
            for key in selected_arg_keys
        }

        if is_resume:
            session_index = int(preflight["resume_session_index"])
            sessions_dir = governance_dir / "resume_sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            expected_preflight = (
                sessions_dir / f"RESUME_PREFLIGHT_{session_index:03d}.json"
            ).resolve()
            if preflight_path.resolve() != expected_preflight:
                raise RuntimeError(
                    "RESUME-01 preflight path is outside the governed session slot."
                )

            original_manifest = governance_dir / "RUNTIME_MANIFEST.json"
            if not original_manifest.is_file():
                raise RuntimeError(
                    "RESUME-01 original TRAIN-01 runtime manifest is missing."
                )
            if _sha256_file(original_manifest) != preflight[
                "original_runtime_manifest_sha256"
            ]:
                raise RuntimeError(
                    "RESUME-01 original runtime manifest hash mismatch."
                )

            resume_checkpoint = Path(str(self.args.resume)).resolve()
            if not resume_checkpoint.is_file():
                raise RuntimeError("RESUME-01 active last.pt is missing.")
            if _sha256_file(resume_checkpoint) != preflight["parent_checkpoint"]["sha256"]:
                raise RuntimeError("RESUME-01 parent last.pt hash changed after preflight.")

            expected_start_epoch = int(
                preflight["parent_checkpoint"]["epoch_zero_based"]
            ) + 1
            if int(self.start_epoch) != expected_start_epoch:
                raise RuntimeError(
                    "RESUME-01 restored start epoch mismatch: "
                    f"{self.start_epoch} != {expected_start_epoch}"
                )
            if int(self.epochs) != int(preflight["total_epochs"]):
                raise RuntimeError(
                    "RESUME-01 epoch-budget drift: "
                    f"{self.epochs} != {preflight['total_epochs']}"
                )

            archived_args = Path(
                preflight["archived_args_before_resume"]["path"]
            ).resolve()
            if not archived_args.is_file():
                raise RuntimeError("RESUME-01 archived pre-resume args.yaml is missing.")
            if _sha256_file(archived_args) != preflight[
                "archived_args_before_resume"
            ]["sha256"]:
                raise RuntimeError("RESUME-01 archived args.yaml hash mismatch.")

            runtime_manifest = {
                "schema_version": "RESUME01-runtime-session-v1.0",
                "experiment_id": experiment_id,
                "resume_session_index": session_index,
                "training_source_commit": preflight["training_source_commit"],
                "execution_commit": preflight["execution_commit"],
                "data_binding": preflight["data_binding"],
                "original_initialization": preflight["initialization"],
                "test_split_present": False,
                "resume_preflight_manifest": {
                    "path": str(preflight_path),
                    "sha256": _sha256_file(preflight_path),
                },
                "original_runtime_manifest": {
                    "path": str(original_manifest),
                    "sha256": _sha256_file(original_manifest),
                    "preserved_unmodified": True,
                },
                "parent_checkpoint": preflight["parent_checkpoint"],
                "restored_start_epoch_zero_based": int(self.start_epoch),
                "next_epoch_one_based": int(self.start_epoch) + 1,
                "total_epochs": int(self.epochs),
                "runtime": runtime_block,
                "dataset": dataset_block,
                "resume_model_audit": self.train01_initialization_audit,
                "effective_training_args": effective_args,
            }

            manifest_path = (
                sessions_dir / f"RESUME_RUNTIME_{session_index:03d}.json"
            )
            if manifest_path.exists():
                raise RuntimeError(
                    "RESUME-01 refuses to overwrite an existing session runtime manifest."
                )
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(runtime_manifest, indent=2, default=str) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            tmp.replace(manifest_path)
            return

        copied_preflight = governance_dir / "PRETRAIN_PREFLIGHT.json"
        if copied_preflight.exists():
            raise RuntimeError(
                "TRAIN-01 refuses to overwrite an existing PRETRAIN_PREFLIGHT.json."
            )
        shutil.copy2(preflight_path, copied_preflight)

        runtime_manifest = {
            "schema_version": "TRAIN01-runtime-manifest-v1.0",
            "experiment_id": experiment_id,
            "training_source_commit": preflight["training_source_commit"],
            "execution_commit": preflight["execution_commit"],
            "data_binding": preflight["data_binding"],
            "initialization": preflight["initialization"],
            "test_split_present": False,
            "preflight_manifest": {
                "path": str(copied_preflight),
                "sha256": _sha256_file(copied_preflight),
            },
            "runtime": runtime_block,
            "dataset": dataset_block,
            "initialization_audit": self.train01_initialization_audit,
            "effective_training_args": effective_args,
        }

        manifest_path = governance_dir / "RUNTIME_MANIFEST.json"
        if manifest_path.exists():
            raise RuntimeError(
                "TRAIN-01 refuses to overwrite an existing RUNTIME_MANIFEST.json."
            )
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(runtime_manifest, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        tmp.replace(manifest_path)
