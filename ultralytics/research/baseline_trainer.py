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
    """Enforce the frozen INIT-01/TRAIN-01 contract in the actual DDP trainer."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        expected_initialization = os.environ.get("RESEMA_INITIALIZATION", "")
        configured_seed = int(self.args.seed)

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
        }

        if weights is None:
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
                "TRAIN-01 runtime dataset dictionary contains a forbidden test split."
            )

        preflight_path = Path(os.environ.get("RESEMA_PREFLIGHT_MANIFEST", ""))
        if not preflight_path.is_file():
            raise RuntimeError("TRAIN-01 preflight manifest is missing.")

        preflight = json.loads(
            preflight_path.read_text(encoding="utf-8", errors="strict")
        )
        experiment_id = os.environ.get("RESEMA_EXPERIMENT_ID", "")

        if preflight.get("experiment_id") != experiment_id:
            raise RuntimeError("TRAIN-01 experiment/preflight identity mismatch.")

        actual_train = len(self.train_loader.dataset)
        actual_val = len(self.test_loader.dataset)
        expected_train = int(preflight["operational_expected"]["train_images"])
        expected_val = int(preflight["operational_expected"]["validation_images"])

        if actual_train != expected_train:
            raise RuntimeError(
                f"TRAIN-01 operational train count mismatch: "
                f"{actual_train} != {expected_train}"
            )
        if actual_val != expected_val:
            raise RuntimeError(
                f"TRAIN-01 operational validation count mismatch: "
                f"{actual_val} != {expected_val}"
            )

        repo_root = Path(os.environ["RESEMA_REPO_ROOT"]).resolve()
        ultralytics_source = Path(sys.modules["ultralytics"].__file__).resolve()
        if repo_root not in ultralytics_source.parents:
            raise RuntimeError(
                "TRAIN-01 imported Ultralytics is not from the governed checkout."
            )

        governance_dir = self.save_dir / "governance"
        governance_dir.mkdir(parents=True, exist_ok=True)

        copied_preflight = governance_dir / "PRETRAIN_PREFLIGHT.json"
        shutil.copy2(preflight_path, copied_preflight)

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
            "runtime": {
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
            },
            "dataset": {
                "train_path": self.data["train"],
                "validation_path": self.data["val"],
                "operational_train_images": actual_train,
                "operational_validation_images": actual_val,
            },
            "initialization_audit": self.train01_initialization_audit,
            "effective_training_args": {
                key: getattr(self.args, key)
                for key in selected_arg_keys
            },
        }

        manifest_path = governance_dir / "RUNTIME_MANIFEST.json"
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(runtime_manifest, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        tmp.replace(manifest_path)
