# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VGRA V1 governed pair-aware trainer (D4-C5C).

No modification to stock DetectionTrainer. This class must be constructed with
explicit TRAIN/VAL frozen assignment manifest paths. It does not itself authorize
training; execution requires a separate D4-E/D4-F experiment contract.
"""

from __future__ import annotations

import hashlib

import yaml
from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.vgra import build_vgra_dataloader, build_vgra_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from .d4e_r3_verified_recovery_mixin import D4EVerifiedRecoveryTrainerMixin
from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from .vgra_fail_closed import (
    GuardedTrainerMixin, GuardedVGRADetectionModel, require_finite_model,
    pinned_model_overrides, verify_early_tensor_transfer,
)
from ultralytics.models.yolo.detect.vgra_targets import train_state_weight_binding
from .d4e_visibility_binding_guard import CLASS_ORDER, verify_binding
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .vgra_validator import VGRADetectionValidator
from .vgra_pretrained_boundary import (
    authenticate_initial_checkpoint, authenticate_arm_yaml, deserialize_initial_pretrained,
)


ASSIGNMENT_MANIFEST_SHA256 = "87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4"
VGRA_YAML_PATH = "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml"


def _sha256(path: Path) -> str:
    hashobj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hashobj.update(chunk)
    return hashobj.hexdigest()


def _check_manifest(path: str | Path, name: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen {name} assignment manifest missing: {path}")
    actual = _sha256(path)
    if actual != ASSIGNMENT_MANIFEST_SHA256:
        raise ValueError(f"{name} frozen D4-C2 assignment SHA drift: {actual}")
    return path


def _fail_closed_config(cfg: Any) -> None:
    """Protect paired-batch integrity and unsupported experimental options."""
    if bool(getattr(cfg, "compile", False)):
        raise ValueError("VGRA V1 compile=True unsupported; refuse compiled native-only forward")
    if getattr(cfg, "fraction", 1.0) != 1.0:
        raise ValueError("VGRA V1 requires fraction=1.0 for complete pair accounting")
    if float(getattr(cfg, "multi_scale", 0.0)) != 0.0:
        raise ValueError("VGRA V1 requires multi_scale=0 pending separate verification")
    if int(getattr(cfg, "batch", 16)) < 2:
        raise ValueError("VGRA V1 batch size must be >= 2")
    if bool(getattr(cfg, "single_cls", False)):
        raise ValueError("VGRA V1 nine-class visibility supervision forbids single_cls=True")


def _require_explicit_development_val(data: dict[str, Any]) -> None:
    """Never fall back from missing development VAL to the sealed TEST split."""
    if not data.get("train") or not data.get("val"):
        raise ValueError("VGRA training requires explicit B-TRAIN and B-VAL paths; TEST fallback is prohibited")
    # Resolve aliases/symlinks: string inequality alone cannot enforce a split firewall.
    train, val = Path(str(data["train"])).resolve(), Path(str(data["val"])).resolve()
    if train == val:
        raise ValueError("D4 TRAIN and VAL paths resolve to the same location")
    if data.get("test"):
        test = Path(str(data["test"])).resolve()
        if val == test or train == test:
            raise ValueError("D4 TRAIN/VAL path resolves to sealed TEST")



def _pre_native_initialization_guard(cfg: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Reject unsafe recipe/split inputs before the side-effecting BaseTrainer constructor.

    This reads only the explicitly supplied local dataset YAML (never calls native
    get_dataset/check_det_dataset). A later freeze must bind the resulting data
    YAML bytes and native-resolved split paths; this preflight alone cannot
    establish patient-effective separation or block all filesystem races.
    """
    if not isinstance(overrides, dict) or not isinstance(overrides.get("data"), (str, Path)):
        raise ValueError("D4 requires explicitly supplied local B-TRAIN/B-VAL dataset YAML in overrides")
    # Same effective config resolver as BaseTrainer, but before its side effects.
    effective = get_cfg(cfg, dict(overrides))
    _fail_closed_config(effective)
    if getattr(effective, "resume", False) is not False:
        raise ValueError("D4 initial training forbids resume before native construction")
    if getattr(effective, "task", None) != "detect":
        raise ValueError("D4 requires native detection task")
    device = getattr(effective, "device", None)
    if isinstance(device, (tuple, list)) and len(device) > 1:
        raise ValueError("D4 multi-GPU configuration forbidden before native construction")
    if isinstance(device, str) and "," in device:
        raise ValueError("D4 multi-GPU configuration forbidden before native construction")

    dataset_file = Path(str(overrides["data"])).expanduser()
    if dataset_file.suffix.lower() not in (".yaml", ".yml") or not dataset_file.is_absolute():
        raise ValueError("D4 dataset YAML must have an explicit absolute local .yaml/.yml path")
    if not dataset_file.is_file():
        raise FileNotFoundError("D4 explicit dataset YAML does not exist")
    raw = dataset_file.read_bytes()
    if not raw or len(raw) > 1024 * 1024:
        raise ValueError("D4 dataset YAML missing or unreasonably large")
    dataset = yaml.safe_load(raw)
    if not isinstance(dataset, dict):
        raise ValueError("D4 dataset YAML must decode to a mapping")
    if dataset.get("nc") not in (None, 9):
        raise ValueError("D4 dataset YAML must describe exactly nine classes")
    # Governed detection-only dataset schema: never permit Ultralytics download
    # instructions, opaque script/URL providers, or unknown native fallback keys.
    if "download" in dataset:
        raise ValueError("D4 dataset download directives are prohibited")
    supported = {"path", "train", "val", "test", "nc", "names", "channels"}
    unexpected = set(dataset) - supported
    if unexpected:
        raise ValueError(f"D4 dataset YAML has unapproved keys: {sorted(map(str, unexpected))}")
    if dataset.get("channels", 3) != 3:
        raise ValueError("D4 initial three-channel checkpoint requires channels=3")
    names = dataset.get("names")
    if isinstance(names, dict):
        if {str(k) for k in names} != {str(i) for i in range(9)} or len(names) != 9:
            raise ValueError("D4 class-name mapping must use keys 0 through 8")
        ordered_names = [names[i] if i in names else names[str(i)] for i in range(9)]
    elif isinstance(names, list) and len(names) == 9:
        ordered_names = list(names)
    else:
        raise ValueError("D4 dataset YAML must explicitly declare nine ordered class names")
    if any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in ordered_names):
        raise ValueError("D4 class names must be nonempty, trimmed strings")
    if len({x.casefold() for x in ordered_names}) != 9:
        raise ValueError("D4 nine class names must be distinct")
    if tuple(ordered_names) != CLASS_ORDER:
        raise ValueError("D4 dataset class order differs from frozen GRAZPEDWRI-DX IDs 0..8")

    root_value = dataset.get("path")
    if root_value is None:
        dataset_root = dataset_file.parent.resolve()
    elif isinstance(root_value, str) and root_value and Path(root_value).is_absolute():
        dataset_root = Path(root_value).resolve()
    else:
        raise ValueError("D4 dataset path must be absolute when specified")

    resolved = {}
    for split in ("train", "val", "test"):
        value = dataset.get(split)
        if split != "test" and (not isinstance(value, str) or not value.strip()):
            raise ValueError("D4 requires explicit B-TRAIN and B-VAL paths before native construction")
        if value is None and split == "test":
            continue
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError("D4 dataset splits must be nonempty local paths, never URLs or split lists")
        split_path = Path(value)
        resolved[split] = (split_path if split_path.is_absolute() else dataset_root / split_path).resolve()
    norm = {key: str(value).casefold() for key, value in resolved.items()}
    if norm["train"] == norm["val"]:
        raise ValueError("D4 B-TRAIN and B-VAL resolve to the same location")
    if "test" in norm and norm["test"] in (norm["train"], norm["val"]):
        raise ValueError("D4 B-TRAIN/B-VAL resolve to the sealed TEST path")
    # This is the sole dataset-YAML read. Native BaseTrainer must consume this
    # in-memory mapping via overridden get_dataset(), never check_det_dataset().
    # Deliberately omit TEST from the runtime training mapping: B-TEST stays sealed.
    native_data = {
        "path": dataset_root,
        "train": str(resolved["train"]),
        "val": str(resolved["val"]),
        "names": dict(enumerate(ordered_names)),
        "nc": 9,
        "channels": 3,
        "yaml_file": str(dataset_file.resolve()),
    }
    return {
        "yaml_path": dataset_file.resolve(),
        "yaml_sha256": hashlib.sha256(raw).hexdigest(),
        "split_paths": resolved,
        "ordered_names": tuple(ordered_names),
        "native_data": native_data,
    }


def _rebind_best_detection_loss_args(selected, checkpoint, training_args, *, arm: str) -> None:
    """Restore attribute-style loss gains after the pinned native BEST loader.

    The native load_checkpoint() overwrites model.args with a plain dictionary.
    The pinned v8DetectionLoss accesses model.args.box/cls/dfl as attributes.
    Reject lost or drifted checkpoint gains instead of silently using defaults.
    This touches only the in-memory selected BEST model, never checkpoint bytes.
    """
    from math import isfinite
    from types import SimpleNamespace

    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("train_args"), dict):
        raise TypeError(f"D4_{arm}_BEST_TRAIN_ARGS_MISSING")
    saved = checkpoint["train_args"]
    loaded_args = getattr(selected, "args", None)
    if not isinstance(loaded_args, dict):
        raise TypeError(f"D4_{arm}_BEST_LOADER_ARGS_UNEXPECTED_TYPE")

    # The saved gains must be explicit, finite, and equal to the governed
    # training recipe. Do not silently use load_checkpoint()'s defaults.
    for key in ("box", "cls", "dfl"):
        if key not in saved or key not in loaded_args or not hasattr(training_args, key):
            raise ValueError(f"D4_{arm}_BEST_LOSS_GAIN_MISSING_{key}")
        try:
            from_ckpt = float(saved[key])
            from_loaded = float(loaded_args[key])
            from_recipe = float(getattr(training_args, key))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"D4_{arm}_BEST_LOSS_GAIN_NONNUMERIC_{key}") from exc
        if not all(isfinite(v) for v in (from_ckpt, from_loaded, from_recipe)):
            raise ValueError(f"D4_{arm}_BEST_LOSS_GAIN_NONFINITE_{key}")
        if from_ckpt != from_recipe or from_loaded != from_ckpt:
            raise ValueError(f"D4_{arm}_BEST_LOSS_GAIN_DRIFT_{key}")

    # Preserve the original native checkpoint values, changing access style
    # only so the pinned loss object can use .box / .cls / .dfl.
    # R8 selected-BEST only: the cached DFL loss projection is an ordinary
    # tensor (not an nn.Module buffer), so a checkpoint may deserialize its
    # projection on CPU even though the restored model is on CUDA. Recreate
    # the disposable evaluation criterion lazily, without moving/mutating any
    # weight, changing optimizer/EMA state, or modifying checkpoint bytes.
    # This guards both MV00 and MV01; both call this same helper from final_eval.
    criterion = getattr(selected, "criterion", None)
    if criterion is not None:
        from ultralytics.utils.loss import v8DetectionLoss

        if not isinstance(criterion, v8DetectionLoss):
            raise TypeError(f"D4_{arm}_BEST_CRITERION_UNEXPECTED_TYPE")
        projection = getattr(criterion, "proj", None)
        criterion_device = getattr(criterion, "device", None)
        if not isinstance(projection, torch.Tensor) or not isinstance(criterion_device, torch.device):
            raise TypeError(f"D4_{arm}_BEST_CRITERION_DEVICE_METADATA_INVALID")
        try:
            model_device = next(selected.parameters()).device
        except StopIteration as exc:
            raise RuntimeError(f"D4_{arm}_BEST_MODEL_HAS_NO_PARAMETERS") from exc
        if criterion_device != model_device or projection.device != model_device:
            # Only the observed CPU-projection / CUDA-model serialization
            # artifact is an accepted automatic repair. Everything else fails.
            if not (
                model_device.type == "cuda"
                and projection.device.type == "cpu"
                and criterion_device in (model_device, projection.device)
            ):
                raise RuntimeError(f"D4_{arm}_BEST_CRITERION_UNEXPECTED_DEVICE_DRIFT")
            selected.criterion = None

    selected.args = SimpleNamespace(**loaded_args)


class VGRADetectionTrainer(D4EVerifiedRecoveryTrainerMixin, GuardedTrainerMixin, DetectionTrainer):
    """Specialized guarded trainer retaining native optimizer mathematical update rules."""

    MODEL_YAML_PATH = VGRA_YAML_PATH

    def __init__(
        self,
        *,
        train_assignment_manifest: str | Path,
        val_assignment_manifest: str | Path,
        train_visibility_evidence: str | Path,
        pretrained_archive: str | Path,
        pretrained_best: str | Path,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ):
        # Guard recipe and explicit B-VAL before ANY native trainer initialization,
        # checkpoint-byte reads, manifest checks, device selection or dataset access.
        planned_best = Path(pretrained_best).resolve()
        overrides = pinned_model_overrides(overrides, self.MODEL_YAML_PATH, planned_best)
        self._r5_c3_preflight = _pre_native_initialization_guard(cfg, overrides)
        self._d4e_train_visibility_evidence = Path(train_visibility_evidence).resolve(strict=True)
        self.vgra_train_manifest = _check_manifest(train_assignment_manifest, "B-TRAIN")
        self.vgra_val_manifest = _check_manifest(val_assignment_manifest, "B-VAL")
        if self.vgra_train_manifest != self.vgra_val_manifest:
            raise ValueError("VGRA V1 requires the same frozen C2 combined assignment manifest for both splits")
        # Snapshot is authenticated at the byte boundary and retained in memory.
        # Native setup_model MUST NOT reopen best.pt or invoke native fallbacks.
        self._r5_c2_checkpoint_snapshot = authenticate_initial_checkpoint(pretrained_archive, pretrained_best)
        self._r5_c2_yaml_snapshot = authenticate_arm_yaml(self.MODEL_YAML_PATH)
        self.d4_verified_best = self._r5_c2_checkpoint_snapshot.canonical_path
        self._r5_c2_approved_weights = None
        if self.d4_verified_best != planned_best:
            raise ValueError("D4 approved checkpoint path changed before native construction")
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        _fail_closed_config(self.args)
        _require_explicit_development_val(self.data)
        if self.world_size > 1:
            raise NotImplementedError("VGRA V1 distributed training is not authorized")

    def get_dataset(self):
        """Return preflight-bound TRAIN/VAL mapping, without native YAML reread.

        Native BaseTrainer.__init__ calls self.get_dataset(). This override
        deliberately avoids check_det_dataset() and its download/exec paths.
        The exact D4-E approved class-order and dataset-inventory hashes remain
        separate release gates; this alone does not authenticate image files.
        """
        preflight = getattr(self, "_r5_c3_preflight", None)
        if not isinstance(preflight, dict) or "native_data" not in preflight:
            raise RuntimeError("D4 authenticated dataset snapshot unavailable")
        if Path(str(self.args.data)).resolve() != preflight["yaml_path"]:
            raise ValueError("D4 dataset YAML path changed after guarded preflight")
        data = preflight["native_data"]
        if not isinstance(data, dict) or data.get("nc") != 9:
            raise ValueError("D4 bound dataset snapshot invalid")
        # Detach returned mutable native mapping from the saved preflight record.
        return {**data, "names": dict(data["names"])}

    def setup_model(self):
        """Guarded pretrained dispatch for BOTH MV-00 and MV-01.

        Replace only the native initial pretrained path; native training loop,
        optimizer, EMA, AMP and BEST lifecycle remain untouched/unauthorized.
        """
        if isinstance(self.model, torch.nn.Module):
            raise RuntimeError("D4 initial model may not be injected as an already-loaded module")
        if str(self.model) != self.MODEL_YAML_PATH:
            raise ValueError("D4 model architecture source differs from pinned arm")
        if Path(str(self.args.pretrained)).resolve() != self.d4_verified_best:
            raise ValueError("D4 pretrained path differs from approved snapshot metadata")
        snapshot = self._r5_c2_checkpoint_snapshot
        if snapshot is None:
            raise RuntimeError("D4 initial checkpoint snapshot already consumed")
        try:
            weights, _ = deserialize_initial_pretrained(snapshot)
            self._r5_c2_approved_weights = weights
            self.model = self.get_model(cfg=self.MODEL_YAML_PATH, weights=weights, verbose=RANK == -1)
        finally:
            self._r5_c2_approved_weights = None
            self._r5_c2_checkpoint_snapshot = None
        return None  # Matches native pretrained-YAML setup_model return policy

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        if cfg is not None and cfg != self.MODEL_YAML_PATH:
            raise ValueError("D4 MV-01 model architecture drift")
        if weights is None or weights is not self._r5_c2_approved_weights:
            raise RuntimeError("D4 MV-01 requires the exact approved in-memory Early weights")
        model = GuardedVGRADetectionModel(
            cfg=self._r5_c2_yaml_snapshot.model_config(),
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model.load(weights)
        verify_early_tensor_transfer(weights, model)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode not in ("train", "val"):
            raise ValueError("VGRA only supports TRAIN or VAL in this stage")
        if self.data["nc"] != 9:
            raise ValueError("VGRA candidate V1 requires exactly nine detection classes")
        _fail_closed_config(self.args)
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        manifest = self.vgra_train_manifest if mode == "train" else self.vgra_val_manifest
        dataset = build_vgra_yolo_dataset(
            self.args,
            img_path,
            batch if batch is not None else self.args.batch,
            self.data,
            assignment_manifest=manifest,
            mode=mode,
            stride=stride,
        )
        # Check the accepted operational counts before touching training weights.
        # B-VAL excludes the one known unreadable image without changing its freeze.
        expected_images, expected_pairs, expected_singles = (
            (14227, 6496, 1235) if mode == "train" else (3049, 1401, 247)
        )
        units = getattr(dataset, "vgra_units", None)
        if not isinstance(units, (list, tuple)):
            raise RuntimeError("D4 paired dataset lacks audited runtime units")
        actual_pairs = sum(len(unit) == 2 for unit in units)
        actual_singles = sum(len(unit) == 1 for unit in units)
        if (len(dataset.im_files), actual_pairs, actual_singles) != (
            expected_images, expected_pairs, expected_singles
        ) or actual_pairs + actual_singles != len(units):
            raise RuntimeError("D4 runtime paired dataset counts differ from C4 freeze")
        return dataset

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if rank != -1 or self.world_size > 1:
            raise NotImplementedError("VGRA V1 requires non-distributed training/validation")
        if mode not in ("train", "val"):
            raise ValueError("VGRA dataloader mode must be train or val")
        _fail_closed_config(self.args)
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)
        if mode == "train":
            # B-TRAIN only. The separate source freeze will register exact count/weight hashes.
            counts, weights = train_state_weight_binding(dataset, nc=9, split="train")
            # D4-E R4: fail closed before binding if native TRAIN-derived states
            # or weights differ from immutable independently accepted R3-11.
            verify_binding(
                self._d4e_train_visibility_evidence,
                counts.detach().cpu().tolist(),
                weights.detach().cpu().tolist(),
            )
            model = unwrap_model(self.model)
            if not isinstance(model, VGRADetectionModel):
                raise TypeError("VGRA trainer requires VGRADetectionModel")
            model.bind_train_visibility_weights(weights.to(next(model.parameters()).device), source="B-TRAIN")
            if not bool(model.vgra_train_weights_ready.item()):
                raise RuntimeError("D4 TRAIN visibility buffer not marked ready")
            if not torch.equal(model.vgra_visibility_weights.detach().cpu(), weights.detach().cpu()):
                raise RuntimeError("D4 bound model visibility weights differ from B-TRAIN")
            self.vgra_train_state_counts = counts.detach().cpu().clone()
            self.vgra_train_state_weights = weights.detach().cpu().clone()

        return build_vgra_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=(mode == "train"),
            seed=self.args.seed,
            rank=rank,
            drop_last=False,
            pin_memory=True,
        )

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "vis_loss")
        return VGRADetectionValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def final_eval(self):
        """Validate the selected BEST checkpoint with the governed pair-aware loader.

        BaseTrainer.final_eval calls validator(model=path) without a trainer,
        which cannot carry AP/LAT metadata. Preserve best-checkpoint selection,
        but explicitly provide both the trainer's paired B-VAL context and
        the loaded best model. Never fall back to native image-only validation.
        """
        from ultralytics.nn.tasks import load_checkpoint

        _require_explicit_development_val(self.data)
        best = Path(self.best)
        if not best.is_file():
            raise FileNotFoundError(f"VGRA selected-best checkpoint missing: {best}")
        selected, checkpoint = load_checkpoint(best, device=self.device)
        if not isinstance(selected, GuardedVGRADetectionModel):
            raise TypeError("D4 MV-01 selected best checkpoint lost guarded model type")
        require_finite_model(selected, "vgra_selected_best")
        _rebind_best_detection_loss_args(selected, checkpoint, self.args, arm="MV01")
        if not bool(selected.vgra_train_weights_ready.item()):
            raise RuntimeError("VGRA best checkpoint lost B-TRAIN visibility weight binding")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        metrics = self.validator(trainer=self, model=selected)
        if not isinstance(metrics, dict):
            raise TypeError("VGRA final paired validation did not return a metrics dictionary")
        self.metrics = dict(metrics)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")


__all__ = ("VGRADetectionTrainer", "_require_explicit_development_val")
