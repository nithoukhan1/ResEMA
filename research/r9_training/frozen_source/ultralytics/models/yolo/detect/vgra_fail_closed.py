# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D4-E2 R2 candidate: shared, fail-closed numerical and pretrained startup guards.

REVIEW-ONLY: not an authorization to train or read real datasets. This module does
not alter native loss equations, update rules, AMP policy, BN, or gradient clipping.
"""
from __future__ import annotations

import hashlib
import math
import zipfile
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.vgra_runtime import VGRADetectionModel
from ultralytics.utils.torch_utils import unwrap_model

EARLY_ARCHIVE_SHA256 = "4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396"
EARLY_BEST_SHA256 = "620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1"
EARLY_ZIP_MEMBER = "ResEMA_single_module_runs/BORG-PT-S42-SCCONV-EARLY-E100/weights/best.pt"
EARLY_STATE_COUNT = 537


class D4NumericalAbort(FloatingPointError):
    """Terminal fail-closed condition: caller must not retry or auto-recover."""


def _abort(stage: str, tensor: str = "", shape: Any = "") -> None:
    # Only source-safe semantic names and shapes: no filenames, patient IDs or raw data.
    raise D4NumericalAbort(f"D4_B1_ABORT stage={stage} tensor={tensor} shape={shape}")


def require_finite_tensor(value: torch.Tensor, stage: str, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{stage}: expected tensor for {name}")
    if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value.detach()).all().item()):
        _abort(stage, name, tuple(value.shape))


def require_finite_loss(loss: torch.Tensor, items: torch.Tensor, *, n_components: int) -> None:
    if not isinstance(loss, torch.Tensor) or loss.ndim != 1 or loss.shape[0] != n_components:
        raise RuntimeError("D4 loss component-vector shape differs from governed architecture")
    if not isinstance(items, torch.Tensor) or items.shape != loss.shape:
        raise RuntimeError("D4 display loss shape mismatch")
    require_finite_tensor(loss, "pre_backward_components", "loss_vector")
    require_finite_tensor(items, "pre_backward_display", "loss_items")
    require_finite_tensor(loss.sum(), "pre_backward_total", "loss_sum")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pretrained_source(archive: str | Path, best: str | Path) -> Path:
    """Authenticate both ZIP and selected independent .pt before trainer/dataset creation."""
    archive, best = Path(archive).resolve(), Path(best).resolve()
    if not archive.is_file() or not best.is_file():
        raise FileNotFoundError("D4 canonical pretrained archive or checkpoint absent")
    if _file_sha256(archive) != EARLY_ARCHIVE_SHA256:
        raise ValueError("D4 canonical pretrained ZIP SHA256 mismatch")
    if _file_sha256(best) != EARLY_BEST_SHA256:
        raise ValueError("D4 canonical pretrained best.pt SHA256 mismatch")
    with zipfile.ZipFile(archive) as z:
        if z.namelist().count(EARLY_ZIP_MEMBER) != 1:
            raise ValueError("D4 canonical pretrained ZIP member absent or ambiguous")
        member_hash = hashlib.sha256()
        with z.open(EARLY_ZIP_MEMBER) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                member_hash.update(chunk)
        if member_hash.hexdigest() != EARLY_BEST_SHA256:
            raise ValueError("D4 canonical ZIP member SHA256 mismatch")
    return best


def pinned_model_overrides(overrides: dict | None, model_yaml: str, best: Path) -> dict:
    """Refuse resume, alternate model, and alternate source checkpoint."""
    cfg = dict(overrides or {})
    if cfg.get("resume"):
        raise ValueError("D4 initial paired screen forbids resume")
    if "model" in cfg and cfg["model"] != model_yaml:
        raise ValueError("D4 model YAML differs from pinned arm")
    if "pretrained" in cfg and Path(str(cfg["pretrained"])).resolve() != best:
        raise ValueError("D4 source pretrained path differs from pinned best.pt")
    cfg["model"] = model_yaml
    cfg["pretrained"] = str(best)
    return cfg


def verify_early_tensor_transfer(source: Any, target: torch.nn.Module) -> None:
    """Fail if any of the 537 native Early tensors were omitted or altered."""
    source_model = source.get("model") if isinstance(source, dict) else source
    if not isinstance(source_model, torch.nn.Module):
        raise TypeError("D4 transfer requires a verified loaded pretrained Early model")
    canonical = source_model.float().state_dict()
    if len(canonical) != EARLY_STATE_COUNT:
        raise RuntimeError(f"D4 source state count differs from {EARLY_STATE_COUNT}")
    actual = target.state_dict()
    for key, value in canonical.items():
        candidate = actual.get(key)
        if candidate is None or candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise RuntimeError(f"D4 missing/incompatible pretrained tensor: {key}")
        if not torch.equal(candidate.detach().cpu(), value.detach().cpu()):
            raise RuntimeError(f"D4 source-to-target transfer differs: {key}")
    if isinstance(target, VGRADetectionModel):
        head = target.model[-1]
        for module in head.vgra.modules():
            rho = getattr(module, "rho", None)
            if isinstance(rho, torch.Tensor) and bool(torch.any(rho.detach() != 0).item()):
                raise RuntimeError("D4 VGRA residual rho must begin at exact zero")


def require_finite_buffers(model: torch.nn.Module, stage: str) -> None:
    for name, buffer in model.named_buffers():
        require_finite_tensor(buffer, stage, f"buffer.{name}")


class GuardedEarlyDetectionModel(DetectionModel):
    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        require_finite_loss(loss, items, n_components=3)
        require_finite_buffers(self, "pre_backward_buffers")
        return loss, items


class GuardedVGRADetectionModel(VGRADetectionModel):
    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        require_finite_loss(loss, items, n_components=4)
        require_finite_buffers(self, "pre_backward_buffers")
        return loss, items


def _state_tensor_scan(state: Any, stage: str, name: str = "") -> None:
    if isinstance(state, torch.Tensor):
        require_finite_tensor(state, stage, name)
    elif isinstance(state, dict):
        for key, item in state.items():
            _state_tensor_scan(item, stage, f"{name}.{str(key)[:64]}")
    elif isinstance(state, (list, tuple)):
        for index, item in enumerate(state):
            _state_tensor_scan(item, stage, f"{name}.{index}")
    elif isinstance(state, float) and not math.isfinite(state):
        _abort(stage, name)


def require_finite_model(model: torch.nn.Module, stage: str) -> None:
    for name, param in model.named_parameters():
        require_finite_tensor(param, stage, f"parameter.{name}")
    for name, buffer in model.named_buffers():
        require_finite_tensor(buffer, stage, f"buffer.{name}")


def require_finite_gradients(model: torch.nn.Module, stage: str) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None:
            require_finite_tensor(param.grad, stage, f"gradient.{name}")


def register_fail_closed_gradient_hooks(model: torch.nn.Module) -> None:
    """Check incoming and post-accumulated parameter gradients on every backward."""
    if getattr(model, "_d4_gradient_hooks_ready", False):
        return
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            def check(grad: torch.Tensor, *, param_name: str = name):
                require_finite_tensor(grad, "backward_gradient_hook", f"gradient.{param_name}")
                return grad
            hooks.append(param.register_hook(check))
            if not hasattr(param, "register_post_accumulate_grad_hook"):
                raise RuntimeError("D4 requires post-accumulation gradient hook support")

            def check_accumulated(leaf: torch.Tensor, *, param_name: str = name):
                # Executed after AccumulateGrad writes leaf.grad, including each microbatch.
                if leaf.grad is None:
                    _abort("backward_post_accumulation_missing", f"gradient.{param_name}")
                require_finite_tensor(
                    leaf.grad, "backward_post_accumulation", f"gradient.{param_name}"
                )

            hooks.append(param.register_post_accumulate_grad_hook(check_accumulated))
    if not hooks:
        raise RuntimeError("D4 guard found no trainable parameters")
    model._d4_gradient_hook_handles = hooks
    model._d4_gradient_hooks_ready = True


class GuardedTrainerMixin:
    """Run through inherited native loop but block unsafe loss, update and recovery paths."""

    def _setup_train(self):
        super()._setup_train()
        model = unwrap_model(self.model)
        require_finite_model(model, "pre_training_model")
        register_fail_closed_gradient_hooks(model)
        _state_tensor_scan(self.optimizer.state_dict(), "pre_training_optimizer", "optimizer")
        _state_tensor_scan(self.scaler.state_dict(), "pre_training_amp", "scaler")
        if self.ema is not None and self.ema.ema is not None:
            require_finite_model(self.ema.ema, "pre_training_ema")
        # A GradScaler may skip Optimizer.step on overflow. An optimizer post-step
        # hook proves that an update was actually invoked, independent of scale.
        if not hasattr(self.optimizer, "register_step_post_hook"):
            raise RuntimeError("D4 requires optimizer step-post-hook support")
        self._d4_optimizer_steps_observed = 0

        def observed_step(optimizer, args, kwargs):
            self._d4_optimizer_steps_observed += 1

        self._d4_optimizer_hook = self.optimizer.register_step_post_hook(observed_step)

    def optimizer_step(self):
        """Exact native clipping/step/EMA order with nonfinite guards and AMP skip veto."""
        model = unwrap_model(self.model)
        require_finite_model(model, "pre_update_model")
        if self.ema is not None and self.ema.ema is not None:
            require_finite_model(self.ema.ema, "pre_update_ema")
        _state_tensor_scan(self.optimizer.state_dict(), "pre_update_optimizer", "optimizer")
        _state_tensor_scan(self.scaler.state_dict(), "pre_update_amp", "scaler")
        self.scaler.unscale_(self.optimizer)
        require_finite_gradients(model, "unscaled_before_clip")
        # Preserve the repository native max_norm exactly, AFTER the gradient scan.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        require_finite_gradients(model, "after_clip")
        scale_before = float(self.scaler.get_scale())
        if not math.isfinite(scale_before) or scale_before <= 0:
            _abort("pre_update_amp_scale", "scaler")
        prior_steps = self._d4_optimizer_steps_observed
        self.scaler.step(self.optimizer)
        if self._d4_optimizer_steps_observed != prior_steps + 1:
            _abort("optimizer_step_skipped_or_duplicated", "optimizer")
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        if not math.isfinite(scale_after) or scale_after <= 0 or scale_after < scale_before:
            _abort("amp_overflow_or_optimizer_step_skipped", "scaler")
        require_finite_model(model, "post_update_model")
        _state_tensor_scan(self.optimizer.state_dict(), "post_update_optimizer", "optimizer")
        _state_tensor_scan(self.scaler.state_dict(), "post_update_amp", "scaler")
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
            require_finite_model(self.ema.ema, "post_update_ema")

    def _handle_nan_recovery(self, epoch):
        """Ban upstream auto-recovery; preserve ordinary finite metric behavior."""
        if self.loss is not None:
            require_finite_tensor(self.loss, "epoch_loss", "loss")
        if self.fitness is not None and not math.isfinite(float(self.fitness)):
            _abort("epoch_fitness", "fitness")
        if self.best_fitness is not None and not math.isfinite(float(self.best_fitness)):
            _abort("epoch_best_fitness", "best_fitness")
        return False


__all__ = (
    "D4NumericalAbort", "GuardedEarlyDetectionModel", "GuardedVGRADetectionModel",
    "GuardedTrainerMixin", "require_finite_loss", "require_finite_model", "require_finite_buffers",
    "require_finite_gradients", "verify_pretrained_source", "verify_early_tensor_transfer",
    "pinned_model_overrides", "register_fail_closed_gradient_hooks",
)
