"""Chat21 Gate 3-B REVIEW-ONLY finite-state checks.

Not integrated with the scientific trainer. No training is authorized by this module.
Must be independently evaluated at the correct native/pre-optimizer/post-EMA hooks.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
import math
from typing import Any


class NonFiniteAbort(RuntimeError):
    """Fail-closed signal; callers must not suppress, retry, or replace an update."""


def _torch():
    import torch
    return torch


def _reject(msg: str) -> None:
    raise NonFiniteAbort("G3B_NONFINITE_ABORT:" + msg)


def _scalar_and_tensor_finite(value: Any, path: str) -> None:
    torch = _torch()
    if torch.is_tensor(value):
        v = value.detach()
        if v.is_sparse:
            v = v.coalesce().values()
        if v.is_quantized:
            v = v.dequantize()
        if v.layout != torch.strided:
            # Compressed sparse layouts are not implicitly acceptable until tested.
            _reject(path + ':UNSUPPORTED_TENSOR_LAYOUT:' + str(v.layout))
        if v.numel() and not bool(torch.isfinite(v).all().item()):
            _reject(path)
        return
    if isinstance(value, Number):
        try:
            ok = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            _reject(path + ':INVALID_NUMERIC')
        if not ok:
            _reject(path)
        return
    if isinstance(value, (str, bytes, bool, type(None))):
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            _scalar_and_tensor_finite(v, path + '.' + str(k)[:60])
        return
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _scalar_and_tensor_finite(v, path + f'[{i}]')
        return
    _reject(path + ':UNREVIEWED_STATE_TYPE:' + type(value).__name__)


def guard_loss(total_loss: Any, components: Any, *, arm: str, epoch: int, batch_index: int) -> None:
    """Must execute BEFORE backward(), after the native loss is formed."""
    if arm not in {'MV00', 'MV01'} or epoch < 0 or batch_index < 0:
        _reject('INVALID_CONTEXT')
    _scalar_and_tensor_finite(total_loss, f'{arm}:E{epoch}:B{batch_index}:TOTAL_LOSS')
    _scalar_and_tensor_finite(components, f'{arm}:E{epoch}:B{batch_index}:LOSS_COMPONENTS')


def guard_model_parameters_buffers(model: Any, *, phase: str) -> None:
    """Must run at initialization, after optimizer+EMA update, and at epoch boundary."""
    if model is None or not callable(getattr(model, 'named_parameters', None)):
        _reject(phase + ':INVALID_MODEL')
    parameters = list(model.named_parameters())
    if not parameters:
        _reject(phase + ':NO_PARAMETERS')
    for name, p in parameters:
        _scalar_and_tensor_finite(p, phase + ':PARAM:' + name)
    for name, b in model.named_buffers():
        _scalar_and_tensor_finite(b, phase + ':BUFFER:' + name)


def guard_gradients(model: Any, *, phase: str) -> None:
    """Invoke before optimizer.step(), AFTER any AMP unscaling (approved recipe AMP=False)."""
    parameters = list(model.named_parameters())
    grads = 0
    for name, p in parameters:
        if p.grad is not None:
            grads += 1
            _scalar_and_tensor_finite(p.grad, phase + ':GRAD:' + name)
    if grads == 0:
        _reject(phase + ':NO_GRADIENTS_AT_EXPECTED_STEP')


def guard_optimizer(optimizer: Any, *, phase: str) -> None:
    if optimizer is None or not hasattr(optimizer, 'state') or not hasattr(optimizer, 'param_groups'):
        _reject(phase + ':INVALID_OPTIMIZER')
    for idx, group in enumerate(optimizer.param_groups):
        # Only state-bearing optimizer hyperparameters, not Parameter references.
        for key, value in group.items():
            if key != 'params':
                _scalar_and_tensor_finite(value, f'{phase}:GROUP[{idx}].{key}')
    for idx, state in enumerate(optimizer.state.values()):
        _scalar_and_tensor_finite(state, f'{phase}:STATE[{idx}]')


def guard_ema(ema_or_wrapper: Any, *, phase: str) -> None:
    if ema_or_wrapper is None:
        _reject(phase + ':MISSING_EMA')
    model = getattr(ema_or_wrapper, 'ema', ema_or_wrapper)
    guard_model_parameters_buffers(model, phase=phase + ':EMA')
    if hasattr(ema_or_wrapper, 'updates'):
        _scalar_and_tensor_finite(ema_or_wrapper.updates, phase + ':EMA_UPDATES')


def guard_after_update(model: Any, optimizer: Any, ema_or_wrapper: Any, *, phase: str) -> None:
    """Invoke only AFTER the native optimizer and EMA updates have both finished."""
    guard_model_parameters_buffers(model, phase=phase + ':AFTER_UPDATE')
    guard_optimizer(optimizer, phase=phase + ':AFTER_UPDATE')
    guard_ema(ema_or_wrapper, phase=phase + ':AFTER_UPDATE')


def guard_epoch_boundary(model: Any, optimizer: Any, ema_or_wrapper: Any, *, arm: str, epoch: int) -> None:
    if arm not in {'MV00', 'MV01'} or epoch < 0:
        _reject('INVALID_EPOCH_CONTEXT')
    guard_after_update(model, optimizer, ema_or_wrapper, phase=f'{arm}:E{epoch}:BOUNDARY')


__all__ = [
    'NonFiniteAbort', 'guard_loss', 'guard_model_parameters_buffers', 'guard_gradients',
    'guard_optimizer', 'guard_ema', 'guard_after_update', 'guard_epoch_boundary',
]
