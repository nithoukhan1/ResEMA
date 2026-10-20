"""Chat21 G3-B5: PRIVACY-HARDENED, CPU-REVIEW-ONLY guard candidate.

Subclass of the unchanged G3-B4 finite guards. No official-run launcher,
no native trainer patch, no clinical data, and no permission to train.

Failure receipts contain strict fixed-schema codes, never str(exception),
tracebacks, tensor keys, model parameter names, dataset paths or identifiers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import stat

from g3b4_failclosed_guard_mixin_candidate import (
    ALLOWED, SAFE_PHASES, G3B4FiniteGuardMixin,
)
from gate3b_finite_guard_candidate import NonFiniteAbort

SCHEMA = 'CHAT21_G3B5_FIRST_FAILURE_R1'
ALLOWED_CODES = frozenset({'G3B5_NONFINITE_OR_INVARIANT', 'G3B5_UNEXPECTED_ERROR'})


def _validate_context(value, maximum, name):
    if value is None:
        return None
    if type(value) is not int or not (0 <= value <= maximum):
        raise RuntimeError('G3B5_INVALID_' + name + '_CONTEXT')
    return value


def _atomic_create_no_overwrite(folder: Path, basename: str, payload: dict) -> None:
    """POSIX: fsync temp, then atomic hard-link install (fails if target exists).

    This deliberately fails closed if filesystem won't support atomic hard link.
    The receipt is committed in its own private pre-created failure directory.
    """
    if os.name != 'posix':
        raise RuntimeError('G3B5_ATOMIC_RECEIPT_REQUIRES_POSIX_REVIEW')
    if not folder.is_dir() or folder.is_symlink() or not folder.is_absolute():
        raise RuntimeError('G3B5_RECEIPT_DIRECTORY_NOT_PRIVATE')
    dest = folder / basename
    if dest.exists() or dest.is_symlink():
        raise RuntimeError('G3B5_RECEIPT_ALREADY_PRESENT')
    tmp = folder / ('.g3b5_' + secrets.token_hex(12) + '.pending')
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    data = (json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('ascii')
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=True) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # os.link creates dest atomically if-and-only-if it does not exist.
        os.link(tmp, dest, follow_symlinks=False)
        try:
            dirfd = os.open(folder, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except OSError:
            raise RuntimeError('G3B5_DIRECTORY_FSYNC_FAILED') from None
    finally:
        tmp.unlink(missing_ok=True)


class G3B5PrivacySafeGuardMixin(G3B4FiniteGuardMixin):
    """Guard must precede verified recovery and native Detect trainer in MRO."""

    def d4e_g3b5_bind_guard(self, *, arm: str, failure_root: str | Path) -> None:
        if getattr(self, '_g3b5_ready', False):
            raise RuntimeError('G3B5_DUPLICATE_BINDING')
        # Require caller-owned private directory, without symlinked ancestors.
        root = Path(failure_root)
        if (os.name != 'posix' or not root.is_absolute() or
                not root.is_dir() or root.absolute() != root.resolve(strict=True) or
                root.stat().st_uid != os.getuid() or
                stat.S_IMODE(root.stat().st_mode) & 0o077):
            raise RuntimeError('G3B5_PRIVATE_FAILURE_ROOT_MODE_REQUIRED')
        # The predecessor verifies recipe-compatible trainer names and root.
        super().d4e_g3b4_bind_guard(arm=arm, failure_root=root)
        self._g3b5_ready = True

    def _g3b4_assert_bound(self):
        if not getattr(self, '_g3b5_ready', False):
            raise NonFiniteAbort('G3B5_BINDING_REQUIRED')
        if G3B5PrivacySafeGuardMixin not in type(self).__mro__:
            raise NonFiniteAbort('G3B5_MRO_MISSING')
        classes = [cls.__name__ for cls in type(self).__mro__]
        required = ['G3B5PrivacySafeGuardMixin', 'G3B4FiniteGuardMixin',
                    'D4EVerifiedRecoveryTrainerMixin', 'D4ERecoveryTrainerMixin',
                    'GuardedTrainerMixin', 'DetectionTrainer']
        if any(classes.count(x) != 1 for x in required):
            raise NonFiniteAbort('G3B5_MRO_REQUIRED_CLASSES_MISSING')
        if [classes.index(x) for x in required] != sorted(classes.index(x) for x in required):
            raise NonFiniteAbort('G3B5_MRO_ORDER_DRIFT')
        if 'D4ER3BoundedMixin' in classes:
            raise NonFiniteAbort('G3B5_REHEARSAL_CLASS_NOT_PRODUCTION')
        return super()._g3b4_assert_bound()

    def _g3b4_report_first_failure(self, phase, epoch, batch_index, error):
        if getattr(self, '_g3b4_first_failure_recorded', False):
            return
        if phase not in SAFE_PHASES:
            raise RuntimeError('G3B5_INVALID_PHASE')
        epoch = _validate_context(epoch, 100, 'EPOCH')
        batch_index = _validate_context(batch_index, 1000000, 'BATCH_INDEX')
        arm = getattr(self, '_g3b4_arm', None)
        if arm not in {'MV00', 'MV01'} or ALLOWED.get(type(self).__name__, (None,))[0] != arm:
            raise RuntimeError('G3B5_ARM_NOT_AUTHORIZED')
        code = ('G3B5_NONFINITE_OR_INVARIANT' if isinstance(error, NonFiniteAbort)
                else 'G3B5_UNEXPECTED_ERROR')
        root = getattr(self, '_g3b4_failure_root', None)
        if not isinstance(root, Path):
            raise RuntimeError('G3B5_FAILURE_ROOT_NOT_BOUND')
        record = {
            'schema': SCHEMA, 'arm': arm, 'phase': phase,
            'epoch': epoch, 'batch_index': batch_index, 'error_code': code,
            'no_batch_data': True, 'b_test_access': 'NONE',
            'official_training_authorized': False,
        }
        assert code in ALLOWED_CODES
        _atomic_create_no_overwrite(root, 'G3B5_FIRST_FAILURE.json', record)
        self._g3b4_first_failure_recorded = True

    def _g3b4_call(self, phase, epoch, batch_index, fn):
        if phase not in SAFE_PHASES:
            raise RuntimeError('G3B5_UNKNOWN_PHASE')
        try:
            self._g3b4_assert_bound()
            return fn()
        except Exception as error:
            try:
                self._g3b4_report_first_failure(phase, epoch, batch_index, error)
            except Exception:
                # No exception text is copied into a new error or a receipt.
                raise RuntimeError('G3B5_FIRST_FAILURE_RECEIPT_COMMIT_FAILED') from None
            # Do not propagate source exception strings with clinical identifiers.
            raise RuntimeError('G3B5_GOVERNED_ABORT_RECORDED') from None


__all__ = ['G3B5PrivacySafeGuardMixin', 'SCHEMA']
