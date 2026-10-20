#!/usr/bin/env python3
"""R9B R2: fail-closed *candidate* native session-boundary adapter.

This is not an executable training launcher. It must be composed with a separately
verified G4-bound native trainer, in an explicitly authorized/frozen release.
It never handles epoch100 or permits a terminal checkpoint to be called resumable.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json

ARMS = {'MV00', 'MV01'}

class R9BPublishedSessionBoundary(Exception):
    """Explicit nonterminal boundary; caller must export/verify off-session."""


def fail(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError('R9B_R2_' + code)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class R9BPostPublicationSessionStopMixin:
    """Intercept the **completed** native G4 epoch hook, not the training batch loop.

    Stop exception is thrown ONLY after inherited guarded native epoch save,
    trusted recovery capsule and G4 publication complete. Native final_eval is
    intentionally NOT invoked for an ordinary end-of-Kaggle-session boundary.
    The caller MUST catch only this exception and require off-site transport.
    """

    def r9b_bind_nonterminal_session_boundary(self, *, arm: str,
                                               target_completed_epoch: int,
                                               publication_root: str | Path,
                                               verify_epoch_fn):
        fail(not getattr(self, '_r9b_session_bound', False), 'REPEATED_BIND')
        fail(arm in ARMS, 'ARM')
        fail(type(target_completed_epoch) is int and 1 <= target_completed_epoch <= 99,
             'NONTERMINAL_TARGET_REQUIRED')
        expected = {'EarlyPairedTrainer':'MV00','VGRADetectionTrainer':'MV01'}.get(type(self).__name__)
        fail(expected == arm, 'NATIVE_ARM_MISMATCH')
        fail(callable(verify_epoch_fn), 'VERIFIER_REQUIRED')
        fail(getattr(self, '_d4e_recovery_configured', False), 'NATIVE_RECOVERY_NOT_CONFIGURED')
        fail(getattr(self, '_g4_pub_bound', False), 'G4_NOT_BOUND')
        root = Path(publication_root).resolve(strict=True)
        fail(root.is_dir() and not root.is_symlink(), 'PUBLICATION_ROOT')
        fail(Path(self._g4_pub_root).resolve() == root, 'G4_PUBLICATION_ROOT_DRIFT')
        self._r9b_session_bound = True
        self._r9b_session_arm = arm
        self._r9b_session_target = target_completed_epoch - 1
        self._r9b_session_publication_root = root
        self._r9b_session_verify_epoch = verify_epoch_fn
        self._r9b_session_stopped = False

    def d4e_recovery_epoch_end(self, epoch: int, nb: int, last_opt_step: int):
        fail(getattr(self,'_r9b_session_bound',False), 'BOUNDARY_NOT_CONFIGURED')
        result = super().d4e_recovery_epoch_end(epoch, nb, last_opt_step)
        if epoch != self._r9b_session_target:
            return result
        fail(not getattr(self, '_r9b_session_stopped', False), 'DUPLICATE_STOP')
        fail(epoch < 99 and type(epoch) is int, 'TERMINAL_NOT_ALLOWED')
        fail(not getattr(self, 'stop', False), 'NATIVE_STOP_MUST_BE_FALSE')
        entries = getattr(self, '_g4_completed', None)
        fail(isinstance(entries, list) and entries, 'G4_PUBLICATION_MISSING')
        entry = entries[-1]
        fail(entry.get('arm') == self._r9b_session_arm and
             entry.get('completed_epoch') == epoch and entry.get('terminal') is False,
             'G4_CURSOR_OR_TERMINAL_DRIFT')
        pin = entry.get('manifest_sha256')
        fail(isinstance(pin,str) and len(pin)==64 and
             all(ch in '0123456789abcdef' for ch in pin), 'G4_MANIFEST_PIN')
        checked = self._r9b_session_verify_epoch(root=self._r9b_session_publication_root,
                   arm=self._r9b_session_arm, epoch=epoch, external_manifest_sha256=pin)
        fail(checked.get('status') == 'G4C_LOCAL_VERIFICATION_PASS_ONLY' and
             checked.get('completed_epoch') == epoch and checked.get('terminal') is False,
             'G4_LOCAL_BYTE_VERIFICATION')
        receipts = getattr(self, '_d4e_epoch_receipts', None)
        fail(isinstance(receipts,list) and receipts, 'CAPSULE_RECEIPT_MISSING')
        receipt = receipts[-1]
        fail(isinstance(receipt.get('trusted_receipt_sha256'),str) and
             len(receipt['trusted_receipt_sha256']) == 64, 'CAPSULE_RECEIPT_PIN')
        fail(int(getattr(self, '_d4e_optimizer_updates',0)) > 0, 'OPTIMIZER_UPDATES_MISSING')
        fail(getattr(self, '_d4_optimizer_steps_observed',None) == self._d4e_optimizer_updates,
             'OPTIMIZER_POSTHOOK_MISMATCH')
        fail(getattr(getattr(self, 'ema',None),'updates',None) == self._d4e_optimizer_updates,
             'EMA_UPDATE_MISMATCH')
        scaler = getattr(self,'scaler',None)
        fail(scaler is not None and scaler.is_enabled() and float(scaler.get_scale()) >= 1.0,
             'AMP_SCALER_NOT_VALID')
        self._r9b_session_stopped = True
        # This is NOT an offsite receipt, nor an authorization of official training.
        event = {'schema':'CHAT23_R9B_R2_NONTERMINAL_BOUNDARY_LOCAL_ONLY',
                 'arm':self._r9b_session_arm,'completed_epoch':epoch,
                 'next_epoch':epoch+1,'terminal':False,
                 'g4_manifest_sha256':pin,
                 'capsule_trusted_receipt_sha256':receipt['trusted_receipt_sha256'],
                 'optimizer_updates':self._d4e_optimizer_updates,
                 'offsession_export_verified':False,'training_authorization_from_adapter':False,
                 'b_test_access':'NONE'}
        print('R9B_NONTERMINAL_BOUNDARY='+json.dumps(event,sort_keys=True),flush=True)
        raise R9BPublishedSessionBoundary('R9B_BOUNDED_AFTER_LOCAL_AUTHENTIC_G4; OFFSITE_EXPORT_PENDING')


def assemble_candidate(native, guarded_bind_function):
    """Actual class composition, with no constructor invocation or GPU execution."""
    name = getattr(native,'__name__',None)
    fail(name in ('EarlyPairedTrainer','VGRADetectionTrainer'), 'UNKNOWN_NATIVE_CLASS')
    fail(callable(guarded_bind_function), 'MISSING_G4_BINDER')
    guarded = guarded_bind_function(native)
    fail(guarded.__name__ == name, 'G4_CLASS_NAME_DRIFT')
    fail(R9BPostPublicationSessionStopMixin not in guarded.__mro__, 'ALREADY_BOUND')
    cls = type(name, (R9BPostPublicationSessionStopMixin,guarded), {'__module__':__name__})
    mro = [c.__name__ for c in cls.__mro__]
    fail(mro.index('R9BPostPublicationSessionStopMixin') < mro.index('G4IntegratedEpochMixin'),
         'UNSAFE_MRO')
    return cls
