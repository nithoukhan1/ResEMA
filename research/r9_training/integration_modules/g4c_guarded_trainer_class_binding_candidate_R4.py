"""Guarded trainer class composition CANDIDATE; does not instantiate or train.

Call only with authenticated native MV00/MV01 trainer types in a disposable
pinned source stage after installing G3-B4 native-loop hook patch.
"""
from __future__ import annotations

from g3b5_privacy_safe_guard_mixin_candidate import G3B5PrivacySafeGuardMixin
from g4_integrated_epoch_mixin_candidate_R4 import G4IntegratedEpochMixin

REQUIRED_CHAIN=('G4IntegratedEpochMixin','G3B5PrivacySafeGuardMixin',
                'G3B4FiniteGuardMixin','D4EVerifiedRecoveryTrainerMixin',
                'D4ERecoveryTrainerMixin','GuardedTrainerMixin','DetectionTrainer')
ALLOWED={'EarlyPairedTrainer':'MV00','VGRADetectionTrainer':'MV01'}


def bind_candidate_class(native):
    name=getattr(native,'__name__',None)
    if name not in ALLOWED:
        raise RuntimeError('G4C_UNKNOWN_NATIVE_TRAINER_CLASS')
    if any(cls.__name__ in ('G3B5PrivacySafeGuardMixin','G4IntegratedEpochMixin','D4ER3BoundedMixin') for cls in native.__mro__):
        raise RuntimeError('G4C_NATIVE_TYPE_ALREADY_MODIFIED_OR_BOUNDED')
    cls=type(name,(G4IntegratedEpochMixin,G3B5PrivacySafeGuardMixin,native),{'__module__':__name__})
    actual=[c.__name__ for c in cls.__mro__]
    if any(actual.count(k)!=1 for k in REQUIRED_CHAIN):
        raise RuntimeError('G4C_GUARD_OR_RECOVERY_MRO_INCOMPLETE')
    if [actual.index(k) for k in REQUIRED_CHAIN] != sorted(actual.index(k) for k in REQUIRED_CHAIN):
        raise RuntimeError('G4C_GUARD_OR_RECOVERY_MRO_ORDER_INVALID')
    for function in ('d4e_g3b5_bind_guard','g4_bind_epoch_publication',
                     'd4e_configure_recovery','d4e_g3b_guard_loss',
                     'd4e_g3b_guard_after_backward','d4e_g3b_guard_pre_step',
                     'd4e_g3b_guard_post_step','d4e_g3b_guard_pre_save',
                     'd4e_recovery_epoch_end','d4e_g3b_guard_terminal'):
        if not callable(getattr(cls,function,None)):
            raise RuntimeError('G4C_REQUIRED_GOVERNED_METHOD_MISSING')
    return cls
