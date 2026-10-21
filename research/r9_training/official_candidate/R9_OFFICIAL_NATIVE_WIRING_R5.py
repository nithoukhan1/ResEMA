#!/usr/bin/env python3
"""R5 native wiring candidate, CPU-auditable, not an authorized GPU worker.

Preserves the frozen MV00/MV01 trainers, G3B5/G4 integration, and replaces
only the R9B one-epoch stop wrapper with the R4 adaptive post-G4 stop wrapper.
This does not prove actual CUDA or terminal execution. Historical source is
imported only when this function is explicitly called in a verified source stage.
"""
from __future__ import annotations
from pathlib import Path

class R5WiringError(RuntimeError): pass


def need(ok, reason):
    if not ok: raise R5WiringError('R9_OFFICIAL_R5_WIRING_' + reason)


def compose_native_candidate(native, guarded_binder, adaptive_mixin):
    """Compose only authenticated source types with exact G4 ordering.

    Inputs injected explicitly, avoiding torch/ultralytics imports at module import
    time. Does not instantiate a trainer; no GPU use. Native class's original
    __name__ is preserved for pair-safe trainer dispatch.
    """
    name = getattr(native, '__name__', None)
    need(name in ('EarlyPairedTrainer','VGRADetectionTrainer'), 'INVALID_NATIVE')
    need(callable(guarded_binder) and isinstance(adaptive_mixin,type), 'BINDER')
    need(adaptive_mixin.__name__ == 'R9OfficialAdaptiveBoundaryMixin', 'ADAPTER_IDENTITY')
    guarded = guarded_binder(native)
    need(guarded.__name__ == name, 'CLASS_RENAME')
    old = [c.__name__ for c in guarded.__mro__]
    need('G4IntegratedEpochMixin' in old and 'G3B5PrivacySafeGuardMixin' in old,
         'G4_AND_GUARD_MISSING')
    need(not any(x in old for x in ('R9BPostPublicationSessionStopMixin',
                                    'R9OfficialAdaptiveBoundaryMixin')),
         'OLDER_BOUNDARY_ALREADY_INSTALLED')
    class AmpGate:
        def __init__(self, *a, **kw):
            self._r7c_amp_gate_ready = True
            super().__init__(*a, **kw)
    final = type(name, (AmpGate, adaptive_mixin, guarded), {'__module__':__name__})
    mro = [c.__name__ for c in final.__mro__]
    need(mro.count('R9OfficialAdaptiveBoundaryMixin') == 1 and
         mro.count('G4IntegratedEpochMixin') == 1 and
         mro.index('R9OfficialAdaptiveBoundaryMixin') < mro.index('G4IntegratedEpochMixin'),
         'UNSAFE_MRO')
    need(all(callable(getattr(final, m, None)) for m in (
        'r9_official_bind_boundary','d4e_recovery_epoch_end',
        'd4e_configure_recovery','d4e_g3b5_bind_guard',
        'g4_bind_epoch_publication','final_eval','d4e_g3b_guard_terminal')),
        'NATIVE_HOOKS')
    return final


def verify_external_release_boundary(*, original_contract, new_release):
    """Review-only: NEVER converts the historical disabled contract into approval."""
    need(original_contract.get('official_training_authorized') is False and
         original_contract.get('run_executable') is False,
         'HISTORICAL_R1_MUST_STAY_FALSE')
    need(new_release.get('status') == 'INTEGRATION_REVIEW_ONLY', 'R5_REVIEW_STATUS')
    need(new_release.get('official_training_authorized') is False and
         new_release.get('gpu_execution_allowed') is False,
         'UNAUTHORIZED_RELEASE')
    need(new_release.get('scientific_source_commit') ==
         original_contract.get('source',{}).get('baseline_git_commit'),
         'FROZEN_SOURCE_COMMIT')
    need(new_release.get('frozen_recipe_sha256') ==
         original_contract.get('source',{}).get('frozen_recipe_sha256'),
         'FROZEN_RECIPE')
    need(new_release.get('b_test_access') == 'NONE', 'TEST_ACCESS')
    return {'status':'PASS_REVIEW_WIRING_ONLY', 'gpu_executed':False,
            'terminal_epoch_100_verified':False,'official_training_authorized':False,
            'b_test_access':'NONE'}
