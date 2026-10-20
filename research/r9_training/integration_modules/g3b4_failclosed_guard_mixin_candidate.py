"""Chat21 Gate3-B4 disposable CPU-integration CANDIDATE; not a release.

Requires explicit binding ahead of D4EVerifiedRecoveryTrainerMixin in the
production trainer MRO, independent source provenance verification, and a
separately approved official launcher.  NO GPU training is authorized.
This module does not alter architecture, losses, optimizer, or recovery state.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from gate3b_finite_guard_candidate import (
    NonFiniteAbort, guard_loss, guard_gradients, guard_after_update,
    guard_epoch_boundary, guard_model_parameters_buffers, guard_optimizer, guard_ema,
    _scalar_and_tensor_finite,
)

ALLOWED = {'EarlyPairedTrainer': ('MV00', 3), 'VGRADetectionTrainer': ('MV01', 4)}
SAFE_PHASES = {'PREFLIGHT', 'LOSS', 'AFTER_BACKWARD', 'PRE_STEP', 'POST_STEP',
               'VALIDATION', 'NO_NATIVE_RETRY', 'PRE_SAVE', 'EPOCH_BOUNDARY', 'TERMINAL'}

class G3B4FiniteGuardMixin:
    """Fail-closed methods designed for future private, verified source bindings.

    The following methods MUST be reached by reviewed native-hook integration;
    stand-alone CPU tests are not proof that such binding exists in the real fork.
    """

    def d4e_g3b4_bind_guard(self, *, arm: str, failure_root: str | Path) -> None:
        if getattr(self, '_g3b4_ready', False):
            raise RuntimeError('G3B4_DUPLICATE_GUARD_BINDING')
        expected = ALLOWED.get(type(self).__name__)
        if expected is None or arm != expected[0]:
            raise RuntimeError('G3B4_CLASS_ARM_NOT_AUTHORIZED')
        root = Path(failure_root)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise RuntimeError('G3B4_FAILURE_ROOT_MUST_BE_EXISTING_ABSOLUTE_DIRECTORY')
        receipt = root / 'G3B4_FIRST_FAILURE.json'
        if receipt.exists() or receipt.is_symlink():
            raise RuntimeError('G3B4_REJECT_STALE_FAILURE_RECEIPT')
        self._g3b4_arm = arm
        self._g3b4_expected_components = expected[1]
        self._g3b4_failure_root = root
        self._g3b4_first_failure_recorded = False
        self._g3b4_ready = True

    def _g3b4_assert_bound(self):
        if not getattr(self, '_g3b4_ready', False):
            raise NonFiniteAbort('G3B4_GUARDS_NOT_EXPLICITLY_BOUND')
        if ALLOWED.get(type(self).__name__) != (self._g3b4_arm, self._g3b4_expected_components):
            raise NonFiniteAbort('G3B4_TRAINER_ARM_OR_COMPONENT_COUNT_DRIFT')
        if not getattr(self, '_d4e_recovery_configured', False):
            raise NonFiniteAbort('G3B4_RECOVERY_NOT_CONFIGURED')
        observed = getattr(self, '_d4e_verified_identity', None)
        active = getattr(self, '_d4e_recovery_identity', None)
        if not isinstance(observed, dict) or not isinstance(active, dict) or observed != active:
            raise NonFiniteAbort('G3B4_VERIFIED_IDENTITY_NOT_ACTIVE')
        if active.get('arm') != self._g3b4_arm:
            raise NonFiniteAbort('G3B4_RECOVERY_ARM_DRIFT')
        if getattr(self, '_d4e_r3_phase', None):
            raise NonFiniteAbort('G3B4_REHEARSAL_MIXIN_NOT_PRODUCTION')
        args = getattr(self, 'args', None)
        if args is None:
            raise NonFiniteAbort('G3B4_MISSING_TRAINING_ARGS')
        for k, v in {'amp':True,'epochs':100,'batch':8,'imgsz':1024,
                     'workers':0,'seed':42,'nbs':64,'save':True,'val':True,
                     'resume':False,'compile':False}.items():
            if getattr(args, k, None) != v:
                raise NonFiniteAbort(f'G3B4_RECIPE_DRIFT:{k}')
        if str(getattr(args,'optimizer','')).upper()!='SGD':
            raise NonFiniteAbort('G3B4_OPTIMIZER_NOT_SGD')
        if getattr(self,'world_size',1)!=1:
            raise NonFiniteAbort('G3B4_DDP_NOT_AUTHORIZED')
        if not bool(getattr(getattr(self,'scaler',None),'is_enabled',lambda: False)()):
            raise NonFiniteAbort('R7C_AMP_SCALER_NOT_ENABLED')
        if self._g3b4_failure_root.is_symlink() or not self._g3b4_failure_root.is_dir():
            raise NonFiniteAbort('G3B4_FAILURE_DIRECTORY_DRIFT')

    def _g3b4_report_first_failure(self, phase, epoch, batch_index, exception):
        if getattr(self,'_g3b4_first_failure_recorded',False):
            return
        self._g3b4_first_failure_recorded = True
        root = getattr(self,'_g3b4_failure_root',None)
        if root is None: return
        # Privacy: no images, patient IDs, labels, dataset paths, raw batch, or stack trace.
        safe_error = re.sub(r'[^a-zA-Z0-9_:\[\]().+\-]', '_', str(exception))[:256]
        payload = {'schema':'CHAT21_G3B4_FIRST_FAILURE_R1',
                   'arm':getattr(self,'_g3b4_arm','UNBOUND'),
                   'phase':phase, 'epoch':epoch,'batch_index':batch_index,
                   'error_type':type(exception).__name__, 'error_code':safe_error,
                   'no_batch_data':True,'official_training_authorized':False,'b_test_access':'NONE'}
        receipt = root/'G3B4_FIRST_FAILURE.json'
        with receipt.open('x',encoding='utf-8') as f:
            json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')

    def _g3b4_call(self, phase, epoch, batch_index, fn):
        if phase not in SAFE_PHASES: raise RuntimeError('G3B4_UNKNOWN_CHECK_PHASE')
        try:
            self._g3b4_assert_bound()
            return fn()
        except Exception as error:
            try: self._g3b4_report_first_failure(phase,epoch,batch_index,error)
            except Exception as receipt_failure:
                raise RuntimeError('G3B4_FAILURE_EVIDENCE_COMMIT_FAILED') from receipt_failure
            # Never return a recoverable status, skip the update, or invoke native retry.
            raise

    def d4e_g3b_guard_preflight(self):
        def fn():
            guard_model_parameters_buffers(self.model,phase='PREFLIGHT_MODEL')
            guard_optimizer(self.optimizer,phase='PREFLIGHT_OPTIMIZER')
            guard_ema(self.ema,phase='PREFLIGHT_EMA')
        return self._g3b4_call('PREFLIGHT',None,None,fn)

    def d4e_g3b_guard_loss(self, *, epoch, batch_index):
        def fn():
            items = self.loss_items
            if not hasattr(items,'__len__') or len(items)!=self._g3b4_expected_components:
                raise NonFiniteAbort('G3B4_LOSS_COMPONENT_ARITY_DRIFT')
            guard_loss(self.loss, items, arm=self._g3b4_arm, epoch=epoch,batch_index=batch_index)
        return self._g3b4_call('LOSS',epoch,batch_index,fn)

    def d4e_g3b_guard_after_backward(self, *, epoch, batch_index):
        # MUST run for every microbatch, even when accumulate threshold is not reached.
        return self._g3b4_call('AFTER_BACKWARD',epoch,batch_index,
            lambda: guard_gradients(self.model,phase=f'{self._g3b4_arm}:E{epoch}:B{batch_index}:MICROBATCH'))

    def d4e_g3b_guard_pre_step(self, *, epoch, batch_index):
        return self._g3b4_call('PRE_STEP',epoch,batch_index,
            lambda: guard_gradients(self.model,phase=f'{self._g3b4_arm}:E{epoch}:B{batch_index}:SCALED_PRE_STEP'))

    def d4e_g3b_guard_post_step(self, *, epoch, batch_index):
        return self._g3b4_call('POST_STEP',epoch,batch_index,
            lambda: guard_after_update(self.model,self.optimizer,self.ema,
                phase=f'{self._g3b4_arm}:E{epoch}:B{batch_index}'))

    def d4e_g3b_guard_validation(self, *, epoch):
        def fn():
            metrics = getattr(self,'metrics',None)
            if not isinstance(metrics,dict) or not metrics:
                raise NonFiniteAbort('G3B4_MISSING_VALIDATION_METRICS')
            _scalar_and_tensor_finite(metrics,'VALIDATION_METRICS')
            for field in ('fitness','best_fitness'):
                value=getattr(self,field,None)
                if value is None: raise NonFiniteAbort('G3B4_MISSING_'+field.upper())
                _scalar_and_tensor_finite(value,'VALIDATION_'+field)
        return self._g3b4_call('VALIDATION',epoch,None,fn)

    def d4e_g3b_guard_no_native_retry(self, *, epoch):
        # This takes the place of native _handle_nan_recovery; never call native retry.
        return self._g3b4_call('NO_NATIVE_RETRY',epoch,None,lambda: None)

    def d4e_g3b_guard_pre_save(self, *, epoch):
        def fn():
            self.d4e_g3b_guard_validation(epoch=epoch)
            if getattr(self,'tloss',None) is None:
                raise NonFiniteAbort('G3B4_MISSING_ACCUMULATED_TRAIN_LOSS')
            _scalar_and_tensor_finite(self.tloss,'PRE_SAVE_TRAIN_LOSS')
            guard_after_update(self.model,self.optimizer,self.ema,phase='PRE_SAVE_STATE')
        return self._g3b4_call('PRE_SAVE',epoch,None,fn)

    def d4e_g3b_guard_epoch_boundary(self, *, epoch):
        return self._g3b4_call('EPOCH_BOUNDARY',epoch,None,
            lambda: guard_epoch_boundary(self.model,self.optimizer,self.ema,arm=self._g3b4_arm,epoch=epoch))

    def d4e_g3b_guard_terminal(self):
        # No terminal export / selected-best integrity proof yet: fail CLOSED.
        def fn():
            guard_after_update(self.model,self.optimizer,self.ema,phase='TERMINAL_STATE')
            raise RuntimeError('G3B4_TERMINAL_ARTIFACT_PROTOCOL_NOT_YET_REVIEWED')
        return self._g3b4_call('TERMINAL',None,None,fn)

__all__ = ['G3B4FiniteGuardMixin']
