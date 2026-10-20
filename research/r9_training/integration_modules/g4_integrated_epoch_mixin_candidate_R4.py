"""Chat21 production-lifecycle integration *candidate*, not a release.

MRO position: G4IntegratedEpochMixin, G3B5PrivacySafeGuardMixin,
verified native trainer. Requires G3-B4 native-loop patch. Does not train.
Off-session transfer is NOT implemented: no unattended GPU release.
"""
from __future__ import annotations

import hashlib
import shutil
import secrets
from pathlib import Path

from g4_epoch_stream_publisher_candidate_R3 import publish_epoch, _copy_exclusive, _private
from gate3b_finite_guard_candidate import guard_after_update, _scalar_and_tensor_finite


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


class G4IntegratedEpochMixin:
    """Only structural wiring candidate; entire lifecycle requires real tests."""

    def g4_bind_epoch_publication(self, *, output_root):
        if getattr(self,'_g4_pub_bound',False):raise RuntimeError('G4C_ALREADY_BOUND')
        if not getattr(self,'_g3b5_ready',False):raise RuntimeError('G4C_PRIVACY_GUARD_MUST_BE_BOUND_FIRST')
        if not getattr(self,'_d4e_verified_identity',None):raise RuntimeError('G4C_VERIFIED_RECOVERY_REQUIRED')
        from g4_epoch_stream_publisher_candidate_R3 import _private
        self._g4_pub_root=_private(Path(output_root))
        self._g4_pub_bound=True
        self._g4_guard_records={}
        self._g4_completed=[]

    def d4e_g3b_guard_validation(self, *, epoch):
        result=super().d4e_g3b_guard_validation(epoch=epoch)
        if getattr(self,'_g4_pub_bound',False):
            self._g4_guard_records.setdefault(epoch,{})['validator_pass']=True
        return result

    def d4e_g3b_guard_pre_save(self, *, epoch):
        result=super().d4e_g3b_guard_pre_save(epoch=epoch)
        if not getattr(self,'_g4_pub_bound',False):raise RuntimeError('G4C_MISSING_PUBLISHER_BINDING')
        self._g4_guard_records.setdefault(epoch,{})['pre_save_epoch']=epoch
        return result

    def save_model(self):
        """Record the exact file digest produced by native best selection."""
        if not getattr(self,'_g4_pub_bound',False):
            raise RuntimeError('G4C_SAVE_WITHOUT_EXPLICIT_BINDING')
        epoch=getattr(self,'epoch',None)
        if type(epoch) is not int or not self._g4_guard_records.get(epoch,{}).get('pre_save_epoch')==epoch:
            raise RuntimeError('G4C_SAVE_WITHOUT_GUARDED_PRESAVE')
        result=super().save_model()
        best=Path(self.best)
        last=Path(self.last)
        if not best.is_file() or not last.is_file():
            raise RuntimeError('G4C_NATIVE_LAST_OR_BEST_MISSING_AFTER_SAVE')
        self._g4_guard_records[epoch]['selected_best_sha256']=sha(best)
        return result

    def d4e_g3b_guard_epoch_boundary(self, *, epoch):
        result=super().d4e_g3b_guard_epoch_boundary(epoch=epoch)
        if not getattr(self,'_g4_pub_bound',False):raise RuntimeError('G4C_MISSING_PUBLISHER_BINDING')
        self._g4_guard_records.setdefault(epoch,{})['boundary_epoch']=epoch
        return result

    def _g4_evidence(self,epoch):
        record=self._g4_guard_records.get(epoch,{})
        return {'pre_save_epoch':record.get('pre_save_epoch'),
                'boundary_epoch':record.get('boundary_epoch'),
                'validator_pass':record.get('validator_pass',False),
                'selected_best_authenticated':bool(record.get('selected_best_sha256')) and Path(self.best).is_file() and sha(self.best)==record['selected_best_sha256'],
                'recovery_identity_verified':getattr(self,'_d4e_verified_identity',None)==getattr(self,'_d4e_recovery_identity',None),
                'b_test_access':'NONE'}

    def d4e_recovery_epoch_end(self, epoch, nb, last_opt_step):
        if not getattr(self,'_g4_pub_bound',False):raise RuntimeError('G4C_PUBLISHER_NOT_CONFIGURED')
        result=super().d4e_recovery_epoch_end(epoch,nb,last_opt_step)
        # At terminal / early stop the original recovery mixin correctly skips
        # the resumable capsule. Publish terminal after paired final_eval.
        if self.stop or epoch+1>=int(self.epochs):return result
        if not getattr(self,'_d4e_epoch_receipts',None):raise RuntimeError('G4C_MISSING_NEW_CAPSULE_RECEIPT')
        receipt=self._d4e_epoch_receipts[-1]
        pin=receipt.get('trusted_receipt_sha256')
        capsule=Path(self._d4e_checkpoint_root)/f"{self._d4e_recovery_identity['run_id']}_epoch{epoch+1:03d}.pt"
        from ultralytics.models.yolo.detect.d4e_trusted_recovery_capsule import read_authenticated_capsule
        recovered=read_authenticated_capsule(checkpoint_path=capsule,root=self._d4e_checkpoint_root,
                                             trusted_receipt_sha256=pin,identity=self._d4e_recovery_identity)
        cursor=recovered.get('cursor',{})
        stopper=recovered.get('early_stopper_state',{})
        if (cursor.get('completed_epoch')!=epoch or cursor.get('next_epoch')!=epoch+1 or
            stopper.get('selected_best_sha256')!=sha(self.best) or
            self._g4_guard_records.get(epoch,{}).get('selected_best_sha256')!=sha(self.best)):
            raise RuntimeError('G4C_CAPSULE_AND_SELECTED_BEST_IDENTITY_DRIFT')
        published=publish_epoch(root=self._g4_pub_root,arm=self._d4e_recovery_identity['arm'],epoch=epoch,
                                last_path=self.last,best_path=self.best,capsule_path=capsule,
                                capsule_receipt_path=capsule.with_suffix('.receipt.json'),
                                capsule_receipt_pin=pin,terminal=False,evidence=self._g4_evidence(epoch))
        self._g4_completed.append(published)
        print('G4C_EPOCH_LOCAL_MANIFEST_SHA256=',published['manifest_sha256'],flush=True)
        print('G4C_OFFSESSION_EXPORT_PENDING=TRUE',flush=True)
        return result

    def final_eval(self):
        """Evaluate authenticated, disposable checkpoint copies; preserve native best/last.

        This is a production *candidate*, not proof of on-Kaggle behavior.
        The selected-best and recovery checkpoints remain byte-identical to
        the files created by the guarded native save_model path. Any mutation
        of the originals, even if final_eval succeeds, is rejected.
        """
        if not getattr(self, '_g4_pub_bound', False):
            raise RuntimeError('G4C_FINAL_EVAL_PUBLISHER_UNBOUND')
        if getattr(self, '_g4_eval_completed', False):
            raise RuntimeError('G4C_FINAL_EVAL_REPEATED')
        epoch = getattr(self, 'epoch', None)
        record = getattr(self, '_g4_guard_records', {}).get(epoch, {})
        if type(epoch) is not int or not 0 <= epoch < 100:
            raise RuntimeError('G4C_FINAL_EVAL_EPOCH_INVALID')
        best = Path(self.best)
        last = Path(self.last)
        if (best == last or not best.is_file() or best.is_symlink() or
                not last.is_file() or last.is_symlink()):
            raise RuntimeError('G4C_FINAL_EVAL_CHECKPOINTS_MISSING_OR_INVALID')
        original_best_sha = sha(best)
        original_last_sha = sha(last)
        if (record.get('selected_best_sha256') != original_best_sha or
                record.get('pre_save_epoch') != epoch or
                record.get('boundary_epoch') != epoch or
                record.get('validator_pass') is not True):
            raise RuntimeError('G4C_FINAL_EVAL_PRETRAINED_SELECTION_DRIFT')
        root = _private(self._g4_pub_root)
        scratch = root / ('.g4c_final_eval_' + secrets.token_hex(12))
        scratch.mkdir(mode=0o700, exist_ok=False)
        old_best, old_last = self.best, self.last
        evaluated = False
        try:
            copied_best = scratch / 'best.pt'
            copied_last = scratch / 'last.pt'
            if _copy_exclusive(best, copied_best)['sha256'] != original_best_sha:
                raise RuntimeError('G4C_FINAL_EVAL_BEST_COPY_DRIFT')
            if _copy_exclusive(last, copied_last)['sha256'] != original_last_sha:
                raise RuntimeError('G4C_FINAL_EVAL_LAST_COPY_DRIFT')
            self.best, self.last = copied_best, copied_last
            try:
                result = super().final_eval()
                if (not isinstance(getattr(self, 'metrics', None), dict) or
                        not self.metrics):
                    raise RuntimeError('G4C_FINAL_EVAL_METRICS_MISSING')
                evaluated = True
            finally:
                self.best, self.last = old_best, old_last
            if sha(best) != original_best_sha or sha(last) != original_last_sha:
                raise RuntimeError('G4C_FINAL_EVAL_ORIGINAL_CHECKPOINT_BYTES_MUTATED')
            self._g4_eval_attestation = {
                'epoch': epoch,
                'selected_best_sha256': original_best_sha,
                'last_sha256': original_last_sha,
                'evaluation_operated_on_disposable_copies': True,
                'original_checkpoint_bytes_preserved': True,
            }
            self._g4_eval_completed = True
            return result
        finally:
            self.best, self.last = old_best, old_last
            # No inference mutation of the preserved original files may be
            # mistaken for a durable terminal checkpoint publication.
            if scratch.is_dir() and not scratch.is_symlink():
                shutil.rmtree(scratch)

    def d4e_g3b_guard_terminal(self):
        """No B4 always-raise: paired final_eval has completed before this hook.

        This candidate is intentionally not declared ready for official D4-F:
        separate independent final_eval provenance and off-session transfer
        acceptance remain hard requirements.
        """
        if not getattr(self,'_g4_pub_bound',False):raise RuntimeError('G4C_TERMINAL_PUBLICATION_UNBOUND')
        attestation=getattr(self,'_g4_eval_attestation',None)
        if not getattr(self,'_g4_eval_completed',False) or not isinstance(attestation,dict):
            raise RuntimeError('G4C_TERMINAL_FINAL_EVAL_NOT_ATTESTED')
        if attestation.get('epoch') != getattr(self,'epoch',None) or attestation.get('selected_best_sha256') != sha(self.best) or attestation.get('last_sha256') != sha(self.last):
            raise RuntimeError('G4C_TERMINAL_POST_EVAL_CHECKPOINT_DRIFT')
        epoch=getattr(self,'epoch',None)
        if type(epoch) is not int or not 0<=epoch<100:raise RuntimeError('G4C_TERMINAL_EPOCH_UNKNOWN')
        def check():
            guard_after_update(self.model,self.optimizer,self.ema,phase='G4C_TERMINAL_STATE')
            metrics=getattr(self,'metrics',None)
            if not isinstance(metrics,dict) or not metrics:raise RuntimeError('G4C_TERMINAL_METRICS_MISSING')
            _scalar_and_tensor_finite(metrics,'G4C_TERMINAL_METRICS')
            if not Path(self.best).is_file() or not Path(self.last).is_file():
                raise RuntimeError('G4C_TERMINAL_CHECKPOINTS_MISSING')
            if not self._g4_guard_records.get(epoch,{}).get('validator_pass'):
                raise RuntimeError('G4C_FINAL_EPOCH_VALIDATION_NOT_CONFIRMED')
            return publish_epoch(root=self._g4_pub_root,arm=self._d4e_recovery_identity['arm'],epoch=epoch,
                                 last_path=self.last,best_path=self.best,terminal=True,evidence=self._g4_evidence(epoch))
        published=self._g3b4_call('TERMINAL',epoch,None,check)
        self._g4_completed.append(published)
        print('G4C_TERMINAL_LOCAL_MANIFEST_SHA256=',published['manifest_sha256'],flush=True)
        print('G4C_OFFSESSION_EXPORT_PENDING=TRUE',flush=True)

__all__=['G4IntegratedEpochMixin']
