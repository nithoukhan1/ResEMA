#!/usr/bin/env python3
"""CPU-only negative/positive contract regression for C1 final production gate.

This does NOT authenticate native CUDA execution, final_eval, or Kaggle offsite
storage; those require separate real acceptance on the final published source.
"""
import copy, unittest, tempfile, json
from pathlib import Path
from unittest.mock import patch
from R9_PRODUCTION_APPROVAL_GATE_C1 import (validate,validate_acceptance,BASE,SCHEMA,PURPOSE,ACCEPTANCE_SCHEMA,ACCEPTANCE_PURPOSE)
import R9_PRODUCTION_DUAL_SESSION_C1 as dual
import R9_PRODUCTION_NATIVE_WORKER_C1 as worker
import R9_PRODUCTION_PLAN_FACTORY_C1 as factory

SHA='d76cc072e687c2cbef29d12649b869ea3b10fad4'
def spec(arm):
    return {'arm':arm,'physical_gpu':{'MV00':'0','MV01':'1'}[arm],
            'run_id':'C1_VALID_'+arm,'mode':'FRESH_FROM_EARLY',
            'start_epoch_one_based':1,'target_epoch_one_based':4,
            'monotonic_session_seconds':36000,
            'reserve_export_seconds':3600}
def plans():return {a:spec(a) for a in ('MV00','MV01')}
def auth():
    return {'schema':SCHEMA,'purpose':PURPOSE,
            'approval_record':'EXPLICIT_FINAL_USER_APPROVAL_RECORDED',
            'release_commit':SHA,'frozen_science_commit':BASE,
            'run_ids':{'MV00':'C1_VALID_MV00','MV01':'C1_VALID_MV01'},
            'official_training_authorized':True,'b_test_access':'NONE',
            'max_epochs':100,'checkpoint_retention':'KEEP_ORIGINAL_UNTIL_EXTERNAL_SHA_VERIFIED',
            'fresh_pretrained_only':True}

class ApprovalTests(unittest.TestCase):
    def test_review_always_nonexecutable(self):
        self.assertFalse(validate(plans(),None,SHA)['official_training_authorized'])
    def test_final_approval_shape_positive(self):
        self.assertEqual(validate(plans(),auth(),SHA,execute=True)['status'],
             'OFFICIAL_RUN_APPROVAL_RECORD_MATCHED_NOT_EXECUTION_PROOF')
    def test_no_authorization(self):
        with self.assertRaises(RuntimeError):validate(plans(),None,SHA,execute=True)
    def test_old_base_is_not_release(self):
        with self.assertRaises(RuntimeError):validate(plans(),auth(),BASE)
    def test_wrong_release_pin(self):
        with self.assertRaises(RuntimeError):validate(plans(),auth(),'f'*40,execute=True)
    def test_btest_unseal_rejected(self):
        a=auth();a['b_test_access']='ENABLED'
        with self.assertRaises(RuntimeError):validate(plans(),a,SHA,execute=True)
    def test_early_launch_without_approval_rejected(self):
        a=auth();a['official_training_authorized']=False
        with self.assertRaises(RuntimeError):validate(plans(),a,SHA,execute=True)
    def test_approval_run_id_mismatch(self):
        a=auth();a['run_ids']['MV00']='OTHER'
        with self.assertRaises(RuntimeError):validate(plans(),a,SHA,execute=True)
    def test_wrong_gpu(self):
        p=plans();p['MV01']['physical_gpu']='0'
        with self.assertRaises(RuntimeError):validate(p,auth(),SHA,execute=True)
    def test_two_arm_required(self):
        with self.assertRaises(RuntimeError):validate({'MV00':spec('MV00')},auth(),SHA,execute=True)
    def test_no_epoch100_extra(self):
        p=plans();p['MV00']['target_epoch_one_based']=101
        with self.assertRaises(RuntimeError):validate(p,auth(),SHA,execute=True)
    def test_large_session_rejected(self):
        p=plans();p['MV00']['target_epoch_one_based']=20
        with self.assertRaises(RuntimeError):validate(p,auth(),SHA,execute=True)
    def test_time_reserve_rejected(self):
        p=plans();p['MV00']['reserve_export_seconds']=30
        with self.assertRaises(RuntimeError):validate(p,auth(),SHA,execute=True)
    def test_resume_cursor_allowed(self):
        p=plans();p['MV00'].update(mode='RESUME_FROM_OUR_OWN_RUN',
            start_epoch_one_based=15,target_epoch_one_based=20)
        self.assertIn('STATUS' if False else 'status',validate(p,auth(),SHA,execute=True))
    def test_rehearsal_resume_not_from_epoch1(self):
        p=plans();p['MV01'].update(mode='RESUME_FROM_OUR_OWN_RUN',start_epoch_one_based=1)
        with self.assertRaises(RuntimeError):validate(p,auth(),SHA,execute=True)
    def test_no_hidden_authority_key(self):
        a=auth();a['unreviewed']=True
        with self.assertRaises(RuntimeError):validate(plans(),a,SHA,execute=True)

def acceptance():
    return {'schema':ACCEPTANCE_SCHEMA,'purpose':ACCEPTANCE_PURPOSE,
            'release_commit':SHA,'frozen_science_commit':BASE,
            'run_ids':{'MV00':'C1_VALID_MV00','MV01':'C1_VALID_MV01'},
            'official_training_authorized':False,'b_test_access':'NONE',
            'max_completed_epoch_one_based':2}

class AcceptanceTests(unittest.TestCase):
    def test_nonofficial_gpu_acceptance_positive(self):
        p=plans()
        for arm in p:p[arm]['target_epoch_one_based']=1
        self.assertFalse(validate_acceptance(p,acceptance(),SHA)['official_training_authorized'])
    def test_nonofficial_cannot_reach_epoch100(self):
        with self.assertRaises(RuntimeError):validate_acceptance(plans(),acceptance(),SHA)
    def test_nonofficial_cannot_set_true_official(self):
        p=plans()
        for arm in p:p[arm]['target_epoch_one_based']=1
        a=acceptance();a['official_training_authorized']=True
        with self.assertRaises(RuntimeError):validate_acceptance(p,a,SHA)
    def test_acceptance_cannot_use_wrong_release_pin(self):
        p=plans()
        for arm in p:p[arm]['target_epoch_one_based']=1
        with self.assertRaises(RuntimeError):validate_acceptance(p,acceptance(),'f'*40)
    def test_separate_session_epoch2_acceptance(self):
        p=plans()
        for arm in p:p[arm].update(mode='RESUME_FROM_OUR_OWN_RUN',
            start_epoch_one_based=2,target_epoch_one_based=2)
        self.assertEqual(validate_acceptance(p,acceptance(),SHA)['b_test_access'],'NONE')
    def test_nonofficial_rehearsal_epoch3_rejected(self):
        p=plans()
        for arm in p:p[arm].update(mode='RESUME_FROM_OUR_OWN_RUN',
            start_epoch_one_based=3,target_epoch_one_based=3)
        with self.assertRaises(RuntimeError):validate_acceptance(p,acceptance(),SHA)

class PublishingTests(unittest.TestCase):
    def test_no_output_success_on_worker_failures(self):
        p=plans()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                dual.stage_available(specs=p,results={'MV00':{'exit_code':1},'MV01':{'exit_code':2}},
                    output_root='/kaggle/working/test-run-not-created',
                    inspector=lambda *a,**k:None,release_sha=SHA,run_class='NONOFFICIAL_FINALPATH_ACCEPTANCE')
    def test_no_offsite_claim_in_source(self):
        import inspect
        for fn in (worker.execute_one,dual.stage_available):
            self.assertIn('offsite',inspect.getsource(fn).lower())
    def test_worker_has_explicit_switch(self):
        import inspect
        self.assertIn('--execute-approved-final-run',inspect.getsource(worker.main))
        self.assertIn('--execute-nonofficial-finalpath-acceptance',inspect.getsource(worker.main))
    def test_dual_has_no_force_cleanup(self):
        import inspect
        self.assertNotIn('unlink(original)',inspect.getsource(dual.stage_available))
    def test_plan_factory_expects_explicit_mounts(self):
        self.assertEqual(len(factory.INPUT_KEYS),6)

if __name__=='__main__':unittest.main(verbosity=2)
