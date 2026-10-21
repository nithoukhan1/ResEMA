#!/usr/bin/env python3
"""CPU-only negative/positive gates; DOES NOT import CUDA or train."""
import copy,unittest
import R9_OFFICIAL_R7_NEWPATH_GPU_ACCEPTANCE as m
from TEST_R9_OFFICIAL_NATIVE_SESSION_R6 import spec,release

def plan():
    x=spec();x['target_epoch_one_based']=2
    return x

def consent(x=None):
    if x is None:x=plan()
    return {'schema':m.SCHEMA,'purpose':m.MAGIC,'arm':x['arm'],
       'run_id':x['run_id'],'source_commit':m.r6.BASE_COMMIT,
       'official_training_authorized':False,'max_completed_epoch_one_based':2,
       'b_test_access':'NONE'}

class Tests(unittest.TestCase):
    def ok(self,x=None,c=None,flag=True):
        x=plan() if x is None else x
        return m.validate_acceptance(x,release(),consent(x) if c is None else c,flag)
    def test_acceptance_exact_bounded(self):
        self.assertFalse(self.ok()['official_training_authorized'])
    def test_no_implicit_flag(self):
        with self.assertRaises(RuntimeError):self.ok(flag=False)
    def test_no_consent(self):
        with self.assertRaises(RuntimeError):self.ok(c={})
    def test_wrong_magic(self):
        c=consent();c['purpose']='yes'
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_wrong_arm(self):
        c=consent();c['arm']='MV01'
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_run_identity(self):
        c=consent();c['run_id']='another'
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_commit_identity(self):
        c=consent();c['source_commit']='0'*40
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_cannot_set_official(self):
        c=consent();c['official_training_authorized']=True
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_btest_stays_sealed(self):
        c=consent();c['b_test_access']='READ'
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_bound_epoch100(self):
        x=plan();x['target_epoch_one_based']=100
        with self.assertRaises(RuntimeError):self.ok(x=x)
    def test_bound_epoch5(self):
        x=plan();x['target_epoch_one_based']=5
        with self.assertRaises(RuntimeError):self.ok(x=x)
    def test_at_most_three_new_epochs(self):
        x=plan();x['target_epoch_one_based']=4
        c=consent(x);c['max_completed_epoch_one_based']=4
        with self.assertRaises(RuntimeError):self.ok(x,c)
    def test_max_ceiling_not_gt4(self):
        c=consent();c['max_completed_epoch_one_based']=5
        with self.assertRaises(RuntimeError):self.ok(c=c)
    def test_fresh_one_epoch(self):
        x=plan();x['target_epoch_one_based']=1
        self.assertEqual(self.ok(x)['target_epoch_one_based'],1)
    def test_resume_own_run_two(self):
        from TEST_R9_OFFICIAL_NATIVE_SESSION_R6 import resume
        x=resume();x['start_epoch_one_based']=2;x['target_epoch_one_based']=2
        c=consent(x)
        self.assertEqual(self.ok(x,c)['target_epoch_one_based'],2)
    def test_gpu_swap_rejected(self):
        x=plan();x['physical_gpu']='1'
        with self.assertRaises(RuntimeError):self.ok(x=x)
    def test_test_path_rejected(self):
        x=plan();x['runtime_yaml']='/kaggle/input/b-test/runtime.yaml'
        with self.assertRaises(RuntimeError):self.ok(x=x)
    def test_no_extra_field(self):
        c=consent();c['extra']=True
        with self.assertRaises(RuntimeError):self.ok(c=c)

if __name__=='__main__':unittest.main(verbosity=2)
