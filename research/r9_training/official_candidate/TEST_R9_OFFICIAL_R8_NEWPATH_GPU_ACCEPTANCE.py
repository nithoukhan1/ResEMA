#!/usr/bin/env python3
"""CPU-only R8 release-pin and explicit consent tests; no GPU or checkpoint IO."""
import unittest
import R9_OFFICIAL_R8_NEWPATH_GPU_ACCEPTANCE as m
from TEST_R9_OFFICIAL_NATIVE_SESSION_R6 import spec,release

PIN='1'*40

def plan():
    x=spec();x['target_epoch_one_based']=2
    return x

def consent(x=None):
    if x is None:x=plan()
    return {'schema':m.SCHEMA,'purpose':m.MAGIC,'arm':x['arm'],
      'run_id':x['run_id'],'source_commit':m.r6.BASE_COMMIT,
      'release_commit':PIN,'official_training_authorized':False,
      'max_completed_epoch_one_based':2,'b_test_access':'NONE'}

class Tests(unittest.TestCase):
    def check(self,x=None,c=None,flag=True,pin=PIN):
        x=plan() if x is None else x
        return m.validate_acceptance(x,release(),consent(x) if c is None else c,flag,pin)
    def test_valid_nonofficial_pin(self):
        r=self.check();self.assertFalse(r['official_training_authorized'])
    def test_missing_pin(self):
        with self.assertRaises(RuntimeError):self.check(pin=None)
    def test_wrong_consent_pin(self):
        c=consent();c['release_commit']='2'*40
        with self.assertRaises(RuntimeError):self.check(c=c)
    def test_pre_science_commit_as_release(self):
        c=consent();c['release_commit']=m.r6.BASE_COMMIT
        with self.assertRaises(RuntimeError):self.check(c=c,pin=m.r6.BASE_COMMIT)
    def test_implicit_flag_rejected(self):
        with self.assertRaises(RuntimeError):self.check(flag=False)
    def test_absent_consent(self):
        with self.assertRaises(RuntimeError):self.check(c={})
    def test_official_flag_rejected(self):
        c=consent();c['official_training_authorized']=True
        with self.assertRaises(RuntimeError):self.check(c=c)
    def test_b_test_access_rejected(self):
        c=consent();c['b_test_access']='OPEN'
        with self.assertRaises(RuntimeError):self.check(c=c)
    def test_gpu_swap_rejected(self):
        x=plan();x['physical_gpu']='1'
        with self.assertRaises(RuntimeError):self.check(x=x)
    def test_btest_path_rejected(self):
        x=plan();x['runtime_yaml']='/kaggle/input/b-test/data.yaml'
        with self.assertRaises(RuntimeError):self.check(x=x)
    def test_epoch100_rejected(self):
        x=plan();x['target_epoch_one_based']=100
        with self.assertRaises(RuntimeError):self.check(x=x)
    def test_target_gt_consent_rejected(self):
        x=plan();x['target_epoch_one_based']=3
        with self.assertRaises(RuntimeError):self.check(x=x)
    def test_wrong_science_commit_rejected(self):
        c=consent();c['source_commit']='0'*40
        with self.assertRaises(RuntimeError):self.check(c=c)
    def test_unknown_consent_key_rejected(self):
        c=consent();c['extra']=1
        with self.assertRaises(RuntimeError):self.check(c=c)
    def test_wrong_run_id_rejected(self):
        c=consent();c['run_id']='other'
        with self.assertRaises(RuntimeError):self.check(c=c)
    def test_invalid_pin_format_rejected(self):
        with self.assertRaises(RuntimeError):self.check(pin='not-a-commit')

if __name__=='__main__':unittest.main(verbosity=2)
