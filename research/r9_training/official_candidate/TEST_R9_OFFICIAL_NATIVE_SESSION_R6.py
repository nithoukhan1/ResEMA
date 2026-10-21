#!/usr/bin/env python3
import copy,unittest
import R9_OFFICIAL_NATIVE_SESSION_R6 as m

ROOT='/kaggle/working/r9_github_checkout/research/r9_training'
RUN='DPR9OFFICIAL20261021_MV00'

def release():
    return {'schema':m.RELEASE_SCHEMA,'status':'INTEGRATION_CANDIDATE_NOT_AUTHORIZED',
            'frozen_science_commit':m.BASE_COMMIT,'official_training_authorized':False,
            'run_executable':False,'b_test_access':'NONE'}

def spec():
    w='/kaggle/working/'+RUN+'_SESSION01'
    return dict(schema=m.SCHEMA,arm='MV00',physical_gpu='0',run_id=RUN,
        mode='FRESH_FROM_EARLY',start_epoch_one_based=1,target_epoch_one_based=10,
        monotonic_session_seconds=36000.0, minimum_epoch_seconds=600.0,
        reserve_export_seconds=1200.0, source_root=ROOT+'/frozen_source',
        modules_root=ROOT+'/integration_modules',
        source_manifest_path=ROOT+'/code/R9_CANDIDATE_MANIFEST_R1.json',
        recipe_path=ROOT+'/R9_RELEASE_RECIPE_CANDIDATE_R1.json',
        runtime_yaml='/kaggle/input/approved/runtime.yaml',
        assignment='/kaggle/input/approved/pair_assignment.csv',
        visibility='/kaggle/input/approved/visibility.json',
        pretrained_best='/kaggle/input/approved/early_best.pt',
        pretrained_archive='/kaggle/input/approved/original_pretrain.zip',
        config_path='/kaggle/input/approved/r0_inputs.json',working_root=w,
        publication_root=w+'/publication',checkpoint_root=w+'/capsules',
        guard_failure_root=w+'/guard_failures',resume_stage=None,resume_archive=None,
        resume_sidecar=None,resume_external_attestation=None,
        release_record_path=ROOT+'/official_candidate/R9_OFFICIAL_R6_RELEASE_RECORD.json')

def resume():
    s=spec();s['mode']='RESUME_FROM_OUR_OWN_RUN';s['start_epoch_one_based']=5
    s['resume_stage']='/kaggle/working/verified_mv00_recovery_stage'
    s['resume_archive']='/kaggle/input/verified/mv00.zipblob'
    s['resume_sidecar']='/kaggle/input/verified/mv00.zip.sha256'
    s['resume_external_attestation']='/kaggle/input/verified/attestation.json'
    return s

class Tests(unittest.TestCase):
    def test_fresh(self):self.assertEqual(m.validate_spec(spec(),release())['status'],'PASS_STRICT_R6_STATIC_REVIEW_ONLY')
    def test_resume(self):self.assertEqual(m.validate_spec(resume(),release())['start_epoch_one_based'],5)
    def test_terminal_review(self):
        x=spec();x['target_epoch_one_based']=100
        self.assertTrue(m.validate_spec(x,release())['terminal_requested'])
    def test_frozen_science_commit(self):
        r=release();r['frozen_science_commit']='0'*40
        with self.assertRaises(RuntimeError):m.validate_spec(spec(),r)
    def test_authorization_flag_rejected(self):
        r=release();r['official_training_authorized']=True
        with self.assertRaises(RuntimeError):m.validate_spec(spec(),r)
    def test_run_executable_rejected(self):
        r=release();r['run_executable']=True
        with self.assertRaises(RuntimeError):m.validate_spec(spec(),r)
    def test_gpu_execution_always_locked(self):
        with self.assertRaisesRegex(RuntimeError,'R6_GPU_EXECUTION_NOT_RELEASED'):
            m.execute_native_candidate(spec(),release(),{})
    def test_gpu_swap(self):
        x=spec();x['physical_gpu']='1'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_start_at_epoch2_not_fresh(self):
        x=spec();x['start_epoch_one_based']=2
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_resume_stage_required(self):
        x=resume();x['resume_stage']=None
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_resume_kaggle_working_is_not_external(self):
        x=resume();x['resume_archive']='/kaggle/working/mv00.zip'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_btest_mount_rejected(self):
        x=spec();x['runtime_yaml']='/kaggle/input/b-test/val.yaml'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_win_drive_rejected(self):
        x=spec();x['pretrained_best']='C:\\some\\original.pt'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_double_slash_rejected(self):
        x=spec();x['runtime_yaml']='/kaggle//input/approved/runtime.yaml'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_parent_traversal_rejected(self):
        x=spec();x['visibility']='/kaggle/input/approved/../test/file'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_output_directory_escape(self):
        x=spec();x['checkpoint_root']='/kaggle/working/another_run/capsules'
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_zero_epoch_rejected(self):
        x=spec();x['target_epoch_one_based']=0
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_invalid_time_budget_rejected(self):
        x=spec();x['monotonic_session_seconds']=1200
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_nan_rejected(self):
        x=spec();x['minimum_epoch_seconds']=float('nan')
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())
    def test_extra_key_rejected(self):
        x=spec();x['ignore_btest']=True
        with self.assertRaises(RuntimeError):m.validate_spec(x,release())

if __name__=='__main__':unittest.main(verbosity=2)
