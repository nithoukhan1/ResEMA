#!/usr/bin/env python3
"""R9B R3 CPU-only negative/positive tests. Absolutely no GPU or model imports."""
from pathlib import Path
import hashlib, importlib.util, json, tempfile, types, shutil

ROOT=Path(__file__).resolve().parent

def mod(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
W=mod('R9B_R3_NATIVE_WORKER_INTEGRATION')
P=mod('R9B_R3_PROVENANCE_ENVELOPE')
results=[]

def assert_true(cond):
    if not cond:raise ValueError('ASSERT_FALSE')

def check(name, fn, *, reject=False):
    try:
        v=fn()
        if reject:raise AssertionError('EXPECTED_REJECTION_NOT_OBSERVED')
        results.append((name,'PASS'))
    except Exception as e:
        if reject and not isinstance(e,AssertionError):results.append((name,'PASS_REJECTED'))
        else:raise AssertionError(name) from e

def base_spec():
    d={n:'/kaggle/working/fixture' for n in W.REQUIRED}
    d.update(resume_archive=None,resume_sidecar=None,resume_external_attestation=None,arm='MV00',physical_gpu='0',mode='FRESH_FROM_EARLY',run_id='R9TESTSAFE',
      target_completed_epoch=1,resume_stage=None,working_root='/kaggle/working/R9_TEST_SAFE',
      source_root='/kaggle/working/source',modules_root='/kaggle/working/modules',
      source_manifest_path='/kaggle/working/R9_CANDIDATE_MANIFEST_R1.json',
      expected_source_archive_sha256='c9aa976bff0710a2581b619008b7056ab398f498201cb857e86ed96e0e9ae734')
    return d

def modify(**kw):
    p=base_spec();p.update(kw);return p

check('fresh_plan',lambda:W.verify_plan(base_spec(),file_check=False))
check('resume_plan',lambda:W.verify_plan(modify(mode='RESUME_FROM_OUR_OWN_RUN',resume_stage='/kaggle/working/restore',resume_archive='/kaggle/working/restore.zip',resume_sidecar='/kaggle/working/restore.zip.sha256',resume_external_attestation='/kaggle/working/attest.json',target_completed_epoch=2),file_check=False))
check('mv01_arm',lambda:W.verify_plan(modify(arm='MV01',physical_gpu='1'),file_check=False))
for name,changes in [
  ('wrong_physical_gpu',{'physical_gpu':'1'}),('wrong_mode',{'mode':'NORMAL_RESUME'}),
  ('no_run_id',{'run_id':'not/valid'}),('target_zero',{'target_completed_epoch':0}),
  ('target_100',{'target_completed_epoch':100}),('target_boolean',{'target_completed_epoch':True}),
  ('wrong_source_pin',{'expected_source_archive_sha256':'0'*64}),
  ('bad_workroot',{'working_root':'/kaggle/input/do_not_write'}),
  ('missing_resume_stage',{'mode':'RESUME_FROM_OUR_OWN_RUN'}),
  ('illegal_fresh_stage',{'resume_stage':'/kaggle/working/fake'}),
  ('invalid_arm',{'arm':'MV02'}),
  ('test_path',{'publication_root':'/kaggle/working/test/result'}),
]:check(name,lambda c=changes:W.verify_plan(modify(**c),file_check=False),reject=True)
q=base_spec();q['unexpected']='x'
check('extra_plan_key_rejected',lambda:W.verify_plan(q,file_check=False),reject=True)
settings=W.native_settings(base_spec())
check('native_sgd_batch8_amp',lambda:assert_true(settings['batch']==8 and settings['amp'] and settings['optimizer']=='SGD'))
check('native_no_ddp',lambda:assert_true(settings['device']=='0' and settings['resume'] is False and settings['workers']==0))
check('native_epoch100_configuration_not_execution',lambda:assert_true(settings['epochs']==100))
class EarlyPairedTrainer:pass
class VGRADetectionTrainer:pass
class G4IntegratedEpochMixin(EarlyPairedTrainer):pass

def binder(native):
    return type(native.__name__,(G4IntegratedEpochMixin,),{})

def assemble(native,bound):
    cls=bound(native)
    M=type('R9BPostPublicationSessionStopMixin',(),{})
    return type(native.__name__,(M,cls),{})
check('class_composition',lambda:W.compose_native_types(EarlyPairedTrainer,binder,assemble))
check('class_other_reject',lambda:W.compose_native_types(object,binder,assemble),reject=True)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);g=root/'manifest.json'
    obj={'schema':'CHAT21_G4C_EPOCH_BUNDLE_CANDIDATE_R2_WITH_AUTHENTIC_RECEIPT',
         'arm':'MV00','completed_epoch':0,'terminal':False,'b_test_access':'NONE',
         'exported_off_session':False,'official_training_authorized':False}
    g.write_text(json.dumps(obj))
    pin=P.digest(g)
    args=dict(manifest_path=g,arm='MV00',epoch=0,pin=pin,
         recipe_sha256=W.SOURCE_RECIPE_SHA,
         source_archive_sha256='c9aa976bff0710a2581b619008b7056ab398f498201cb857e86ed96e0e9ae734',
         run_id='R9TESTSAFE')
    out=P.envelope(**args)
    check('candidate_provenance',lambda:assert_true(out['official_training_authorized'] is False))
    check('no_retroactive_g4_auth',lambda:assert_true(json.loads(g.read_text())['official_training_authorized'] is False))
    check('provenance_wrong_arm',lambda:P.envelope(**{**args,'arm':'MV01'}),reject=True)
    check('provenance_wrong_epoch',lambda:P.envelope(**{**args,'epoch':1}),reject=True)
    check('provenance_wrong_pin',lambda:P.envelope(**{**args,'pin':'0'*64}),reject=True)
    check('provenance_wrong_recipe',lambda:P.envelope(**{**args,'recipe_sha256':'0'*64}),reject=True)
    check('provenance_wrong_source',lambda:P.envelope(**{**args,'source_archive_sha256':'0'*64}),reject=True)
    obj['official_training_authorized']=True;g.write_text(json.dumps(obj))
    check('tampered_g4_rejected',lambda:P.envelope(**{**args,'pin':P.digest(g)}),reject=True)

# Ensure the actual source manifest is integrity pinned, without reading the whole
# 336-file tree here (the R9A parent source audit already did that separately).
m=ROOT/'R9_CANDIDATE_MANIFEST_R1.json'
check('source_manifest_pin',lambda:assert_true(W.file_sha(m)=='e1abea8ae36f15d4e2589e4d267395acf7921bad347be72697dac114c47df53b'))
check('no_gpu_or_train_entrypoint',lambda:assert_true('trainer.train()' not in (ROOT/'R9B_R3_NATIVE_WORKER_INTEGRATION.py').read_text()))
report={'schema':'CHAT23_R9B_R3_CPU_INTEGRATION_TEST','pass_count':len(results),
  'failed':0,'tests':results,'gpu_used':False,'model_loaded':False,
  'training_performed':False,'official_training_authorized':False,'b_test_access':'NONE'}
print(json.dumps(report,indent=2))
