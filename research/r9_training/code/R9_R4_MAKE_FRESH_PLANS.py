#!/usr/bin/env python3
"""Create two fresh one-epoch NONOFFICIAL acceptance plans from explicit mounted paths.
No GPU, no training. Inputs must point to authentic GRAZPEDWRI Split-B assets.
"""
from __future__ import annotations
import argparse,json,re,importlib.util
from pathlib import Path
import R9B_R3_NATIVE_WORKER_INTEGRATION as r3
KEYS=('runtime_yaml','assignment','visibility','pretrained_best','pretrained_archive','config_path','stage_root')
def need(ok,msg):
    if not ok:raise RuntimeError('R9_R4_PLAN_FAIL_CLOSED_'+msg)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-map',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--run-prefix',required=True)
    a=p.parse_args(); mapping=json.loads(Path(a.input_map).read_text())
    need(set(mapping)==set(KEYS),'EXPECTED_INPUT_KEYS_'+','.join(KEYS))
    need(re.fullmatch('[A-Za-z0-9_-]{1,100}',a.run_prefix),'RUN_PREFIX')
    for k in KEYS:
        need(isinstance(mapping[k],str) and Path(mapping[k]).is_absolute(),'ABSOLUTE_PATH_'+k)
        need(Path(mapping[k]).is_file() if k!='stage_root' else Path(mapping[k]).is_dir(),
             'FILE_OR_STAGE_MISSING_'+k)
    stage=Path(mapping['stage_root']);out=Path(a.output_dir)
    need(out.parent.is_dir() and not out.exists(),'NEW_OUTPUT_DIRECTORY')
    # Verify frozen source and recipe before creating plans.
    r3.verify_science_source(stage/'source',stage/'modules',
                             Path(__file__).with_name('R9_CANDIDATE_MANIFEST_R1.json'))
    need(r3.file_sha(stage/'R9_RELEASE_RECIPE_CANDIDATE_R1.json')==r3.SOURCE_RECIPE_SHA,
         'FROZEN_RECIPE_SHA')
    # Read-only historical R0 input identity validation BEFORE GPU is started.
    r0_file=stage/'source'/'ultralytics/models/yolo/detect/d4e_r3_read_only_identity_preflight.py'
    module_spec=importlib.util.spec_from_file_location('r9r4_r0_read_only',str(r0_file))
    need(module_spec is not None and module_spec.loader is not None,'R0_PREFLIGHT_IMPORT')
    r0=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(r0)
    r0_paths=json.loads(Path(mapping['config_path']).read_text())
    plans={}
    for arm,gpu in r3.ARMS.items():
        name=a.run_prefix+'_'+arm
        work=f'/kaggle/working/{name}_R9R4'
        plan={'arm':arm,'physical_gpu':gpu,'mode':'FRESH_FROM_EARLY','run_id':name,
              'target_completed_epoch':1,
              'runtime_yaml':mapping['runtime_yaml'],'assignment':mapping['assignment'],
              'visibility':mapping['visibility'],'pretrained_best':mapping['pretrained_best'],
              'pretrained_archive':mapping['pretrained_archive'],'config_path':mapping['config_path'],
              'recipe_path':str(stage/'R9_RELEASE_RECIPE_CANDIDATE_R1.json'),
              'working_root':work,'publication_root':work+'/publication',
              'checkpoint_root':work+'/capsules','guard_failure_root':work+'/guard_failures',
              'source_root':str(stage/'source'),'modules_root':str(stage/'modules'),
              'resume_stage':None,'resume_archive':None,'resume_sidecar':None,
              'resume_external_attestation':None,
              'expected_source_archive_sha256':'c9aa976bff0710a2581b619008b7056ab398f498201cb857e86ed96e0e9ae734',
              'source_manifest_path':str(Path(__file__).with_name('R9_CANDIDATE_MANIFEST_R1.json'))}
        r3.verify_plan(plan,file_check=True)
        r0_record=r0.verify_inputs(r0_paths,arm,name,r3.DATA_YAML_SHA,
                                   str(stage/'R9_RELEASE_RECIPE_CANDIDATE_R1.json'))
        need(r0_record['status']=='PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE',
             'R0_AUTHENTIC_BYTES_'+arm)
        need(r3.file_sha(mapping['pretrained_best'])==
             r0_record['assets']['early_best_pt']['sha256'],'PRETRAINED_BEST_R0_PIN_'+arm)
        need(not Path(work).exists(),'WORK_OUTPUT_ALREADY_EXISTS_'+arm)
        plans[arm]=plan
    out.mkdir(mode=0o700)
    for arm,plan in plans.items():
        with (out/(arm+'.json')).open('x') as f:json.dump(plan,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'schema':'R9_R4_FRESH_PLAN_GENERATION_CPU','status':'PASS_TWICE_FULL_PINNED_PREFLIGHT_NO_GPU',
          'mv00':str(out/'MV00.json'),'mv01':str(out/'MV01.json'),
          'official_training_authorized':False,'training_performed':False,'b_test_access':'NONE'},indent=2))
if __name__=='__main__':main()
