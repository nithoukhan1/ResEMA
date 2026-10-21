#!/usr/bin/env python3
"""CPU-only, source-pinned dual-arm session-plan factory (no training).

Requires explicit Kaggle mounted asset paths and durable external receipts for
resume. Fresh initialization ALWAYS uses original approved Early pretrained.
"""
from __future__ import annotations
import argparse, json, os, re, importlib.util
from pathlib import Path
import R9_OFFICIAL_NATIVE_SESSION_R6 as r6
from R9_PRODUCTION_APPROVAL_GATE_C1 import validate as approval_validate

INPUT_KEYS={'runtime_yaml','assignment','visibility','pretrained_best',
            'pretrained_archive','config_path'}
REQUEST_KEYS={'run_ids','arms','monotonic_session_seconds',
              'minimum_epoch_seconds','reserve_export_seconds'}
PER_ARM_KEYS={'mode','start_epoch_one_based','target_epoch_one_based',
              'resume_archive','resume_sidecar','resume_external_attestation'}


def create(input_map, request, *, checkout, release_sha, output_dir):
    r6.need(type(input_map) is dict and set(input_map)==INPUT_KEYS,'INPUT_MAP_KEYS')
    r6.need(type(request) is dict and set(request)==REQUEST_KEYS,'REQUEST_KEYS')
    r6.need(type(request['run_ids']) is dict and set(request['run_ids'])==set(r6.ARMS),
            'RUN_IDS')
    r6.need(type(request['arms']) is dict and set(request['arms'])==set(r6.ARMS),
            'ARM_PLANS')
    root=Path(checkout).resolve(strict=True)
    r6.need(root.is_dir() and str(root).startswith('/kaggle/working/'),'CHECKOUT_ROOT')
    from R9_OFFICIAL_RELEASE_IDENTITY_R8 import validate_checkout
    validate_checkout(root,release_sha)
    rel=root/'research/r9_training'
    source=rel/'frozen_source'
    modules=rel/'integration_modules'
    manifest=rel/'code/R9_CANDIDATE_MANIFEST_R1.json'
    recipe=rel/'R9_RELEASE_RECIPE_CANDIDATE_R1.json'
    release=rel/'official_candidate/R9_OFFICIAL_R6_RELEASE_RECORD.json'
    r6.need(all(x.is_file() for x in (manifest,recipe,release)), 'FIXED_SOURCE_FILES')
    for k,p in input_map.items():
        r6.safe_path(p,prefix='/kaggle/input')
        r6.need(Path(p).is_file() and not Path(p).is_symlink(),'INPUT_MISSING_'+k)
    import sys
    code=str(rel/'code')
    if code not in sys.path:sys.path.insert(0,code)
    import R9B_R3_NATIVE_WORKER_INTEGRATION as r3
    r3.verify_science_source(source,modules,manifest)
    r6.need(r3.file_sha(recipe)==r3.SOURCE_RECIPE_SHA, 'RECIPE_BYTES')
    r6.need(r3.file_sha(input_map['runtime_yaml'])==r3.DATA_YAML_SHA,'YAML_BYTES')
    r0file=source/'ultralytics/models/yolo/detect/d4e_r3_read_only_identity_preflight.py'
    r0_spec=importlib.util.spec_from_file_location('r9_c1_r0',r0file)
    r6.need(r0_spec is not None and r0_spec.loader is not None,'R0_IMPORT')
    r0=importlib.util.module_from_spec(r0_spec);r0_spec.loader.exec_module(r0)
    r0paths=r6._read_json(input_map['config_path'])
    out=Path(output_dir)
    r6.need(str(out).startswith('/kaggle/working/') and out.parent.is_dir() and
            not out.exists(),'NEW_PLAN_ROOT')
    plans={}
    for arm,gpu in r6.ARMS.items():
        entry=request['arms'][arm]
        r6.need(type(entry) is dict and set(entry)==PER_ARM_KEYS,'ARM_PLAN_KEYS_'+arm)
        rid=request['run_ids'][arm]
        r6.need(type(rid) is str and re.fullmatch('[A-Za-z0-9_-]{1,128}',rid), 'RUN_ID')
        epoch=entry['start_epoch_one_based']
        work=f'/kaggle/working/{rid}_C1_START{epoch:03d}_SESSION'
        path={
          'schema':r6.SCHEMA,'arm':arm,'physical_gpu':gpu,'run_id':rid,
          'mode':entry['mode'],
          'start_epoch_one_based':epoch,
          'target_epoch_one_based':entry['target_epoch_one_based'],
          'monotonic_session_seconds':request['monotonic_session_seconds'],
          'minimum_epoch_seconds':request['minimum_epoch_seconds'],
          'reserve_export_seconds':request['reserve_export_seconds'],
          'source_root':str(source),'modules_root':str(modules),
          'source_manifest_path':str(manifest),'recipe_path':str(recipe),
          'runtime_yaml':input_map['runtime_yaml'],
          'assignment':input_map['assignment'],'visibility':input_map['visibility'],
          'pretrained_best':input_map['pretrained_best'],
          'pretrained_archive':input_map['pretrained_archive'],
          'config_path':input_map['config_path'],
          'working_root':work,'publication_root':work+'/publication',
          'checkpoint_root':work+'/capsules',
          'guard_failure_root':work+'/guard_failures',
          'resume_stage':None if entry['mode']=='FRESH_FROM_EARLY' else
                        work+'_INCOMING_OFFSITE_STAGE',
          'resume_archive':entry['resume_archive'],
          'resume_sidecar':entry['resume_sidecar'],
          'resume_external_attestation':entry['resume_external_attestation'],
          'release_record_path':str(release),
        }
        r6.validate_spec(path,r6._read_json(release))
        r6.need(not Path(work).exists(),'WORK_ALREADY_EXISTS')
        pins=r0.verify_inputs(r0paths,arm,rid,r3.DATA_YAML_SHA,str(recipe))
        r6.need(pins['status']=='PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE' and
                r3.file_sha(input_map['pretrained_best'])==pins['assets']['early_best_pt']['sha256'],
                'ORIGINAL_PRETRAINED_R0_'+arm)
        plans[arm]=path
    approval_validate(plans,None,release_sha,execute=False)
    out.mkdir(mode=0o700)
    for arm,p in plans.items():
        with (out/(arm+'.json')).open('x') as f:
            json.dump(p,f,sort_keys=True,indent=2);f.write('\n')
    result={'status':'TWO_ARM_PINNED_PLANS_CREATED_NO_GPU',
            'plans':{a:str(out/(a+'.json')) for a in plans},
            'published_release_commit':release_sha,
            'official_training_authorized':False,'b_test_access':'NONE'}
    (out/'PLAN_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-map',required=True)
    ap.add_argument('--session-request',required=True)
    ap.add_argument('--checkout',required=True)
    ap.add_argument('--release-sha',required=True)
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args()
    print(json.dumps(create(r6._read_json(a.input_map),r6._read_json(a.session_request),
        checkout=a.checkout,release_sha=a.release_sha,output_dir=a.output_dir),indent=2))

if __name__=='__main__':main()
