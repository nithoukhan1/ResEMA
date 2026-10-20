#!/usr/bin/env python3
"""R9B R3: native worker integration *candidate* for fresh/resume, NOT runnable training.

Separates import-free plan+lineage authentication from native integration method wiring.
The executable CLI has ONLY an offline --plan mode. The function prepare_native()
requires separately accepted/frozen artifacts and is not called from this CLI.
No GPU, model load, training, test access, or authorization claim.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil
from pathlib import Path

ARMS={'MV00':'0','MV01':'1'}
SOURCE_RECIPE_SHA='dd0ef5903dc3229abfe0c5e9298d0c288e3c682d2d14a8314530caf4ad3ad074'
DATA_YAML_SHA='f2cc6e31f5f1be152f4dd7603a397e142963923f1d35de613eb7c61b168bf0c0'
REQUIRED={'arm','physical_gpu','mode','run_id','target_completed_epoch','runtime_yaml','assignment',
          'visibility','pretrained_best','pretrained_archive','config_path','recipe_path',
          'working_root','publication_root','checkpoint_root','guard_failure_root','source_root',
          'modules_root','resume_stage','resume_archive','resume_sidecar','resume_external_attestation',
          'expected_source_archive_sha256','source_manifest_path'}


def require(x, msg):
    if not x:raise RuntimeError('R9B_R3_FAIL_CLOSED_'+msg)

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()

def read_json(p):
    path=Path(p)
    require(path.is_file() and not path.is_symlink(), 'JSON_NOT_REGULAR')
    return json.loads(path.read_text('utf-8'))


def verify_science_source(source_root, modules_root, manifest_path):
    """Authenticate all 336 source and 7 module bytes before native execution."""
    p=Path(manifest_path)
    require(p.is_file() and not p.is_symlink(),'MANIFEST_NOT_REGULAR')
    require(file_sha(p)=='e1abea8ae36f15d4e2589e4d267395acf7921bad347be72697dac114c47df53b',
            'R9A_MANIFEST_SHA')
    manifest=read_json(p)
    require(manifest.get('schema')=='CHAT23_R9A_GOVERNED_RELEASE_ENGINEERING_CANDIDATE','MANIFEST_SCHEMA')
    require(manifest.get('candidate_recipe_sha256')==SOURCE_RECIPE_SHA,'MANIFEST_RECIPE')
    require(manifest.get('parent_r8_source_archive_sha256')==
           'e9cccab3623bfca2f9251526eab14e1e4d513df44f7549cb9d80a046412c012d',
           'MANIFEST_R8_PARENT')
    src=Path(source_root);mods=Path(modules_root)
    for root,field,n in ((src,'candidate_source_sha256',336),
                         (mods,'unchanged_integration_modules_sha256',7)):
        mapping=manifest[field]
        require(len(mapping)==n,'FILE_COUNT')
        actual={f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()
                and '__pycache__' not in f.parts and f.suffix!='.pyc'}
        require(actual==set(mapping),'FILE_INVENTORY')
        for relative,pin in mapping.items():
            f=root/relative
            require(f.is_file() and not f.is_symlink(),'FILE_TYPE')
            require(file_sha(f)==pin,'FILE_HASH_'+relative)
    return {'source_files_verified':336,'integration_modules_verified':7}


def verify_plan(s, *, file_check=True):
    """Fail closed on incorrect arm, epoch, identity, recipe and resume linkage."""
    require(isinstance(s,dict) and set(s)==REQUIRED,'PLAN_KEYS')
    arm=s['arm'];mode=s['mode']
    require(arm in ARMS and s['physical_gpu']==ARMS[arm],'PHYSICAL_GPU_ARM')
    require(mode in ('FRESH_FROM_EARLY','RESUME_FROM_OUR_OWN_RUN'),'MODE')
    require(isinstance(s['run_id'],str) and re.fullmatch('[A-Za-z0-9_-]{1,128}',s['run_id']),'RUN_ID')
    target=s['target_completed_epoch']
    require(type(target) is int and 1<=target<=99,'BOUNDARY_EPOCH_NONTERMINAL')
    require(s['expected_source_archive_sha256']=='c9aa976bff0710a2581b619008b7056ab398f498201cb857e86ed96e0e9ae734','SOURCE_PARENT_PIN')
    require(isinstance(s['source_root'],str) and isinstance(s['modules_root'],str),'SOURCE_ROOTS')
    require(isinstance(s['source_manifest_path'],str),'SOURCE_MANIFEST')
    require(isinstance(s['working_root'],str) and s['working_root'].startswith('/kaggle/working/'),'PRIVATE_WORKING_ROOT')
    for key in ('working_root','publication_root','checkpoint_root','guard_failure_root'):
        require(isinstance(s[key],str) and '/test/' not in s[key].lower(), 'TEST_REFERENCE_'+key)
    for field in ('resume_stage','resume_archive','resume_sidecar','resume_external_attestation'):
        require(s[field] is None if mode=='FRESH_FROM_EARLY' else isinstance(s[field],str),
                'RESUME_INPUT_'+field)
    if file_check:
        for key in ('runtime_yaml','assignment','visibility','pretrained_best','pretrained_archive','config_path','recipe_path'):
            p=Path(s[key]);require(p.is_file() and not p.is_symlink(),'MISSING_'+key)
            require(all(part.lower() not in ('test','b-test','b_test') for part in p.parts),'BTEST_PATH_'+key)
        require(file_sha(s['runtime_yaml'])==DATA_YAML_SHA,'YAML_SHA')
        require(file_sha(s['recipe_path'])==SOURCE_RECIPE_SHA,'RECIPE_SHA')
        require(Path(s['source_root']).is_dir() and Path(s['modules_root']).is_dir(),'SOURCE_STAGE')
        verify_science_source(s['source_root'], s['modules_root'], s['source_manifest_path'])
    summary={'schema':'CHAT23_R9B_R3_CPU_PLAN_NOT_AUTHORIZED','arm':arm,'mode':mode,
             'run_id':s['run_id'],'target_completed_epoch':target,
             'parent_source_archive_sha256':s['expected_source_archive_sha256'],
             'recipe_sha256':SOURCE_RECIPE_SHA,'runtime_yaml_sha256':DATA_YAML_SHA,
             'offsite_export_verified':False,'native_training_performed':False,
             'official_training_authorized':False,'gpu_acceptance_authorized':False,'b_test_access':'NONE'}
    if mode=='RESUME_FROM_OUR_OWN_RUN':
        require(s['resume_stage'] is not None,'RESUME_REQUIRED')
        if file_check:
            receipt=verify_resume_stage(s)
            summary['resume_next_epoch']=receipt['next_epoch']
            summary['externally_attested_transport_sha256']=receipt['transport_archive_sha256']
    return summary


def verify_resume_stage(s):
    """Cross-check staged capsule with original transported bytes + external receipt."""
    root=Path(s['resume_stage']);receipt=read_json(root/'R9_STAGE_RECEIPT.json')
    require(receipt.get('schema')=='R9_NATIVE_RESTORE_BYTES_STAGE_NOT_GPU_RESUME','RESUME_SCHEMA')
    require(receipt.get('arm')==s['arm'] and receipt.get('official_training_authorized') is False,
            'RESUME_ARM_AUTHORITY')
    require(receipt.get('external_byte_verification_attested') is True,'EXTERNAL_ATTESTATION')
    require(type(receipt.get('completed_epoch')) is int and
            receipt['completed_epoch']+2<=s['target_completed_epoch'],'NO_NEW_EPOCH_AFTER_RESTORE')
    archive=Path(s['resume_archive']);sidecar=Path(s['resume_sidecar'])
    attestation=read_json(s['resume_external_attestation'])
    require(archive.is_file() and not archive.is_symlink() and
            sidecar.is_file() and not sidecar.is_symlink(),'OFFSITE_ARCHIVE_AND_SIDECAR')
    from R9_OFFSESSION_TRANSPORT_R1 import inspected_archive
    transport_check=inspected_archive(archive,sidecar=sidecar)
    require(transport_check.get('status')=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION' and
            transport_check.get('arm')==s['arm'] and
            transport_check.get('completed_epoch')==receipt['completed_epoch'] and
            transport_check.get('terminal') is False,
            'OFFSITE_ZIP_INVENTORY_OR_CONTENT')
    archive_sha=file_sha(archive)
    side=sidecar.read_text('utf-8').strip().split()
    require(len(side)==2 and side[0]==archive_sha and side[1]==archive.name,'ARCHIVE_SIDECAR')
    require(attestation.get('schema')=='R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE' and
            attestation.get('arm')==s['arm'] and
            attestation.get('completed_epoch')==receipt['completed_epoch'] and
            attestation.get('zip_sha256')==archive_sha and
            receipt.get('transport_archive_sha256')==archive_sha,'EXTERNAL_CONTEXT')
    manifest=read_json(root/'manifest.json')
    require(file_sha(root/'manifest.json')==attestation.get('g4_manifest_sha256')==transport_check['g4_manifest_sha256'],
            'STAGED_G4_MANIFEST_EXTERNAL_PIN')
    require(manifest.get('arm')==s['arm'] and
            manifest.get('completed_epoch')==receipt['completed_epoch'],'G4_RESUME_IDENTITY')
    require(manifest.get('terminal') is False and manifest.get('b_test_access')=='NONE',
            'TERMINAL_OR_TEST_RESUME_FORBIDDEN')
    require(receipt['trusted_receipt_sha256']==manifest.get('capsule_receipt_pin'),
            'TRUSTED_RECEIPT_DRIFT')
    require({p.name for p in root.iterdir()}==set(manifest['files'])|{
            'manifest.json','R9_STAGE_RECEIPT.json'},'STAGED_FILE_SET')
    best=Path(receipt['selected_best_path']);capsule=Path(receipt['capsule_path'])
    require(best.resolve().parent==root.resolve() and
            capsule.resolve().parent==root.resolve(),'RESUME_PATH_ESCAPE')
    require(file_sha(best)==manifest['files']['selected_best.pt']['sha256'],
            'RESUME_BEST_SHA')
    require(file_sha(capsule)==manifest['files'][capsule.name]['sha256'],
            'RESUME_CAPSULE_SHA')
    capsule_receipt_path=capsule.with_suffix('.receipt.json')
    capsule_receipt=read_json(capsule_receipt_path)
    require(capsule_receipt.get('identity',{}).get('run_id')==s['run_id'],
            'RESUME_RUN_ID')
    require(capsule_receipt.get('identity',{}).get('arm')==s['arm'],
            'RESUME_CAPSULE_ARM')
    require(capsule_receipt.get('identity',{}).get('recipe_sha256')==SOURCE_RECIPE_SHA,
            'RESUME_RECIPE_ID')
    require(capsule_receipt.get('cursor',{}).get('next_epoch')==receipt['next_epoch'],
            'RESUME_CURSOR_DRIFT')
    require(file_sha(capsule_receipt_path)==receipt['trusted_receipt_sha256'],
            'CAPSULE_RECEIPT_SHA')
    for name,data in manifest['files'].items():
        require(file_sha(root/name)==data['sha256'],'STAGED_FILE_HASH_'+name)
    return receipt


def native_settings(s):
    """Frozen settings derived from tested R7C native worker; no runnable CLI."""
    return {'data':s['runtime_yaml'],'task':'detect','imgsz':1024,'batch':8,'device':'0','seed':42,
        'deterministic':True,'epochs':100,'patience':100,'optimizer':'SGD','lr0':0.01,
        'lrf':0.01,'momentum':0.937,'weight_decay':0.0005,'nbs':64,'cos_lr':True,
        'amp':True,'workers':0,'cache':False,'compile':False,'rect':False,
        'multi_scale':0.0,'fraction':1.0,'single_cls':False,'mosaic':0.0,
        'mixup':0.0,'cutmix':0.0,'copy_paste':0.0,'plots':False,'save':True,
        'val':True,'resume':False,'exist_ok':False,'project':s['working_root'],
        'name':s['run_id']+'_'+s['arm']+'_SESSION'}


def compose_native_types(Native, bind_candidate_class, assemble_candidate):
    """Pure class wiring; requires the exact authenticated private native class."""
    expected={'EarlyPairedTrainer','VGRADetectionTrainer'}
    require(Native.__name__ in expected,'NATIVE_CLASS')
    G4=assemble_candidate(Native,bind_candidate_class)
    class AmpReadyMixin:
        def __init__(self,*a,**kw):
            self._r7c_amp_gate_ready=True
            super().__init__(*a,**kw)
    Final=type(Native.__name__,(AmpReadyMixin,G4),{'__module__':__name__})
    names=[x.__name__ for x in Final.__mro__]
    require(names.count('R9BPostPublicationSessionStopMixin')==1,'BOUNDARY_MRO')
    require(names.count('G4IntegratedEpochMixin')==1,'G4_MRO')
    require(names.index('R9BPostPublicationSessionStopMixin')<names.index('G4IntegratedEpochMixin'),'MRO_ORDER')
    return Final


def configure_authenticated_trainer(s, Trainer, verify_epoch):
    """Actual native construction order; NOT called by this package's CLI.

    This is an integration candidate subject to independent native acceptance.
    In particular the R0 config requires historical overlay pins to be present.
    """
    verify_plan(s)
    # The calling GPU worker MUST independently validate physical assignment and
    # pinned-source module import *before* calling this function.
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==s['physical_gpu'], 'PHYSICAL_GPU_ENV'
    for key in ('publication_root','checkpoint_root','guard_failure_root'):
        p=Path(s[key]);require(p.is_dir() and not p.is_symlink(),'PRIVATE_ROOT_'+key)
    trainer=Trainer(train_assignment_manifest=s['assignment'],val_assignment_manifest=s['assignment'],
        train_visibility_evidence=s['visibility'],pretrained_archive=s['pretrained_archive'],
        pretrained_best=s['pretrained_best'],overrides=native_settings(s))
    resume=s['mode']=='RESUME_FROM_OUR_OWN_RUN'
    kw={}
    if resume:
        stage=read_json(Path(s['resume_stage'])/'R9_STAGE_RECEIPT.json')
        original=Path(stage['selected_best_path'])
        dest=Path(trainer.best)
        require(not dest.exists() and not dest.is_symlink(),'BEST_DEST_ALREADY_EXISTS')
        with original.open('rb') as f, dest.open('xb') as g: shutil.copyfileobj(f,g)
        require(file_sha(dest)==file_sha(original),'BEST_COPY_SHA')
        # Recovery capsule must reside *inside* checkpoint_root; historical
        # source rejects a capsule outside the trusted recovery root. Copy the
        # verified bytes into a NEW private run root, never mutate the offsite stage.
        source_cap=Path(stage['capsule_path'])
        for src in (source_cap,source_cap.with_suffix('.receipt.json')):
            dst=Path(s['checkpoint_root'])/src.name
            require(not dst.exists() and not dst.is_symlink(),'CAPSULE_RESTORE_OVERWRITE')
            with src.open('rb') as f, dst.open('xb') as g: shutil.copyfileobj(f,g)
            require(file_sha(dst)==file_sha(src),'CAPSULE_RESTORE_SHA')
        kw={'checkpoint_path':str(Path(s['checkpoint_root'])/source_cap.name),
            'trusted_receipt_sha256':stage['trusted_receipt_sha256']}
    trainer.d4e_configure_recovery(mode=s['mode'], config_path=s['config_path'],
        recipe_path=s['recipe_path'],approved_runtime_yaml_sha256=DATA_YAML_SHA,
        arm=s['arm'],run_id=s['run_id'],checkpoint_root=s['checkpoint_root'],**kw)
    trainer.d4e_g3b5_bind_guard(arm=s['arm'],failure_root=s['guard_failure_root'])
    trainer.g4_bind_epoch_publication(output_root=s['publication_root'])
    trainer.r9b_bind_nonterminal_session_boundary(arm=s['arm'],
        target_completed_epoch=s['target_completed_epoch'],
        publication_root=s['publication_root'],verify_epoch_fn=verify_epoch)
    return trainer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-only',required=True,help='Strict JSON plan; CPU only')
    ap.add_argument('--report',required=True,help='New report JSON; no overwrite')
    a=ap.parse_args()
    spec=read_json(a.plan_only)
    result=verify_plan(spec,file_check=True)
    out=Path(a.report)
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'REPORT_EXISTS')
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))
    print('R9B_R3_CPU_PLAN_ONLY; NO GPU EXECUTION, NO TRAINING AUTHORIZATION')

if __name__=='__main__':main()
