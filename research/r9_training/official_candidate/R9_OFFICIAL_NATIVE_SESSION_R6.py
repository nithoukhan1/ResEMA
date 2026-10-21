#!/usr/bin/env python3
"""R6 integrated native multi-epoch session candidate; GPU execution LOCKED.

This is a reviewable *implementation*, not an approved production release. The
CLI can only audit plans; the native entrypoint requires a separately frozen
and authorized release which does NOT exist in this bundle. No B-TEST path.
No frozen-source files, original G4 receipts or historical R1 flags are edited.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, re, shutil, sys, time
from pathlib import Path, PurePosixPath

BASE_COMMIT = 'b783eb3a470ab010785a8a24a90a666e9a3f2ed6'
SCHEMA = 'DETECTION_CHAT25_R6_PER_ARM_NATIVE_SESSION_CANDIDATE'
RELEASE_SCHEMA = 'DETECTION_CHAT25_R6_INTEGRATION_RELEASE_RECORD'
ARMS = {'MV00':'0','MV01':'1'}
KEYS = {'schema','arm','physical_gpu','run_id','mode','start_epoch_one_based',
        'target_epoch_one_based','monotonic_session_seconds','minimum_epoch_seconds',
        'reserve_export_seconds','source_root','modules_root','source_manifest_path',
        'recipe_path','runtime_yaml','assignment','visibility','pretrained_best',
        'pretrained_archive','config_path','working_root','publication_root',
        'checkpoint_root','guard_failure_root','resume_stage','resume_archive',
        'resume_sidecar','resume_external_attestation','release_record_path'}
HEX = re.compile(r'[0-9a-f]{64}\Z')
RUN = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')


def need(ok, code):
    if not ok:
        raise RuntimeError('R9_R6_NATIVE_FAIL_CLOSED_'+code)


def safe_path(s, *, prefix, allow_root=False):
    """Kaggle absolute-path semantics independent of Windows testing host."""
    need(type(s) is str and s and '\\' not in s and
         all(ord(c)>=32 and ord(c)!=127 for c in s), 'PATH_SYNTAX')
    p=PurePosixPath(s)
    need(p.is_absolute() and p.as_posix()==s and '..' not in p.parts,
         'PATH_NONCANONICAL')
    need(s.startswith(prefix.rstrip('/')+'/') or (allow_root and s==prefix),
         'PATH_LOCATION')
    need(not any(x.lower() in ('b-test','b_test','test') for x in p.parts),
         'TEST_PATH_FORBIDDEN')
    return p


def validate_spec(spec, release):
    need(type(release) is dict and release.get('schema')==RELEASE_SCHEMA,
         'RELEASE_SCHEMA')
    need(release.get('status')=='INTEGRATION_CANDIDATE_NOT_AUTHORIZED' and
         release.get('official_training_authorized') is False and
         release.get('run_executable') is False and
         release.get('b_test_access')=='NONE', 'RELEASE_NOT_LOCKED')
    need(release.get('frozen_science_commit')==BASE_COMMIT,'SOURCE_COMMIT')
    need(type(spec) is dict and set(spec)==KEYS and spec['schema']==SCHEMA,
         'SPEC_KEYS')
    arm=spec['arm']
    need(arm in ARMS and spec['physical_gpu']==ARMS[arm], 'ARM_GPU')
    need(type(spec['run_id']) is str and RUN.fullmatch(spec['run_id']), 'RUN_ID')
    need(spec['mode'] in ('FRESH_FROM_EARLY','RESUME_FROM_OUR_OWN_RUN'), 'MODE')
    st,tar=spec['start_epoch_one_based'],spec['target_epoch_one_based']
    need(type(st) is int and type(tar) is int and 1<=st<=tar<=100, 'CURSOR')
    if spec['mode']=='FRESH_FROM_EARLY':
        need(st==1 and all(spec[k] is None for k in ('resume_stage','resume_archive',
            'resume_sidecar','resume_external_attestation')), 'FRESH_INIT')
    else:
        need(st>=2 and all(type(spec[k]) is str for k in (
            'resume_stage','resume_archive','resume_sidecar','resume_external_attestation')),
            'RESUME_INPUTS')
        safe_path(spec['resume_archive'],prefix='/kaggle/input')
        safe_path(spec['resume_sidecar'],prefix='/kaggle/input')
        safe_path(spec['resume_external_attestation'],prefix='/kaggle/input')
        safe_path(spec['resume_stage'],prefix='/kaggle/working')
    for k in ('source_root','modules_root','source_manifest_path','recipe_path',
              'working_root','publication_root','checkpoint_root','guard_failure_root'):
        safe_path(spec[k],prefix='/kaggle/working')
    for k in ('runtime_yaml','assignment','visibility','pretrained_best',
              'pretrained_archive','config_path'):
        safe_path(spec[k],prefix='/kaggle/input')
    w=PurePosixPath(spec['working_root'])
    need(all(PurePosixPath(spec[k]).parent==w for k in (
        'publication_root','checkpoint_root','guard_failure_root')), 'WORK_DIR_STRUCTURE')
    safe_path(spec['release_record_path'],prefix='/kaggle/working')
    for k in ('monotonic_session_seconds','minimum_epoch_seconds','reserve_export_seconds'):
        x=spec[k]
        need(type(x) in (float,int) and math.isfinite(x) and x>0, 'TIME_'+k)
    need(spec['monotonic_session_seconds']>
         spec['minimum_epoch_seconds']+spec['reserve_export_seconds'],
         'SESSION_TOO_SHORT')
    return {'status':'PASS_STRICT_R6_STATIC_REVIEW_ONLY','arm':arm,
            'mode':spec['mode'],'start_epoch_one_based':st,
            'target_epoch_one_based':tar,'terminal_requested':tar==100,
            'official_training_authorized':False,'native_gpu_run':False,
            'external_persistence_verified':False,'b_test_access':'NONE'}


def require_future_approval(approval, release):
    """R6 deliberately cannot authorize its own GPU execution.

    Only a separately accepted next release can replace this gate with an
    independently reviewed approval/provenance contract. This avoids an
    untrusted user-editable JSON boolean accidentally starting official compute.
    """
    need(False,'R6_GPU_EXECUTION_NOT_RELEASED__REQUIRES_SEPARATE_R7_ACCEPTANCE')


def _read_json(path):
    p=Path(path)
    need(p.is_file() and not p.is_symlink(),'MISSING_JSON_FILE')
    def unique(pairs):
        d={}
        for k,v in pairs:
            need(k not in d,'JSON_DUPLICATE_KEY')
            d[k]=v
        return d
    return json.loads(p.read_text('utf-8'),object_pairs_hook=unique)


def native_preflight(spec, r3):
    """Run exact source/recipe + R0 original-pretrained pin before CUDA import."""
    need(r3.file_sha(spec['recipe_path'])==r3.SOURCE_RECIPE_SHA,'RECIPE_SHA')
    need(r3.file_sha(spec['runtime_yaml'])==r3.DATA_YAML_SHA,'DATA_YAML_SHA')
    r3.verify_science_source(spec['source_root'],spec['modules_root'],
                             spec['source_manifest_path'])
    r0_file=Path(spec['source_root'])/'ultralytics/models/yolo/detect/d4e_r3_read_only_identity_preflight.py'
    ms=importlib.util.spec_from_file_location('r6_r0_pinned_assets',r0_file)
    need(ms is not None and ms.loader is not None,'R0_SPEC')
    r0=importlib.util.module_from_spec(ms)
    ms.loader.exec_module(r0)
    checked=r0.verify_inputs(_read_json(spec['config_path']),spec['arm'],
        spec['run_id'],r3.DATA_YAML_SHA,spec['recipe_path'])
    need(checked.get('status')=='PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE',
         'R0_SOURCE_ASSET_PIN')
    need(r3.file_sha(spec['pretrained_best'])==
         checked['assets']['early_best_pt']['sha256'],'ORIGINAL_PRETRAINED_PIN')
    return checked


def verify_offsite_resume(spec, r3):
    """Never trust self-asserted offsite state as actual independent verification.

    Verifies authentic transported bytes and user-supplied separate receipts, but
    preservation still needs its own independent external verification milestone.
    """
    from R9_OFFSESSION_TRANSPORT_R1 import inspected_archive
    from R9_OFFSESSION_TRANSPORT_R1 import stage as original_stage
    from argparse import Namespace
    assert spec['mode']=='RESUME_FROM_OUR_OWN_RUN'
    imported=_read_json(spec['resume_external_attestation'])
    checked=inspected_archive(spec['resume_archive'],sidecar=Path(spec['resume_sidecar']))
    need(checked['status']=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION',
         'RESUME_TRANSPORT')
    need(checked['arm']==spec['arm'] and checked['terminal'] is False and
         checked['completed_epoch']==spec['start_epoch_one_based']-2,'RESUME_CURSOR')
    need(imported.get('schema')=='R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE' and
         imported.get('arm')==spec['arm'] and
         imported.get('zip_sha256')==checked['zip_sha256'],'OFFSITE_RECEIPT_MISMATCH')
    need(not Path(spec['resume_stage']).exists(),'EXISTING_STAGE')
    # Source G4 receipts retain their ORIGINAL historical authority fields.
    original_stage(Namespace(archive=spec['resume_archive'],sidecar=spec['resume_sidecar'],
        external_attestation=spec['resume_external_attestation'],
        destination=spec['resume_stage'],expected_arm=spec['arm'],
        expected_completed_epoch=checked['completed_epoch']))
    stage=_read_json(Path(spec['resume_stage'])/'R9_STAGE_RECEIPT.json')
    need(stage['next_epoch']==spec['start_epoch_one_based']-1 and
         stage['external_byte_verification_attested'] is True,'STAGED_CURSOR')
    capsule=_read_json(Path(stage['capsule_path']).with_suffix('.receipt.json'))
    need(capsule['identity']['run_id']==spec['run_id'] and
         capsule['identity']['arm']==spec['arm'] and
         capsule['identity']['recipe_sha256']==r3.SOURCE_RECIPE_SHA,
         'CAPSULE_OWN_RUN_IDENTITY')
    return stage


def _copy_authenticated(source,target,hash_fn):
    src,dst=Path(source),Path(target)
    need(src.is_file() and not src.is_symlink() and dst.parent.is_dir() and
         not dst.exists(),'RECOVERY_COPY_PRECONDITION')
    with src.open('rb') as r,dst.open('xb') as w:
        shutil.copyfileobj(r,w,1<<20)
        w.flush();os.fsync(w.fileno())
    need(hash_fn(src)==hash_fn(dst),'RECOVERY_BYTE_COPY_SHA')


def bind_native(spec, Trainer, verify_epoch, r3, stage=None):
    """Use native R7C 100-epoch settings and preserved G4/R4 hook order."""
    # Build exactly the R9 R3 native training settings (epochs=100, batch=8).
    train=Trainer(train_assignment_manifest=spec['assignment'],
        val_assignment_manifest=spec['assignment'],
        train_visibility_evidence=spec['visibility'],
        pretrained_archive=spec['pretrained_archive'],
        pretrained_best=spec['pretrained_best'],
        overrides=r3.native_settings(spec))
    kw={}
    if stage is not None:
        _copy_authenticated(stage['selected_best_path'],train.best,r3.file_sha)
        original=Path(stage['capsule_path'])
        for f in (original,original.with_suffix('.receipt.json')):
            _copy_authenticated(f,Path(spec['checkpoint_root'])/f.name,r3.file_sha)
        kw={'checkpoint_path':str(Path(spec['checkpoint_root'])/original.name),
            'trusted_receipt_sha256':stage['trusted_receipt_sha256']}
    train.d4e_configure_recovery(mode=spec['mode'],
        config_path=spec['config_path'],recipe_path=spec['recipe_path'],
        approved_runtime_yaml_sha256=r3.DATA_YAML_SHA,arm=spec['arm'],
        run_id=spec['run_id'],checkpoint_root=spec['checkpoint_root'],**kw)
    train.d4e_g3b5_bind_guard(arm=spec['arm'],failure_root=spec['guard_failure_root'])
    train.g4_bind_epoch_publication(output_root=spec['publication_root'])
    train.r9_official_bind_boundary(arm=spec['arm'],
        target_epoch_one_based=spec['target_epoch_one_based'],
        monotonic_deadline_seconds=time.monotonic()+spec['monotonic_session_seconds'],
        minimum_next_epoch_seconds=spec['minimum_epoch_seconds'],
        reserve_export_seconds=spec['reserve_export_seconds'],
        verify_epoch_fn=verify_epoch)
    return train


def execute_native_candidate(spec, release, approval):
    """Integrated path prepared for focused later GPU acceptance, R6 LOCKED.

    This callable is deliberately not exposed through this bundle's CLI. It has
    no successful R6 approval; use the later reviewed/frozen release instead.
    """
    validate_spec(spec,release)
    require_future_approval(approval,release)  # unconditional fail closed in R6
    # Everything below remains unexecuted until separately reviewed/frozen.
    import R9B_R3_NATIVE_WORKER_INTEGRATION as r3
    native_preflight(spec,r3)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==ARMS[spec['arm']], 'PHYSICAL_GPU')
    need(all(os.environ.get(k) is None for k in ('RANK','LOCAL_RANK','WORLD_SIZE')),
         'DDP_FORBIDDEN')
    stage=verify_offsite_resume(spec,r3) if spec['mode']=='RESUME_FROM_OUR_OWN_RUN' else None
    work=Path(spec['working_root'])
    need(work.parent.is_dir() and not work.exists(),'NEW_WORK_ROOT')
    work.mkdir(mode=0o700)
    for key in ('publication_root','checkpoint_root','guard_failure_root'):
        Path(spec[key]).mkdir(mode=0o700)
    import torch
    need(torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0).upper(),
         'ONE_PHYSICAL_T4')
    need('ultralytics' not in sys.modules,'ULTRALYTICS_IMPORT_PRELOADED')
    source=Path(spec['source_root']).resolve(strict=True)
    modules=Path(spec['modules_root']).resolve(strict=True)
    sys.path.insert(0,str(modules));sys.path.insert(0,str(source))
    os.chdir(source)
    import ultralytics
    need(Path(ultralytics.__file__).resolve().is_relative_to(source),
         'UNPINNED_ULTRALYTICS')
    from ultralytics.models.yolo.detect.early_paired_trainer import EarlyPairedTrainer
    from ultralytics.models.yolo.detect.vgra_trainer import VGRADetectionTrainer
    from g4c_guarded_trainer_class_binding_candidate_R4 import bind_candidate_class
    from g4_epoch_stream_publisher_candidate_R3 import verify_epoch
    from R9_OFFICIAL_NATIVE_WIRING_R5 import compose_native_candidate
    from R9_OFFICIAL_ADAPTIVE_BOUNDARY_R4 import (
        R9OfficialAdaptiveBoundaryMixin,R9OfficialVerifiedPostG4Stop,
        verify_terminal_after_train)
    native={'MV00':EarlyPairedTrainer,'MV01':VGRADetectionTrainer}[spec['arm']]
    Trainer=compose_native_candidate(native,bind_candidate_class,R9OfficialAdaptiveBoundaryMixin)
    trainer=bind_native(spec,Trainer,verify_epoch,r3,stage)
    nonterminal=False
    start=time.monotonic()
    try:
        trainer.train()
    except R9OfficialVerifiedPostG4Stop:
        nonterminal=True
    last=trainer._g4_completed[-1]
    completed=last['completed_epoch']
    need(completed>=spec['start_epoch_one_based']-1,'NO_NEW_EPOCH')
    if nonterminal:
        need(completed<99 and last['terminal'] is False, 'NONTERMINAL_CURSOR')
        checked=verify_epoch(root=spec['publication_root'],arm=spec['arm'],epoch=completed,
             external_manifest_sha256=last['manifest_sha256'])
        need(checked['status']=='G4C_LOCAL_VERIFICATION_PASS_ONLY','G4_BYTES')
    else:
        need(spec['target_epoch_one_based']==100 and completed==99,'UNEXPECTED_NATIVE_RETURN')
        verify_terminal_after_train(trainer=trainer,verify_epoch_fn=verify_epoch,arm=spec['arm'])
    from argparse import Namespace
    from R9_OFFSESSION_TRANSPORT_R1 import export,inspected_archive
    destination=work/(f"{spec['arm']}_epoch{completed+1:03d}_G4_LOCAL_EXPORT.zip")
    export(Namespace(publication_root=spec['publication_root'],arm=spec['arm'],
        completed_epoch=completed,manifest_sha256=last['manifest_sha256'],destination=str(destination)))
    archived=inspected_archive(destination,sidecar=Path(str(destination)+'.sha256'))
    need(archived['arm']==spec['arm'] and archived['terminal']==(not nonterminal) and
         archived['completed_epoch']==completed,'TRANSPORT_CURSOR')
    # Authentic G4/transport are local. Never assert offsite persistence here.
    record={'schema':'DETECTION_CHAT25_R6_LOCAL_NATIVE_SESSION_RESULT',
        'status':'GPU_EXECUTION_LOCAL_BYTES_ONLY_OFFSITE_PENDING',
        'arm':spec['arm'],'run_id':spec['run_id'],
        'completed_epoch_zero_based':completed,'terminal':not nonterminal,
        'local_zip':str(destination),'local_zip_sha256':r3.file_sha(destination),
        'duration_seconds':time.monotonic()-start,'b_test_access':'NONE',
        'independent_offsite_verified':False}
    with (work/'R6_NATIVE_SESSION_REPORT.json').open('x') as f:
        json.dump(record,f,indent=2,sort_keys=True)
        f.write('\n');f.flush();os.fsync(f.fileno())
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-spec',required=True)
    p.add_argument('--release-record',required=True)
    args=p.parse_args()
    spec,release=_read_json(args.review_spec),_read_json(args.release_record)
    need(Path(args.release_record).resolve()==Path(spec['release_record_path']).resolve(),
         'RELEASE_PATH_DRIFT')
    print(json.dumps(validate_spec(spec,release),sort_keys=True,indent=2))
    print('R6 REVIEW-ONLY: NO GPU EXECUTION / NO OFFICIAL AUTHORIZATION')

if __name__=='__main__':
    main()
