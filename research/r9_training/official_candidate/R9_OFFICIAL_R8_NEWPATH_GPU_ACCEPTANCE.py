#!/usr/bin/env python3
"""R8 release-pinned native NEW-PATH worker, NONOFFICIAL only.

A bounded acceptance implementation, not an official release. Default CLI
performs review only; actual T4 execution needs three independent conditions:
an exact acceptance-consent JSON, the explicit --run-nonofficial-newpath switch,
and strict targets of 1..4 epochs (no terminal epoch100). No user consent exists
in this artifact. Original R6 release and frozen G4 receipts stay nonofficial.

The official 100-epoch production release is STILL BLOCKED. This new worker
is meant only to generate evidence about R6-native interaction on new paths.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import R9_OFFICIAL_NATIVE_SESSION_R6 as r6
from R9_OFFICIAL_RELEASE_IDENTITY_R8 import validate_checkout, pin_shape

SCHEMA='DETECTION_CHAT25_R8_RELEASE_PINNED_NEWPATH_CONSENT_V1'
MAGIC='I_APPROVE_R8_RELEASE_PINNED_NONOFFICIAL_NEWPATH_GPU_ACCEPTANCE_ONLY'


def validate_acceptance(spec,release,consent,flag=False,pinned_release_sha=None):
    summary=r6.validate_spec(spec,release)
    r6.need(type(consent) is dict and set(consent)=={
        'schema','purpose','arm','run_id','source_commit','official_training_authorized',
        'max_completed_epoch_one_based','b_test_access','release_commit'},'ACCEPTANCE_CONSENT_KEYS')
    r6.need(flag is True and consent['schema']==SCHEMA and
        consent['purpose']==MAGIC,'EXPLICIT_NEWPATH_GPU_CONSENT')
    r6.need(consent['arm']==spec['arm'] and consent['run_id']==spec['run_id'] and
        consent['source_commit']==r6.BASE_COMMIT,'ACCEPTANCE_IDENTITY')
    pin_shape(pinned_release_sha)
    r6.need(consent['release_commit']==pinned_release_sha,'ACCEPTANCE_EXACT_RELEASE_PIN')
    r6.need(consent['official_training_authorized'] is False and
        consent['b_test_access']=='NONE','ACCEPTANCE_NOT_OFFICIAL')
    r6.need(type(consent['max_completed_epoch_one_based']) is int and
        1<=consent['max_completed_epoch_one_based']<=4 and
        spec['target_epoch_one_based'] <= consent['max_completed_epoch_one_based'] and
        spec['target_epoch_one_based']<100,'ACCEPTANCE_TARGET_BOUNDED')
    r6.need(spec['target_epoch_one_based']-spec['start_epoch_one_based']+1<=3,
            'ACCEPTANCE_MAX_3_NEW_EPOCHS')
    return {'status':'R8_NONOFFICIAL_RELEASE_PINNED_ACCEPTANCE_CONSENT_ACCEPTED_NOT_GPU_PROOF',
            'arm':spec['arm'],'target_epoch_one_based':spec['target_epoch_one_based'],
            'official_training_authorized':False,'b_test_access':'NONE'}


def execute_bounded_newpath(spec,release,consent,pinned_release_sha):
    """Only after explicit consent, verified source and physical-GPU checks."""
    validate_acceptance(spec,release,consent,True,pinned_release_sha)
    # Bootstrap the historical native modules from the *fixed checked-out Git
    # source*, not arbitrary PYTHONPATH or an ambient installed package.
    import subprocess
    source_root=Path(spec['source_root']).resolve(strict=True)
    checkout=source_root.parents[2]
    code=checkout/'research/r9_training/code'
    r6.need(code.is_dir() and not code.is_symlink(),'MISSING_PINNED_CODE_ROOT')
    git_head=subprocess.run(['git','-C',str(checkout),'rev-parse','HEAD'],
        capture_output=True,text=True,check=False)
    r6.need(git_head.returncode==0 and git_head.stdout.strip()==pinned_release_sha,
            'EXACT_PINNED_RELEASE_HEAD')
    release_identity=validate_checkout(checkout,pinned_release_sha)
    r6.need(release_identity['status']=='R8_EXACT_CHECKOUT_SHA_AND_ADDITIVE_DIFF_VERIFIED_NOT_REMOTE_PROOF',
            'PINNED_RELEASE_IDENTITY')
    r6.need((code/'R9B_R3_NATIVE_WORKER_INTEGRATION.py').is_file() and
            (code/'R9_OFFSESSION_TRANSPORT_R1.py').is_file(),
            'PINNED_NATIVE_CODE_FILES')
    sys.path.insert(0,str(code))
    import R9B_R3_NATIVE_WORKER_INTEGRATION as r3
    r6.native_preflight(spec,r3)
    r6.need(os.environ.get('CUDA_VISIBLE_DEVICES')==r6.ARMS[spec['arm']],
            'PHYSICAL_GPU')
    r6.need(all(os.environ.get(k) is None for k in
        ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT')),
        'NO_DDP')
    stage = r6.verify_offsite_resume(spec,r3) if spec['mode']=='RESUME_FROM_OUR_OWN_RUN' else None
    root=Path(spec['working_root'])
    r6.need(root.parent.is_dir() and not root.exists(),'UNIQUE_WORK_ROOT')
    root.mkdir(mode=0o700)
    for k in ('publication_root','checkpoint_root','guard_failure_root'):
        q=Path(spec[k]);r6.need(q.parent==root and not q.exists(),'NEW_CHILD_ROOT_'+k)
        q.mkdir(mode=0o700)
    import torch
    r6.need(torch.cuda.is_available() and torch.cuda.device_count()==1 and
            'T4' in torch.cuda.get_device_name(0).upper(),'SINGLE_T4')
    r6.need('ultralytics' not in sys.modules,'PRIOR_ULTRALYTICS_IMPORT')
    source=Path(spec['source_root']).resolve(strict=True)
    modules=Path(spec['modules_root']).resolve(strict=True)
    sys.path.insert(0,str(modules));sys.path.insert(0,str(source))
    os.chdir(source)
    import ultralytics
    r6.need(Path(ultralytics.__file__).resolve().is_relative_to(source),
            'FROZEN_ULTRALYTICS_ONLY')
    from ultralytics.models.yolo.detect.early_paired_trainer import EarlyPairedTrainer
    from ultralytics.models.yolo.detect.vgra_trainer import VGRADetectionTrainer
    from g4c_guarded_trainer_class_binding_candidate_R4 import bind_candidate_class
    from g4_epoch_stream_publisher_candidate_R3 import verify_epoch
    from R9_OFFICIAL_NATIVE_WIRING_R5 import compose_native_candidate
    from R9_OFFICIAL_ADAPTIVE_BOUNDARY_R4 import (
        R9OfficialAdaptiveBoundaryMixin,R9OfficialVerifiedPostG4Stop)
    native={'MV00':EarlyPairedTrainer,'MV01':VGRADetectionTrainer}[spec['arm']]
    Trainer=compose_native_candidate(native,bind_candidate_class,
        R9OfficialAdaptiveBoundaryMixin)
    trainer=r6.bind_native(spec,Trainer,verify_epoch,r3,stage)
    began=time.monotonic()
    try:
        trainer.train()
    except R9OfficialVerifiedPostG4Stop:
        pass
    else:
        raise RuntimeError('R8_ACCEPTANCE_EXPECTS_AUTHENTIC_NONTERMINAL_BOUNDARY')
    r6.need(bool(getattr(trainer,'_g4_completed',None)),'NO_LOCAL_G4')
    e=trainer._g4_completed[-1]
    epoch=e['completed_epoch']
    r6.need(type(epoch) is int and epoch < 99 and e['terminal'] is False and
        epoch+1<=consent['max_completed_epoch_one_based'] and
        epoch >= spec['start_epoch_one_based']-1,'EPOCH_CURSOR')
    checked=verify_epoch(root=spec['publication_root'],arm=spec['arm'],
        epoch=epoch,external_manifest_sha256=e['manifest_sha256'])
    r6.need(checked['status']=='G4C_LOCAL_VERIFICATION_PASS_ONLY' and
        checked['terminal'] is False,'LOCAL_G4_AUTHENTIC')
    from argparse import Namespace
    from R9_OFFSESSION_TRANSPORT_R1 import export,inspected_archive
    original=root/f"{spec['arm']}_epoch{epoch+1:03d}_G4_LOCAL_EXPORT.zip"
    export(Namespace(publication_root=spec['publication_root'],arm=spec['arm'],
        completed_epoch=epoch,manifest_sha256=e['manifest_sha256'],destination=str(original)))
    transport=inspected_archive(original,sidecar=Path(str(original)+'.sha256'))
    r6.need(transport['arm']==spec['arm'] and transport['terminal'] is False and
        transport['completed_epoch']==epoch,'TRANSPORT_RESULT')
    summary={'schema':'DETECTION_CHAT25_R8_RELEASE_PINNED_NONOFFICIAL_NATIVE_GPU_RESULT',
        'status':'NONOFFICIAL_NEWPATH_GPU_LOCAL_G4_VERIFIED_EXTERNAL_NOT_VERIFIED',
        'arm':spec['arm'],'run_id':spec['run_id'],
        'completed_epoch_zero_based':epoch,'terminal':False,
        'local_zip':str(original),'local_zip_sha256':r3.file_sha(original),
        'duration_seconds':round(time.monotonic()-began,3),
        'published_release_commit':pinned_release_sha,
        'official_training_authorized':False,
        'independent_offsite_verified':False,'b_test_access':'NONE'}
    with (root/'R8_NEWPATH_ACCEPTANCE_REPORT.json').open('x') as f:
        json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
        f.flush();os.fsync(f.fileno())
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--consent')
    p.add_argument('--run-nonofficial-newpath',action='store_true')
    p.add_argument('--pinned-release-commit',help='Exact 40-hex external Kaggle notebook release pin')
    args=p.parse_args()
    spec,release=r6._read_json(args.spec),r6._read_json(args.release)
    r6.need(Path(args.release).resolve()==Path(spec['release_record_path']).resolve(),
            'RELEASE_RECORD_PATH_DRIFT')
    if not args.run_nonofficial_newpath:
        print(json.dumps(r6.validate_spec(spec,release),indent=2,sort_keys=True))
        print('R8 REVIEW ONLY; GPU EXECUTION NOT INVOKED');return
    r6.need(args.consent is not None,'CONSENT_FILE_REQUIRED')
    consent=r6._read_json(args.consent)
    validate_acceptance(spec,release,consent,True,args.pinned_release_commit)
    print(json.dumps(execute_bounded_newpath(spec,release,consent,args.pinned_release_commit),indent=2,sort_keys=True))

if __name__=='__main__':main()
