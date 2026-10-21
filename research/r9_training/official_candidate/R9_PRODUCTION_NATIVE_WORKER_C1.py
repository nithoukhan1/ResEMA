#!/usr/bin/env python3
"""One physical-T4 100-epoch native worker, pre-release integration candidate.

No GPU execution without an independently frozen external release SHA, matching
both-arm user approval, and explicit --execute-approved-final-run. Keeps all
historical frozen/G4 false-authority fields intact; new run authority is attached
separately in its own provenance report. This file is NOT GPU-validated.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import R9_OFFICIAL_NATIVE_SESSION_R6 as r6
from R9_OFFICIAL_RELEASE_IDENTITY_R8 import validate_checkout, pin_shape
from R9_PRODUCTION_APPROVAL_GATE_C1 import (validate as approval_validate,
                                            validate_acceptance as acceptance_validate)


def execute_one(spec, r6_release, approval, pinned_commit, other_spec, *, run_class):
    arm = spec['arm']
    r6.validate_spec(spec, r6_release)
    r6.validate_spec(other_spec, r6_release)
    specs = {arm: spec, other_spec['arm']: other_spec}
    r6.need(run_class in ('OFFICIAL_100EPOCH','NONOFFICIAL_FINALPATH_ACCEPTANCE'),
            'RUN_CLASS')
    if run_class == 'OFFICIAL_100EPOCH':
        approval_validate(specs, approval, pinned_commit, execute=True)
    else:
        acceptance_validate(specs, approval, pinned_commit)

    # Ensure science and release actually originate from pinned checkout.
    source = Path(spec['source_root']).resolve(strict=True)
    checkout = source.parents[2]
    release_dir = checkout/'research/r9_training/official_candidate'
    code = checkout/'research/r9_training/code'
    r6.need(release_dir.is_dir() and code.is_dir() and not code.is_symlink(),
            'PINNED_RELEASE_LAYOUT')
    h = subprocess.run(['git','-C',str(checkout),'rev-parse','HEAD'],
                       capture_output=True,text=True,check=False)
    r6.need(h.returncode == 0 and h.stdout.strip() == pinned_commit,
            'IMMUTABLE_GIT_CHECKOUT')
    identity = validate_checkout(checkout, pinned_commit)
    r6.need(identity['status'] ==
        'R8_EXACT_CHECKOUT_SHA_AND_ADDITIVE_DIFF_VERIFIED_NOT_REMOTE_PROOF',
        'SOURCE_ADD_ONLY_GIT_IDENTITY')
    for filename in ('R9B_R3_NATIVE_WORKER_INTEGRATION.py',
                     'R9_OFFSESSION_TRANSPORT_R1.py'):
        r6.need((code/filename).is_file(), 'MISSING_ORIGINAL_R9_'+filename)
    sys.path.insert(0,str(code))
    import R9B_R3_NATIVE_WORKER_INTEGRATION as r3
    r6.native_preflight(spec,r3)
    r6.need(os.environ.get('CUDA_VISIBLE_DEVICES') == r6.ARMS[arm], 'PHYSICAL_T4_MAPPING')
    r6.need(all(os.environ.get(k) is None for k in
                ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT')),
            'DDP_FORBIDDEN')
    stage = r6.verify_offsite_resume(spec,r3) if spec['mode'] == 'RESUME_FROM_OUR_OWN_RUN' else None
    root = Path(spec['working_root'])
    r6.need(root.parent.is_dir() and not root.exists(), 'FRESH_SESSION_ROOT')
    root.mkdir(mode=0o700)
    for k in ('publication_root','checkpoint_root','guard_failure_root'):
        q=Path(spec[k]);r6.need(q.parent == root and not q.exists(), 'NEW_CHILD_'+k)
        q.mkdir(mode=0o700)
    import torch
    r6.need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            'T4' in torch.cuda.get_device_name(0).upper(), 'SINGLE_PHYSICAL_T4')
    r6.need('ultralytics' not in sys.modules, 'AMBIENT_ULTRALYTICS_FORBIDDEN')
    mods = Path(spec['modules_root']).resolve(strict=True)
    sys.path.insert(0,str(mods));sys.path.insert(0,str(source))
    os.chdir(source)
    import ultralytics
    r6.need(Path(ultralytics.__file__).resolve().is_relative_to(source),
            'FROZEN_ULTRALYTICS_IMPORT')
    from ultralytics.models.yolo.detect.early_paired_trainer import EarlyPairedTrainer
    from ultralytics.models.yolo.detect.vgra_trainer import VGRADetectionTrainer
    from g4c_guarded_trainer_class_binding_candidate_R4 import bind_candidate_class
    from g4_epoch_stream_publisher_candidate_R3 import verify_epoch
    from R9_OFFICIAL_NATIVE_WIRING_R5 import compose_native_candidate
    from R9_OFFICIAL_ADAPTIVE_BOUNDARY_R4 import (
        R9OfficialAdaptiveBoundaryMixin, R9OfficialVerifiedPostG4Stop,
        verify_terminal_after_train)
    native = {'MV00':EarlyPairedTrainer,'MV01':VGRADetectionTrainer}[arm]
    Trainer = compose_native_candidate(native,bind_candidate_class,
                                        R9OfficialAdaptiveBoundaryMixin)
    trainer = r6.bind_native(spec,Trainer,verify_epoch,r3,stage)
    begin = time.monotonic()
    stopped_at_verified_nonterminal = False
    try:
        trainer.train()
    except R9OfficialVerifiedPostG4Stop:
        stopped_at_verified_nonterminal = True
    records = getattr(trainer,'_g4_completed',None)
    r6.need(type(records) is list and records, 'AT_LEAST_ONE_COMPLETED_G4')
    rec=records[-1]
    completed = rec['completed_epoch']
    r6.need(type(completed) is int and
            spec['start_epoch_one_based']-1 <= completed <= spec['target_epoch_one_based']-1,
            'NO_CURSOR_DRIFT')
    if stopped_at_verified_nonterminal:
        r6.need(completed < 99 and rec['terminal'] is False, 'NONTERMINAL_BOUNDARY')
        chk = verify_epoch(root=spec['publication_root'],arm=arm,epoch=completed,
                           external_manifest_sha256=rec['manifest_sha256'])
        r6.need(chk['status']=='G4C_LOCAL_VERIFICATION_PASS_ONLY' and not chk['terminal'],
                'POST_BOUNDARY_G4')
    else:
        r6.need(spec['target_epoch_one_based']==100 and completed==99,
                'NORMAL_RETURN_ONLY_ALLOWED_AT_EPOCH100')
        verify_terminal_after_train(trainer=trainer,verify_epoch_fn=verify_epoch,arm=arm)
    from argparse import Namespace
    from R9_OFFSESSION_TRANSPORT_R1 import export,inspected_archive
    original=root/f'{arm}_epoch{completed+1:03d}_G4_LOCAL_EXPORT.zip'
    export(Namespace(publication_root=spec['publication_root'],arm=arm,
        completed_epoch=completed,manifest_sha256=rec['manifest_sha256'],
        destination=str(original)))
    inspector=inspected_archive(original,sidecar=Path(str(original)+'.sha256'))
    r6.need(inspector['status']=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION' and
            inspector['arm']==arm and inspector['completed_epoch']==completed and
            inspector['terminal']==(not stopped_at_verified_nonterminal),
            'ORIGINAL_LOCAL_TRANSPORT_VERIFICATION')
    result={'schema':'DETECTION_CHAT25_FINAL_NATIVE_SESSION_PROVENANCE_C1',
        'status':'NATIVE_GPU_SESSION_LOCAL_G4_VALIDATED_OFFSITE_UNVERIFIED',
        'arm':arm,'run_id':spec['run_id'],'session_start_epoch_one_based':spec['start_epoch_one_based'],
        'completed_epoch_zero_based':completed,
        'next_epoch_one_based':None if completed==99 else completed+2,
        'terminal':not stopped_at_verified_nonterminal,
        'run_class':run_class,
        'official_run_authorization_record_matched':run_class=='OFFICIAL_100EPOCH',
        'frozen_component_g4_authority_retained_original':True,
        'original_g4_zip':str(original), 'original_g4_zip_sha256':r3.file_sha(original),
        'published_release_commit':pinned_commit,
        'duration_seconds':round(time.monotonic()-begin,2),
        'offsite_persistence_verified':False,'b_test_access':'NONE'}
    report=root/'C1_FINAL_NATIVE_SESSION_REPORT.json'
    with report.open('x',encoding='utf-8') as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
        f.flush();os.fsync(f.fileno())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True)
    p.add_argument('--peer-spec',required=True)
    p.add_argument('--historical-r6-release',required=True)
    p.add_argument('--authorization')
    p.add_argument('--release-sha',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--execute-approved-final-run',action='store_true')
    mode.add_argument('--execute-nonofficial-finalpath-acceptance',action='store_true')
    a=p.parse_args()
    spec = r6._read_json(a.spec)
    peer = r6._read_json(a.peer_spec)
    frozen_release = r6._read_json(a.historical_r6_release)
    r6.validate_spec(spec,frozen_release)
    r6.validate_spec(peer,frozen_release)
    pin_shape(a.release_sha)
    if not (a.execute_approved_final_run or a.execute_nonofficial_finalpath_acceptance):
        print(json.dumps(approval_validate({spec['arm']:spec,peer['arm']:peer},None,
                                          a.release_sha,execute=False),indent=2))
        print('REVIEW ONLY: NO_GPU_NO_AUTHORIZATION_NO_BTEST');return
    r6.need(a.authorization is not None,'USER_APPROVAL_RECORD_REQUIRED')
    approval = r6._read_json(a.authorization)
    run_class='OFFICIAL_100EPOCH' if a.execute_approved_final_run else 'NONOFFICIAL_FINALPATH_ACCEPTANCE'
    print(json.dumps(execute_one(spec,frozen_release,approval,a.release_sha,peer,
                                 run_class=run_class),indent=2))

if __name__=='__main__':main()
