#!/usr/bin/env python3
"""R9 R4: REAL native **nonofficial bounded acceptance** worker for one arm.

Run under a child process with exactly one physical T4 visible. The ONLY accepted
completion is an R9B post-G4 published boundary (not the terminal/epoch100 path).
Real GPU path has NOT been exercised by the author. Do not use for official runs.
"""
from __future__ import annotations
import argparse, json, os, sys, time, importlib.util
from pathlib import Path

import R9B_R3_NATIVE_WORKER_INTEGRATION as r3

def need(ok,msg):
    if not ok:raise RuntimeError('R9_R4_FAIL_CLOSED_'+msg)

def save_new(p,obj):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--accept-nonofficial-bounded-gpu',action='store_true')
    a=ap.parse_args()
    need(a.accept_nonofficial_bounded_gpu,'EXPLICIT_NONOFFICIAL_GPU_SWITCH_REQUIRED')
    s=r3.read_json(a.plan)
    before=os.environ.get('CUDA_VISIBLE_DEVICES')
    need(before == r3.ARMS.get(s.get('arm')),'PHYSICAL_GPU_MISMATCH')
    need(all(os.environ.get(k) is None for k in ('LOCAL_RANK','RANK','WORLD_SIZE')),'DDP_ENV_FORBIDDEN')
    summary=r3.verify_plan(s,file_check=True)
    need(summary['official_training_authorized'] is False,'AUTHORITY')
    need(s['target_completed_epoch'] <= 99,'TERMINAL_FORBIDDEN')
    mode=s['mode'];arm=s['arm']
    if mode=='FRESH_FROM_EARLY':
        need(s['target_completed_epoch']==1,'FRESH_GATE_ONE_EPOCH_ONLY')
    else:
        need(s['target_completed_epoch']==summary['resume_next_epoch']+1,
             'RESUME_GATE_EXACTLY_ONE_NEW_EPOCH')
    work=Path(s['working_root']);need(work.parent.is_dir() and not work.exists(),'FRESH_WORKING_DIR')
    work.mkdir(mode=0o700)
    for k in ('publication_root','checkpoint_root','guard_failure_root'):
        q=Path(s[k]);need(q.parent==work and not q.exists(),'UNIQUE_PRIVATE_CHILD_'+k)
        q.mkdir(mode=0o700)
    # Repeat the R0 historical source/asset/overlay verification in the child
    # before CUDA initialization. The stage builder has already pinned R9 source.
    import importlib.util
    r0_file=Path(s['source_root'])/'ultralytics/models/yolo/detect/d4e_r3_read_only_identity_preflight.py'
    modspec=importlib.util.spec_from_file_location('r9r4_r0_gate',str(r0_file))
    need(modspec is not None and modspec.loader is not None,'R0_IMPORT')
    r0=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(r0)
    r0_record=r0.verify_inputs(r3.read_json(s['config_path']),s['arm'],s['run_id'],
                               r3.DATA_YAML_SHA,s['recipe_path'])
    need(r0_record.get('status')=='PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE',
         'R0_PINNED_ASSET_CHECK')
    need(r3.file_sha(s['pretrained_best'])==r0_record['assets']['early_best_pt']['sha256'],
         'PRETRAINED_BEST_PIN')
    print('R9_R4_PLAN_VERIFIED='+json.dumps(summary,sort_keys=True),flush=True)
    import torch
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_VISIBLE_CUDA_DEVICE_REQUIRED')
    gpu_name=torch.cuda.get_device_name(0)
    need('T4' in gpu_name.upper(),'EXPECTED_T4')
    need('ultralytics' not in sys.modules,'ULTRALYTICS_PRELOADED')
    source=Path(s['source_root']).resolve(strict=True)
    mods=Path(s['modules_root']).resolve(strict=True)
    sys.path.insert(0,str(mods));sys.path.insert(0,str(source))
    os.chdir(source)
    import ultralytics
    need(Path(ultralytics.__file__).resolve().is_relative_to(source),'NONPINNED_ULTRALYTICS')
    from ultralytics.models.yolo.detect.early_paired_trainer import EarlyPairedTrainer
    from ultralytics.models.yolo.detect.vgra_trainer import VGRADetectionTrainer
    from g4c_guarded_trainer_class_binding_candidate_R4 import bind_candidate_class
    from g4_epoch_stream_publisher_candidate_R3 import verify_epoch
    from R9B_NATIVE_SESSION_BOUNDARY_R2 import assemble_candidate, R9BPublishedSessionBoundary
    Native={'MV00':EarlyPairedTrainer,'MV01':VGRADetectionTrainer}[arm]
    Trainer=r3.compose_native_types(Native,bind_candidate_class,assemble_candidate)
    trainer=r3.configure_authenticated_trainer(s,Trainer,verify_epoch)
    start=time.monotonic()
    try:
        trainer.train()
    except R9BPublishedSessionBoundary:
        need(getattr(trainer,'_r9b_session_stopped',False),'NOT_VERIFIED_R9_BOUNDARY')
    else:
        raise RuntimeError('R9_R4_NO_BOUNDED_G4_STOP__DO_NOT_ACCEPT')
    duration=time.monotonic()-start
    expected=s['target_completed_epoch']-1
    need(getattr(trainer,'_g4_completed',None),'NO_G4_ENTRIES')
    entry=trainer._g4_completed[-1]
    need(entry['completed_epoch']==expected and entry['terminal'] is False,'EPOCH_CURSOR')
    proof=verify_epoch(root=Path(s['publication_root']),arm=arm,
                       epoch=expected,external_manifest_sha256=entry['manifest_sha256'])
    need(proof.get('status')=='G4C_LOCAL_VERIFICATION_PASS_ONLY','PUBLICATION_BYTES')
    need(trainer._d4e_optimizer_updates>0,'NO_OPTIMIZER_UPDATES')
    need(trainer.ema.updates==trainer._d4e_optimizer_updates,'EMA_STEPS')
    need(trainer.scaler.is_enabled() and float(trainer.scaler.get_scale())>=1,'SCALER')
    from argparse import Namespace
    from R9_OFFSESSION_TRANSPORT_R1 import export, inspected_archive
    dest=work/f'{arm}_epoch{expected+1:03d}_G4_LOCAL_EXPORT.zip'
    export(Namespace(publication_root=s['publication_root'],arm=arm,
                     completed_epoch=expected,manifest_sha256=entry['manifest_sha256'],
                     destination=str(dest)))
    checked=inspected_archive(dest,sidecar=Path(str(dest)+'.sha256'))
    need(checked['status']=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION' and
         checked['completed_epoch']==expected and checked['terminal'] is False,
         'LOCAL_TRANSPORT_VERIFY')
    report={'schema':'R9_R4_NATIVE_BOUND_ACCEPTANCE_NOT_OFFICIAL',
            'status':'PASS_NATIVE_GPU_AND_LOCAL_EXPORT__EXTERNAL_TRANSFER_PENDING',
            'arm':arm,'mode':mode,'run_id':s['run_id'],
            'physical_gpu_visible_before_import':before,'gpu_name':gpu_name,
            'completed_epoch':expected,'optimizer_updates':trainer._d4e_optimizer_updates,
            'ema_updates':trainer.ema.updates,'amp_scale':float(trainer.scaler.get_scale()),
            'duration_seconds':round(duration,3),'g4_manifest_sha256':entry['manifest_sha256'],
            'local_export_zip':str(dest),'local_export_zip_sha256':r3.file_sha(dest),
            'local_export_sidecar':str(dest)+'.sha256',
            'offsession_export_verified':False,'full_epoch_100_tested':False,
            'official_training_authorized':False,'b_test_access':'NONE'}
    save_new(work/'R9_R4_NATIVE_BOUNDED_REPORT.json',report)
    print('R9_R4_NATIVE_BOUND_GPU_REPORT='+json.dumps(report,sort_keys=True),flush=True)
    print('IMPORTANT_LOCAL_KAGGLE_ZIP_IS_NOT_OFFSITE_VERIFIED__DOWNLOAD_AND_VERIFY',flush=True)
if __name__=='__main__':main()
