#!/usr/bin/env python3
"""Dual-T4 final-production session supervisor and fail-closed output export.

Default is CPU review. Execution is impossible without explicit BOTH-arm user
approval, pinned release, and --execute-approved-final-run. Saves independently
verified local ZIP bytes in /kaggle/working as .zipblob per arm for Kaggle
Save & Run All output capture. It CANNOT certify persistence until a DIFFERENT
session/Windows storage verifies bytes. No source/archive deletion at any point.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import R9_OFFICIAL_NATIVE_SESSION_R6 as r6
from R9_PRODUCTION_APPROVAL_GATE_C1 import (validate as approval_validate,
                                            validate_acceptance as acceptance_validate)

ARMS=('MV00','MV01')
REPORT='C1_FINAL_NATIVE_SESSION_REPORT.json'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def validate_output(arm,spec,*,inspector,release_sha,run_class):
    root=Path(spec['working_root'])
    report=root/REPORT
    r6.need(report.is_file() and not report.is_symlink(),'REPORT_NOT_PRESENT_'+arm)
    info=r6._read_json(report)
    r6.need(info.get('schema')=='DETECTION_CHAT25_FINAL_NATIVE_SESSION_PROVENANCE_C1'
            and info.get('status')=='NATIVE_GPU_SESSION_LOCAL_G4_VALIDATED_OFFSITE_UNVERIFIED'
            and info.get('arm')==arm and info.get('run_id')==spec['run_id']
            and info.get('published_release_commit') == release_sha
            and info.get('run_class') == run_class
            and info.get('offsite_persistence_verified') is False
            and info.get('b_test_access')=='NONE', 'REPORT_FIELDS_'+arm)
    cursor=info.get('completed_epoch_zero_based')
    r6.need(type(cursor) is int and
            spec['start_epoch_one_based']-1 <= cursor < 100 and
            cursor < spec['target_epoch_one_based'], 'REPORT_CURSOR_'+arm)
    r6.need(info.get('terminal') is (cursor==99), 'TERMINAL_CURSOR_'+arm)
    original=root/f'{arm}_epoch{cursor+1:03d}_G4_LOCAL_EXPORT.zip'
    sidecar=Path(str(original)+'.sha256')
    r6.need(info.get('original_g4_zip')==str(original) and
            original.is_file() and not original.is_symlink() and
            sidecar.is_file() and not sidecar.is_symlink(), 'ARCHIVE_PATH_'+arm)
    h=digest(original)
    r6.need(info['original_g4_zip_sha256']==h,'ARCHIVE_HASH_'+arm)
    r6.need(sidecar.read_text('utf-8').strip().split()==[h,original.name],
            'SIDECAR_'+arm)
    inspection=inspector(original,sidecar=sidecar)
    r6.need(inspection['status']=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION' and
            inspection['arm']==arm and inspection['completed_epoch']==cursor and
            inspection['terminal']==info['terminal'] and inspection['zip_sha256']==h,
            'AUTHENTIC_ARCHIVE_'+arm)
    return {'arm':arm,'run_id':info['run_id'],
            'completed_epoch_zero_based':cursor,'terminal':info['terminal'],
            'zip_sha256':h,'zip_bytes':original.stat().st_size,
            'sources':{'zip':original,'sidecar':sidecar,'report':report}}

def stage_available(*,specs,results,output_root,inspector,release_sha,run_class,max_bytes=8*(1<<30)):
    r6.need(set(specs)==set(ARMS) and set(results)==set(ARMS),'TWO_ARM_KEYS')
    root=Path(output_root)
    r6.need(str(root).startswith('/kaggle/working/') and root.parent.is_dir()
            and not root.exists() and not root.is_symlink(),'FRESH_KAGGLE_OUTPUT_ROOT')
    valid={};failed={};total=0
    for arm in ARMS:
        if results[arm].get('exit_code') != 0:
            failed[arm]={'reason':'WORKER_NONZERO_EXIT','exit_code':results[arm]['exit_code']}
            continue
        try:
            data=validate_output(arm,specs[arm],inspector=inspector,
                                 release_sha=release_sha,run_class=run_class)
            valid[arm]=data
            total+=sum(Path(v).stat().st_size for v in data['sources'].values())
        except Exception as exc:
            failed[arm]={'reason':'OUTPUT_VALIDATION_FAILED','detail':str(exc)}
    r6.need(type(max_bytes) is int and total <= max_bytes, 'KAGGLE_OUTPUT_BYTE_BUDGET')
    r6.need(bool(valid), 'NO_VALID_RECOVERABLE_ARM_OUTPUT')
    root.mkdir(mode=0o700)
    try:
        index={}
        for arm,data in valid.items():
            d=root/arm;d.mkdir(mode=0o700)
            e={k:v for k,v in data.items() if k!='sources'}
            for kind,path in data['sources'].items():
                dest=d/(path.name+'.zipblob' if kind=='zip' else path.name)
                with path.open('rb') as src,dest.open('xb') as dst:
                    shutil.copyfileobj(src,dst,1<<20);dst.flush();os.fsync(dst.fileno())
                r6.need(digest(path)==digest(dest),'OUTPUT_BYTE_EQUIVALENCE_'+arm+'_'+kind)
                e[kind+'_relative']=dest.relative_to(root).as_posix()
                e[kind+'_sha256']=digest(dest)
            index[arm]=e
        status='BOTH_ARMS_LOCAL_OUTPUTS_VERIFIED_EXTERNAL_PENDING' if len(valid)==2 else (
               'PARTIAL_ARM_LOCAL_OUTPUT_SAVED_OTHER_FAILED_REQUIRES_MANUAL_REVIEW')
        summary={'schema':'DETECTION_CHAT25_PRODUCTION_SESSION_KAGGLE_OUTPUT_INDEX_C1',
                 'status':status,'arms':index,'failures':failed,
                 'originals_preserved_without_deletion':True,
                 'offsite_persistence_verified':False,
                 'b_test_access':'NONE','output_bytes':total,
                 'both_arms_completed':len(valid)==2,
                 'published_release_commit':release_sha, 'run_class':run_class}
        with (root/'C1_OUTPUT_INDEX.json').open('x',encoding='utf-8') as f:
            json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
            f.flush();os.fsync(f.fileno())
        return summary
    except BaseException:
        shutil.rmtree(root)
        raise

def run_two(specs,release,approval,release_sha,*,worker,output_root,logs_root,spec_paths,approval_path,run_class):
    if run_class=='OFFICIAL_100EPOCH':
        approval_validate(specs,approval,release_sha,execute=True)
    elif run_class=='NONOFFICIAL_FINALPATH_ACCEPTANCE':
        acceptance_validate(specs,approval,release_sha)
    else:raise RuntimeError('C1_DUAL_INVALID_RUN_CLASS')
    for arm in ARMS:r6.validate_spec(specs[arm],release)
    r6.need(all(Path(specs[a]['working_root']).parent.is_dir() and
                not Path(specs[a]['working_root']).exists() for a in ARMS),
                'NEW_WORK_ROOTS')
    logdir=Path(logs_root)
    r6.need(str(logdir).startswith('/kaggle/working/') and logdir.parent.is_dir()
            and not logdir.exists(),'NEW_LOGDIR')
    logdir.mkdir(mode=0o700)
    proc={};streams={};results={}
    started=time.monotonic()
    try:
        for arm in ARMS:
            env=dict(os.environ)
            for k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
                env.pop(k,None)
            env['CUDA_VISIBLE_DEVICES']=r6.ARMS[arm]
            env['PYTHONUNBUFFERED']='1'
            log=(logdir/(arm+'_native_stdout_stderr.log')).open('x',encoding='utf-8')
            streams[arm]=log
            cmd=[sys.executable,'-u',str(worker),'--spec',spec_paths[arm],
                '--peer-spec',spec_paths['MV01' if arm=='MV00' else 'MV00'],
                '--historical-r6-release',specs[arm]['release_record_path'],
                '--authorization',approval_path,
                '--release-sha',release_sha,
                '--execute-approved-final-run' if run_class=='OFFICIAL_100EPOCH' else
                '--execute-nonofficial-finalpath-acceptance']
            proc[arm]=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
        # Supplied session deadline never exceeds 10h, leaving Kaggle output
        # collection reserve; worker has an earlier epoch-boundary deadline.
        timeout=min(max(specs[a]['monotonic_session_seconds'] for a in ARMS)+2400,39600)
        while any(p.poll() is None for p in proc.values()):
            if time.monotonic()-started > timeout:
                for p in proc.values():
                    if p.poll() is None:p.terminate()
                break
            time.sleep(2)
    finally:
        for a,p in proc.items():
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=15)
            results[a]={'exit_code':p.returncode,
                        'log':str(logdir/(a+'_native_stdout_stderr.log'))}
        for s in streams.values():s.close()
    code=Path(specs['MV00']['source_root']).resolve().parents[2]/'research/r9_training/code'
    sys.path.insert(0,str(code))
    from R9_OFFSESSION_TRANSPORT_R1 import inspected_archive
    idx=stage_available(specs=specs,results=results,output_root=output_root,
                        inspector=inspected_archive,release_sha=release_sha,run_class=run_class)
    print(json.dumps({'children':results,'output':idx},indent=2,default=str),flush=True)
    return idx


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mv00',required=True);p.add_argument('--mv01',required=True)
    p.add_argument('--release-sha',required=True)
    p.add_argument('--authorization')
    p.add_argument('--output-root',required=True)
    p.add_argument('--logs-root',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--execute-approved-final-run',action='store_true')
    mode.add_argument('--execute-nonofficial-finalpath-acceptance',action='store_true')
    a=p.parse_args()
    specs={arm:r6._read_json(path) for arm,path in
           [('MV00',a.mv00),('MV01',a.mv01)]}
    release=r6._read_json(specs['MV00']['release_record_path'])
    r6.need(specs['MV01']['release_record_path']==specs['MV00']['release_record_path'],
            'RELEASE_PATH_MISMATCH')
    for arm in ARMS:
        r6.validate_spec(specs[arm],release)
    approval_validate(specs,None,a.release_sha,execute=False)
    if not (a.execute_approved_final_run or a.execute_nonofficial_finalpath_acceptance):
        print('C1 DUAL PLAN REVIEW PASS: NO_GPU_NO_AUTHORIZATION_NO_TEST');return
    r6.need(a.authorization is not None,'EXPLICIT_APPROVAL_REQUIRED')
    authorization=r6._read_json(a.authorization)
    run_class='OFFICIAL_100EPOCH' if a.execute_approved_final_run else 'NONOFFICIAL_FINALPATH_ACCEPTANCE'
    if run_class=='OFFICIAL_100EPOCH':
        approval_validate(specs,authorization,a.release_sha,execute=True)
    else:
        acceptance_validate(specs,authorization,a.release_sha)
    worker=Path(__file__).with_name('R9_PRODUCTION_NATIVE_WORKER_C1.py')
    idx=run_two(specs,release,authorization,a.release_sha,worker=worker,
                output_root=a.output_root,logs_root=a.logs_root,
                spec_paths={'MV00':str(Path(a.mv00).resolve()),'MV01':str(Path(a.mv01).resolve())},
                approval_path=str(Path(a.authorization).resolve()),
                run_class=run_class)
    print('C1_KAGGLE_SESSION_OUTPUT_LOCAL_ONLY='+idx['status'])
    if idx['both_arms_completed'] is not True:
        print('PARTIAL_RUN: manual incident review required; never resume failed arm without verified original capsule',flush=True)

if __name__=='__main__':main()
