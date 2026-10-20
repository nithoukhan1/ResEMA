#!/usr/bin/env python3
"""Launch EXACTLY TWO real bounded nonofficial GPU children, physically isolated T4s.

Input: user-written R9B R3 plan JSON for each arm (both must pass full byte preflight).
The launcher does not grant official training authorization.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import R9B_R3_NATIVE_WORKER_INTEGRATION as r3

def need(x,m):
    if not x:raise RuntimeError('R9_R4_LAUNCH_FAIL_CLOSED_'+m)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mv00',required=True);ap.add_argument('--mv01',required=True)
    ap.add_argument('--log-root',required=True)
    ap.add_argument('--accept-nonofficial-bounded-gpu',action='store_true')
    a=ap.parse_args()
    need(a.accept_nonofficial_bounded_gpu,'EXPLICIT_BOUND_ACCEPTANCE_SWITCH')
    specs={'MV00':r3.read_json(a.mv00),'MV01':r3.read_json(a.mv01)}
    for arm,s in specs.items():
        need(s.get('arm')==arm,'ARM_SPEC')
        r3.verify_plan(s,file_check=True)
        need(s['mode']=='FRESH_FROM_EARLY' and s['target_completed_epoch']==1,
             'LAUNCHER_ONLY_FRESH_ONE_EPOCH')
    need(specs['MV00']['run_id'] != specs['MV01']['run_id'],
         'UNIQUE_RUN_ID_PER_ARM_REQUIRED')
    roots=[Path(s['working_root']).resolve() for s in specs.values()]
    need(roots[0]!=roots[1] and all(not p.exists() for p in roots),'SEPARATE_FRESH_RUN_ROOTS')
    logroot=Path(a.log_root)
    need(logroot.parent.is_dir() and not logroot.exists(),'NEW_LOG_ROOT')
    logroot.mkdir(mode=0o700)
    worker=Path(__file__).with_name('R9_R4_BOUND_GPU_WORKER.py')
    proc={};handles={};results={}
    try:
        for arm in ('MV00','MV01'):
            env=dict(os.environ)
            for k in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(k,None)
            env['CUDA_VISIBLE_DEVICES']=r3.ARMS[arm];env['PYTHONUNBUFFERED']='1'
            log=logroot/f'{arm}_raw.log'
            handles[arm]=log.open('x',encoding='utf8')
            proc[arm]=subprocess.Popen([sys.executable,'-u',str(worker),
                       '--plan',a.mv00 if arm=='MV00' else a.mv01,
                       '--accept-nonofficial-bounded-gpu'],
                       stdout=handles[arm],stderr=subprocess.STDOUT,env=env)
        for arm,p in proc.items():
            rc=p.wait()
            results[arm]={'exit_code':rc,'log':str(logroot/f'{arm}_raw.log')}
        need(all(x['exit_code']==0 for x in results.values()),'CHILD_PROCESS_FAILURE_READ_RAW_LOGS')
        for arm,s in specs.items():
            f=Path(s['working_root'])/'R9_R4_NATIVE_BOUNDED_REPORT.json'
            need(f.is_file() and json.loads(f.read_text())['status']==
                 'PASS_NATIVE_GPU_AND_LOCAL_EXPORT__EXTERNAL_TRANSFER_PENDING',
                 'CHILD_REPORT_'+arm)
        print(json.dumps({'schema':'R9_R4_DUAL_NONOFFICIAL_NATIVE_BOUND',
                          'status':'PASS_TWO_ARMS_LOCALLY__EXTERNAL_DOWNLOAD_PENDING',
                          'results':results,'official_training_authorized':False},indent=2))
    finally:
        for p in proc.values():
            if p.poll() is None:p.terminate();p.wait()
        for f in handles.values():f.close()
        if results and any(x['exit_code']!=0 for x in results.values()):
            print('R9_R4_CHILD_FAILURE_LOGS='+json.dumps(results),file=sys.stderr,flush=True)
if __name__=='__main__':main()
