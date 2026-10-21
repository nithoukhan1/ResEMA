#!/usr/bin/env python3
"""Two physical-T4 subprocess supervision candidate; CLI review-only.

The subprocess routine is usable for controlled CPU/synthetic tests. It is NOT a
GPU training authorization or an executable Kaggle notebook release. In particular,
child report provenance and offsite preservation must be integrated and accepted.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ARM_GPU={'MV00':'0','MV01':'1'}


def need(cond, code):
    if not cond: raise RuntimeError('R9_R5_DUAL_FAIL_CLOSED_'+code)


def make_child_environment(arm, base=None):
    need(arm in ARM_GPU, 'ARM')
    env=dict(os.environ if base is None else base)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    env['CUDA_VISIBLE_DEVICES']=ARM_GPU[arm]
    env['PYTHONUNBUFFERED']='1'
    return env


def validate_process_descriptors(descriptors):
    need(type(descriptors) is dict and set(descriptors)==set(ARM_GPU), 'TWO_ARMS')
    for arm,s in descriptors.items():
        need(type(s) is dict and set(s)=={'argv','run_id'},'DESCRIPTOR_KEYS')
        need(isinstance(s['argv'],list) and len(s['argv'])>=2 and
             all(isinstance(x,str) and x and '\x00' not in x for x in s['argv']),
             'ARGV_'+arm)
        need(isinstance(s['run_id'],str) and len(s['run_id'])>0,'RUN_ID_'+arm)
        need(all(x not in ' '.join(s['argv']).lower() for x in (
            '--device 1,0','--device 0,1','--distributed','--ddp')), 'DDP_'+arm)
    need(descriptors['MV00']['run_id'] != descriptors['MV01']['run_id'],'COLLIDING_RUN_ID')


def run_children_cpu_or_separately_authorized(*, descriptors, log_root, env_base=None,
                                              poll_seconds=0.05, timeout_seconds=10.0):
    """Supervise children. Synthetic test harness only until actual GPU release approved.

    On any child error/timeout, terminate its sibling; never report dual success.
    Failures cannot be assumed to have externally persisted checkpoints.
    """
    validate_process_descriptors(descriptors)
    need(type(poll_seconds) in (float,int) and poll_seconds>0 and
         type(timeout_seconds) in (float,int) and timeout_seconds>0,'TIMING')
    root=Path(log_root)
    need(root.parent.is_dir() and not root.exists() and not root.is_symlink(),'NEW_LOG_ROOT')
    root.mkdir(mode=0o700)
    proc={}; logs={}; completed={}
    begun=time.monotonic()
    try:
        for arm in ARM_GPU:
            f=(root/f'{arm}.stdout_stderr.log').open('x',encoding='utf-8')
            logs[arm]=f
            proc[arm]=subprocess.Popen(
                descriptors[arm]['argv'], stdout=f,stderr=subprocess.STDOUT,
                env=make_child_environment(arm, env_base),
                stdin=subprocess.DEVNULL,close_fds=True)
        while len(completed)<len(ARM_GPU):
            for arm,p in proc.items():
                rc=p.poll()
                if rc is not None and arm not in completed:
                    completed[arm]=rc
                if rc is not None and rc!=0:
                    raise RuntimeError('R9_R5_DUAL_FAIL_CLOSED_CHILD_'+arm+'_RC_'+str(rc))
            if time.monotonic()-begun>timeout_seconds:
                raise TimeoutError('R9_R5_DUAL_FAIL_CLOSED_SUPERVISOR_TIMEOUT')
            if len(completed)<len(ARM_GPU):time.sleep(poll_seconds)
        need(all(completed[x]==0 for x in ARM_GPU),'NONZERO_EXIT')
        return {'status':'TWO_CHILDREN_EXIT_ZERO_UNVERIFIED_APPLICATION_RESULT',
                'returncodes':completed,
                'log_root':str(root),'gpu_training_verified':False,
                'offsite_persistence_verified':False,'official_training_authorized':False}
    finally:
        for p in proc.values():
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait(timeout=5)
        for f in logs.values():f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-plan',required=True)
    p.add_argument('--contract',required=True)
    args=p.parse_args()
    from R9_OFFICIAL_SESSION_PLAN_GATE_R3 import strict_load, validate
    summary=validate(strict_load(args.contract),strict_load(args.review_plan))
    need(summary['status']=='PASS_STATIC_CONTRACT_NOT_EXECUTABLE','REVIEW_PLAN')
    print(json.dumps({**summary,
          'native_worker_integrated':False,
          'kaggle_save_run_all_verified':False,
          'gpu_training_authorized':False},indent=2,sort_keys=True))
    print('R5 REVIEW ONLY; NO SUBPROCESSES STARTED')

if __name__=='__main__':main()
