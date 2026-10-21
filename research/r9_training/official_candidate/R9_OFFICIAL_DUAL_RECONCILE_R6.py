#!/usr/bin/env python3
"""R6 dual-arm post-process proof gate, independent cursors; no GPU invocation."""
from __future__ import annotations
import json
from pathlib import Path


def need(c,k):
    if not c:raise RuntimeError('R9_R6_DUAL_FAIL_CLOSED_'+k)


def reconcile(records, run_group_prefix):
    need(type(records) is dict and set(records)=={'MV00','MV01'},'EXACT_TWO_ARMS')
    need(isinstance(run_group_prefix,str) and 1<=len(run_group_prefix)<=100,
         'RUN_GROUP_ID')
    result={}
    for arm in ('MV00','MV01'):
        d=records[arm]
        need(type(d) is dict and d.get('schema')=='DETECTION_CHAT25_R6_LOCAL_NATIVE_SESSION_RESULT',
             'SCHEMA_'+arm)
        need(d.get('arm')==arm and d.get('b_test_access')=='NONE' and
             d.get('independent_offsite_verified') is False,'ARM_AND_AUTHORITY_'+arm)
        need(d.get('status')=='GPU_EXECUTION_LOCAL_BYTES_ONLY_OFFSITE_PENDING',
             'LOCAL_RESULT_'+arm)
        run=d.get('run_id')
        need(isinstance(run,str) and run.startswith(run_group_prefix+'_') and
             run.endswith('_'+arm), 'RUN_ID_'+arm)
        epoch=d.get('completed_epoch_zero_based')
        need(type(epoch) is int and 0<=epoch<=99,'EPOCH_'+arm)
        need(d.get('terminal') is (epoch==99),'TERMINAL_'+arm)
        result[arm]={'run_id':run,'completed_epoch_zero_based':epoch,
                     'next_epoch_zero_based':None if epoch==99 else epoch+1,
                     'terminal':epoch==99,'external_persistence_verified':False}
    need(records['MV00']['run_id']!=records['MV01']['run_id'],'RUN_COLLISION')
    return {'schema':'R9_R6_DUAL_RECONCILIATION_LOCAL_ONLY',
            'status':'BOTH_ARMS_LOCAL_REPORTS_PASS_OFFSITE_PENDING',
            'arms':result,'official_training_authorized_here':False,
            'b_test_access':'NONE'}


def review_report_paths(paths, run_group_prefix):
    need(type(paths) is dict and set(paths)=={'MV00','MV01'},'PATH_KEYS')
    records={}
    for arm,p in paths.items():
        path=Path(p)
        need(path.is_file() and not path.is_symlink(),'MISSING_REPORT')
        records[arm]=json.loads(path.read_text('utf8'))
    return reconcile(records,run_group_prefix)
