#!/usr/bin/env python3
"""One Windows command: accepted R1/R3/R4 (65) plus R5 (20) CPU-only.

Does not mutate Git or grant official experiment authorization. Outputs report
outside repository; original frozen-science verification delegated to R4 gate.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

NEW_SUITES=(('R5 native MRO wiring','TEST_R9_OFFICIAL_WIRING_R5.py',9),
            ('R5 local ZIP byte staging','TEST_R9_OFFICIAL_OUTPUT_PRESERVE_R5.py',6),
            ('R5 dual-process supervision','TEST_R9_OFFICIAL_DUAL_PROCESS_R5.py',5))


def need(ok,msg):
    if not ok:raise RuntimeError('R9_R5_COMBINED_FAIL_CLOSED_'+msg)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',required=True);ap.add_argument('--report',required=True)
    a=ap.parse_args()
    root=Path(a.repo_root).resolve(strict=True)
    active=root/'research/r9_training/official_candidate'
    output=Path(a.report).resolve()
    need(active.is_dir() and output.parent.is_dir() and not output.exists() and
         root not in output.parents,'PATHS')
    previous=output.with_name(output.stem+'_R4_BASELINE.json')
    need(not previous.exists(),'R4_REPORT_EXISTS')
    call=[sys.executable,str(active/'R9_OFFICIAL_COMBINED_CPU_GATE_R4.py'),
          '--repo-root',str(root),'--report',str(previous)]
    r=subprocess.run(call,cwd=root,capture_output=True,text=True)
    print(r.stdout,flush=True)
    if r.returncode: print(r.stderr,file=sys.stderr,flush=True)
    need(r.returncode==0 and previous.is_file(),'R4_GATE')
    old=json.loads(previous.read_text('utf-8'))
    need(old.get('total_tests_passed')==65 and
         old.get('original_source_files_verified')==343 and
         old.get('official_training_authorized') is False,'R4_BASELINE_REPORT')
    suites=[]
    for label, filename, count in NEW_SUITES:
        p=active/filename
        need(p.is_file() and not p.is_symlink(),'FILE_'+filename)
        r=subprocess.run([sys.executable,str(p)],cwd=active,capture_output=True,text=True)
        if r.returncode:print(r.stdout+r.stderr,file=sys.stderr,flush=True)
        need(r.returncode==0 and 'Ran '+str(count)+' tests' in r.stdout+r.stderr and
             '\nOK' in r.stdout+r.stderr,'TEST_'+filename)
        suites.append({'suite':label,'tests_passed':count,'status':'PASS'})
        print('PASS '+label+': '+str(count),flush=True)
    release=json.loads((active/'R9_OFFICIAL_R5_RELEASE_RECORD.json').read_text('utf-8'))
    from R9_OFFICIAL_NATIVE_WIRING_R5 import verify_external_release_boundary
    contract=json.loads((active/'R9_OFFICIAL_RELEASE_CONTRACT_R1.json').read_text('utf-8'))
    verify_external_release_boundary(original_contract=contract,new_release=release)
    record={
        'schema':'CHAT25_R5_CONSOLIDATED_CPU_GATE',
        'status':'PASS_REVIEW_ONLY_NOT_GPU_EXECUTABLE',
        'baseline_git_commit':old['git_baseline_commit'],
        'frozen_source_files_verified':343,
        'accepted_prior_tests_passed':65,
        'additional_r5_tests':suites,
        'total_tests_passed':65+sum(s['tests_passed'] for s in suites),
        'native_gpu_worker_executed':False,
        'kaggle_notebook_save_run_all_validated':False,
        'terminal_epoch100_proven':False,
        'offsite_persistence_proven_by_r5':False,
        'official_training_authorized':False,
        'b_test_access':'NONE'
    }
    with output.open('x',encoding='utf8') as f:
        json.dump(record,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    print('CHAT25_R5_COMBINED_CPU_GATE_PASS_TESTS='+str(record['total_tests_passed']))
    print('NO_GPU_NO_GIT_CHANGES_NO_OFFICIAL_AUTHORIZATION')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('R5 FAIL-CLOSED',repr(e),file=sys.stderr)
        sys.exit(2)
