#!/usr/bin/env python3
"""Single R6 Windows acceptance: prior R1-R5 (85) + R6 (35) = 120 CPU tests.

Runs only subprocess tests and strict immutable record inspection. No CUDA,
Git commit/push, checkpoint loading, B-TEST, or release approval.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path

SUITES=(('Native session safety','TEST_R9_OFFICIAL_NATIVE_SESSION_R6.py',20),
        ('Output index and independent dual cursors','TEST_R9_OFFICIAL_EXPORT_DUAL_R6.py',15))

def need(cond,k):
    if not cond:raise RuntimeError('R9_R6_CPU_GATE_FAIL_CLOSED_'+k)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',required=True)
    p.add_argument('--report',required=True)
    a=p.parse_args()
    root=Path(a.repo_root).resolve(strict=True)
    candidate=root/'research/r9_training/official_candidate'
    report=Path(a.report).resolve()
    need(candidate.is_dir() and report.parent.is_dir() and
         not report.exists() and root not in report.parents,'REPORT_PATH')
    old=report.with_name(report.stem+'_R5_BASELINE.json')
    need(not old.exists() and not old.with_name(old.stem+'_R4_BASELINE.json').exists(),
         'BASELINE_REPORT_EXISTS')
    prev=subprocess.run([sys.executable,str(candidate/'R9_OFFICIAL_COMBINED_CPU_GATE_R5.py'),
                         '--repo-root',str(root),'--report',str(old)],cwd=root,
                         capture_output=True,text=True)
    print(prev.stdout,flush=True)
    if prev.returncode:
        print(prev.stderr,file=sys.stderr,flush=True)
    need(prev.returncode==0 and old.is_file(),'PREVIOUS_ACCEPTED_GATE')
    baseline=json.loads(old.read_text('utf8'))
    need(baseline.get('total_tests_passed')==85 and
         baseline.get('frozen_source_files_verified')==343 and
         baseline.get('official_training_authorized') is False,'R5_GATE_CONTENT')
    results=[]
    for desc,name,count in SUITES:
        file=candidate/name
        need(file.is_file() and not file.is_symlink(),'TEST_FILE_'+name)
        out=subprocess.run([sys.executable,str(file)],cwd=candidate,
                           capture_output=True,text=True)
        both=out.stdout+out.stderr
        if out.returncode:print(both,file=sys.stderr,flush=True)
        need(out.returncode==0 and 'Ran '+str(count)+' tests' in both and
             '\nOK' in both,'R6_TESTS_'+name)
        print(f'PASS R6 {desc}: {count}',flush=True)
        results.append({'suite':desc,'passed':count})
    record=json.loads((candidate/'R9_OFFICIAL_R6_RELEASE_RECORD.json').read_text('utf8'))
    from R9_OFFICIAL_NATIVE_SESSION_R6 import RELEASE_SCHEMA,BASE_COMMIT
    need(record.get('schema')==RELEASE_SCHEMA and
         record.get('frozen_science_commit')==BASE_COMMIT and
         record.get('official_training_authorized') is False and
         record.get('run_executable') is False and
         record.get('gpu_acceptance_performed') is False and
         record.get('terminal_epoch100_gpu_verified') is False and
         record.get('b_test_access')=='NONE','RELEASE_RECORD_STILL_LOCKED')
    summary={'schema':'DETECTION_CHAT25_R6_CPU_CONSOLIDATED_GATE',
       'status':'PASS_REVIEW_CANDIDATE_NOT_GPU_EXECUTABLE',
       'prior_r1_to_r5_tests':85,'r6_suites':results,
       'total_tests_passed':85+sum(x['passed'] for x in results),
       'frozen_source_files_verified':343,
       'frozen_science_commit':BASE_COMMIT,
       'native_gpu_run':False,'kaggle_save_run_all_verified':False,
       'epoch100_gpu_proven':False,'offsite_state_verified':False,
       'official_training_authorized':False,'b_test_access':'NONE'}
    with report.open('x',encoding='utf8') as f:
        json.dump(summary,f,sort_keys=True,indent=2);f.write('\n')
        f.flush();os.fsync(f.fileno())
    print('CHAT25_R6_CPU_GATE_PASS_TESTS=120')
    print('R6_CANDIDATE_ONLY_NO_GPU_NO_GIT_PUSH_NO_AUTHORIZATION')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R6 CPU GATE FAILED:',repr(exc),file=sys.stderr)
        raise SystemExit(2)
