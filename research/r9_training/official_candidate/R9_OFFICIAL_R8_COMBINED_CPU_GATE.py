#!/usr/bin/env python3
"""One R1–R8 Windows precommit CPU gate; no GPU, network, Git writes or authorization."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
SUITES=(('R8 real-Git release lineage','TEST_R9_OFFICIAL_RELEASE_IDENTITY_R8.py',15),
        ('R8 GPU acceptance pin/consent','TEST_R9_OFFICIAL_R8_NEWPATH_GPU_ACCEPTANCE.py',16))

def need(ok,code):
 if not ok:raise RuntimeError('R9_R8_CPU_FAIL_CLOSED_'+code)

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--report',required=True)
 x=p.parse_args();root=Path(x.repo_root).resolve(strict=True);candidate=root/'research/r9_training/official_candidate'
 report=Path(x.report).resolve()
 need(report.parent.is_dir() and not report.exists() and root not in report.parents,'NEW_OUTSIDE_REPORT')
 old=report.with_name(report.stem+'_R7_BASELINE.json')
 need(not old.exists(),'NEW_R7_BASELINE')
 run=subprocess.run([sys.executable,str(candidate/'R9_OFFICIAL_R7_CPU_GATE.py'),
                      '--repo-root',str(root),'--report',str(old)],cwd=root,capture_output=True,text=True)
 print(run.stdout,flush=True)
 if run.returncode:print(run.stderr,file=sys.stderr,flush=True)
 need(run.returncode==0 and old.is_file(),'R7_GATE_FAILED')
 baseline=json.loads(old.read_text('utf-8'))
 need(baseline.get('total_tests_passed')==152 and
      baseline.get('frozen_source_files_verified')==343 and
      baseline.get('official_training_authorized') is False,'R7_BASELINE_CONTENT')
 outcomes=[]
 for label,script,n in SUITES:
  s=candidate/script;need(s.is_file() and not s.is_symlink(),'MISSING_'+script)
  result=subprocess.run([sys.executable,str(s)],cwd=candidate,capture_output=True,text=True)
  output=result.stdout+result.stderr
  if result.returncode:print(output,file=sys.stderr,flush=True)
  need(result.returncode==0 and f'Ran {n} tests' in output and '\nOK' in output,
       'FAILED_'+script)
  print(f'PASS {label}: {n}',flush=True)
  outcomes.append({'name':label,'tests':n})
 r8=json.loads((candidate/'R9_OFFICIAL_R8_RELEASE_RECORD.json').read_text('utf-8'))
 need(r8.get('schema')=='DETECTION_CHAT25_R8_RELEASE_IDENTITY_REVIEW_V1' and
      r8.get('official_release_git_commit') is None and
      r8.get('official_training_authorized') is False and
      r8.get('r8_nonofficial_gpu_acceptance_authorized') is False and
      r8.get('terminal_epoch100_verified') is False and
      r8.get('b_test_access')=='NONE','R8_REVIEW_RECORD')
 receipt={'schema':'DETECTION_CHAT25_R8_CONSOLIDATED_CPU_ACCEPTANCE',
          'status':'PASS_183_CPU_TESTS_PRECOMMIT_ONLY_NOT_GPU_RELEASE',
          'prior_tests':152,'new_tests':31,'total_tests_passed':183,
          'frozen_source_files_verified':343,'r8_suites':outcomes,
          'published_release_commit':None,'native_r8_gpu_run':False,
          'terminal_epoch100_gpu_proven':False,
          'kaggle_save_version_verified':False,
          'offsite_verified':False,'official_training_authorized':False,
          'b_test_access':'NONE'}
 with report.open('x',encoding='utf-8') as f:
  json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print('CHAT25_R8_COMBINED_CPU_GATE_PASS_TESTS=183')
 print('NO_GPU_NO_GIT_COMMIT_NO_PUSH_BTEST_SEALED_NOT_AUTHORIZED')

if __name__=='__main__':
 try:main()
 except Exception as e:
  print('R8 CPU GATE FAILED:',repr(e),file=sys.stderr);sys.exit(2)
