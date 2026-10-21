#!/usr/bin/env python3
"""R1-R7 consolidated CPU review only: 120 established + 32 R7 tests.

No CUDA, no checkpoint deserialization, no patient data, no Git mutation,
no GPU authority. Writes exact-new JSON result outside the Git worktree.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
SUITES=(('R7 native new-path consent/limits','TEST_R9_OFFICIAL_R7_NEWPATH_GPU_ACCEPTANCE.py',18),
        ('R7 two-arm local G4 output integrity','TEST_R9_OFFICIAL_R7_ACCEPTANCE_EXPORT.py',14))

def need(ok,code):
 if not ok:raise RuntimeError('R9_R7_CPU_FAIL_CLOSED_'+code)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--repo-root',required=True);parser.add_argument('--report',required=True)
 args=parser.parse_args();root=Path(args.repo_root).resolve(strict=True)
 c=root/'research/r9_training/official_candidate'
 report=Path(args.report).resolve()
 need(c.is_dir() and report.parent.is_dir() and not report.exists() and
      root not in report.parents,'REPORT_PATH')
 older=report.with_name(report.stem+'_R6_BASELINE.json')
 need(not older.exists() and not older.with_name(older.stem+'_R5_BASELINE.json').exists(),
      'PREEXISTING_BASELINE')
 r=subprocess.run([sys.executable,str(c/'R9_OFFICIAL_R6_CPU_GATE.py'),
     '--repo-root',str(root),'--report',str(older)],cwd=root,capture_output=True,text=True)
 print(r.stdout,flush=True)
 if r.returncode:print(r.stderr,file=sys.stderr,flush=True)
 need(r.returncode==0 and older.is_file(),'R6_120_TEST_GATE')
 d=json.loads(older.read_text('utf-8'))
 need(d.get('total_tests_passed')==120 and d.get('frozen_source_files_verified')==343 and
      d.get('official_training_authorized') is False,'R6_REPORT_CONTENT')
 results=[]
 for desc,name,n in SUITES:
  p=c/name;need(p.is_file() and not p.is_symlink(),'MISSING_'+name)
  t=subprocess.run([sys.executable,str(p)],cwd=c,capture_output=True,text=True)
  if t.returncode:print(t.stderr+t.stdout,file=sys.stderr,flush=True)
  need(t.returncode==0 and f'Ran {n} tests' in t.stderr+t.stdout and
       '\nOK' in t.stderr+t.stdout,'R7_TESTS_'+name)
  print(f'PASS {desc}: {n}',flush=True)
  results.append({'description':desc,'passed':n})
 release=json.loads((c/'R9_OFFICIAL_R7_RELEASE_RECORD.json').read_text('utf-8'))
 need(release.get('schema')=='DETECTION_CHAT25_R7_NEWPATH_RELEASE_REVIEW' and
      release.get('frozen_science_commit')=='b783eb3a470ab010785a8a24a90a666e9a3f2ed6' and
      release.get('official_training_authorized') is False and
      release.get('release_gpu_production_ready') is False and
      release.get('epoch100_native_verified') is False and
      release.get('offsite_persistence_verified') is False and
      release.get('b_test_access')=='NONE','RELEASE_SCOPE')
 result={'schema':'DETECTION_CHAT25_R7_CPU_CONSOLIDATED_ACCEPTANCE',
    'status':'PASS_R7_CPU_CANDIDATE_ONLY_GPU_ACCEPTANCE_UNTESTED',
    'prior_tests':120,'r7_tests':32,'total_tests_passed':152,
    'frozen_source_files_verified':343,'frozen_science_commit':release['frozen_science_commit'],
    'r7_suites':results,'native_r7_gpu_run':False,'terminal_epoch100_gpu_proven':False,
    'kaggle_saved_version_verified':False,'offsite_verified':False,
    'official_training_authorized':False,'b_test_access':'NONE'}
 with report.open('x',encoding='utf-8') as f:
  json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 print('CHAT25_R7_COMBINED_CPU_GATE_PASS_TESTS=152')
 print('REVIEW_ONLY_NO_GPU_NO_BTEST_NO_OFFICIAL_TRAINING_AUTHORIZATION')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print('R7 CPU GATE FAILED:',repr(exc),file=sys.stderr);sys.exit(2)
