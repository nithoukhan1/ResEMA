#!/usr/bin/env python3
"""Combined local R1+R3+R4 preflight. CPU-only, never authorizes or runs GPU.

R2 is intentionally omitted: it is preserved as failed historical evidence.
This gate verifies four test suites and original source + branch if provided.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_COMMIT = 'b783eb3a470ab010785a8a24a90a666e9a3f2ed6'
BRANCH = 'research/vgra-r9-training'
SUITES = [
    ('R1 contract', 'TEST_R9_OFFICIAL_RELEASE_CONTRACT_R1.py', 10),
    ('R1 boundary policy', 'TEST_R9_SESSION_BOUNDARY_POLICY_R1.py', 12),
    ('R3 cross-platform session gate', 'TEST_R9_OFFICIAL_SESSION_PLAN_GATE_R3.py', 28),
    ('R4 adaptive post-G4 boundary', 'TEST_R9_OFFICIAL_ADAPTIVE_BOUNDARY_R4.py', 15),
]


def require(ok, code):
    if not ok:
        raise RuntimeError('R9_COMBINED_R4_FAIL_CLOSED_' + code)


def run(args, cwd):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    require(r.returncode == 0, 'SUBPROCESS_' + Path(args[1]).name if len(args)>1 else 'SUBPROCESS')
    return (r.stdout + r.stderr).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    root = Path(a.repo_root).resolve(strict=True)
    dest = Path(a.report).resolve()
    require(dest.parent.is_dir() and not dest.exists() and not dest.is_symlink(), 'REPORT_NEW')
    require(root not in dest.parents, 'REPORT_MUST_BE_OUTSIDE_GIT_WORKTREE')
    active = root / 'research' / 'r9_training' / 'official_candidate'
    require(active.is_dir() and not active.is_symlink(), 'CANDIDATE_DIR')
    branch = run(['git', 'branch', '--show-current'], root)
    head = run(['git', 'rev-parse', 'HEAD'], root)
    require(branch == BRANCH and head == BASE_COMMIT, 'GIT_IDENTITY')
    status = run(['git', 'status', '--porcelain=v1', '--untracked-files=all'], root)
    require(all(line.startswith('?? research/r9_training/official_candidate/') for line in status.splitlines()),
            'OTHER_FILES_MODIFIED')
    details = []
    for name, filename, expected in SUITES:
        f = active / filename
        require(f.is_file() and not f.is_symlink(), 'MISSING_' + filename)
        r = subprocess.run([sys.executable, str(f)], cwd=active, capture_output=True, text=True, check=False)
        log = (r.stdout + r.stderr).strip()
        require(r.returncode == 0 and f'Ran {expected} tests' in log and '\nOK' in log,
                'TEST_' + filename)
        details.append({'suite': name, 'tests_passed': expected, 'status': 'PASS'})
        print(f'PASS {name}: {expected}', flush=True)
    source_preflight = active / 'R9_OFFICIAL_RELEASE_PREFLIGHT_R1.py'
    contract = active / 'R9_OFFICIAL_RELEASE_CONTRACT_R1.json'
    r = subprocess.run([sys.executable, str(source_preflight), '--contract',str(contract),
                       '--repo-root',str(root)], cwd=root, capture_output=True, text=True, check=False)
    require(r.returncode == 0 and 'PASS_ORIGINAL_FROZEN_SOURCE_BYTES' in r.stdout and
            'NO_TRAINING_NO_AUTHORIZATION_NO_KAGGLE_NO_TEST_ACCESS' in r.stdout,
            'FROZEN_SOURCE_BYTES')
    record = {
        'schema': 'R9_CHAT25_R4_COMBINED_CPU_GATE_REVIEW_ONLY',
        'status': 'PASS_CPU_STATIC_ENGINEERING_ONLY',
        'git_branch': branch, 'git_baseline_commit': head,
        'suites': details, 'total_tests_passed': sum(x['tests_passed'] for x in details),
        'original_source_files_verified': 343,
        'native_multi_epoch_gpu_executed': False,
        'terminal_epoch100_gpu_verified': False,
        'independent_offsite_publication_verified': False,
        'official_training_authorized': False,
        'b_test_access': 'NONE',
    }
    with dest.open('x', encoding='utf8') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print('CHAT25_R4_COMBINED_CPU_GATE_PASS_TESTS=' + str(record['total_tests_passed']))
    print('REVIEW_ONLY_NO_GPU_NO_GIT_CHANGES_NO_OFFICIAL_AUTHORIZATION')


if __name__ == '__main__':
    try: main()
    except Exception as e:
        print('FAIL_CLOSED:',e, file=sys.stderr)
        sys.exit(2)
