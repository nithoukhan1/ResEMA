#!/usr/bin/env python3
"""OFFLINE R9 official-release candidate contract check. NEVER launches GPU training.

This does NOT certify a release, produce official authorization, or provide an
executable 100-epoch worker. Optional --repo-root verifies original source bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

COMMIT = 'b783eb3a470ab010785a8a24a90a666e9a3f2ed6'
MANIFEST_SHA = 'e1abea8ae36f15d4e2589e4d267395acf7921bad347be72697dac114c47df53b'
RECIPE_SHA = 'dd0ef5903dc3229abfe0c5e9298d0c288e3c682d2d14a8314530caf4ad3ad074'
YAML_SHA = 'f2cc6e31f5f1be152f4dd7603a397e142963923f1d35de613eb7c61b168bf0c0'
ARMS = {'MV00': 'SCConv-Early-YOLO11s', 'MV01': 'SCConv-Early-YOLO11s+VGRA-V1'}


def need(condition, message):
    if not condition:
        raise RuntimeError('R9_OFFICIAL_R1_FAIL_CLOSED_' + message)


def read_json(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink(), 'JSON_NOT_REGULAR_FILE')
    return json.loads(p.read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_contract(record):
    """Only pass a review proposal with authorization deliberately FALSE."""
    need(isinstance(record, dict) and set(record) == {
        'schema', 'status', 'official_training_authorized', 'run_executable',
        'source', 'experiment', 'release_gates'}, 'CONTRACT_KEYS')
    need(record['schema'] == 'DETECTION_R9_OFFICIAL_RELEASE_CONTRACT_PROPOSAL_R1', 'SCHEMA')
    need(record['status'] == 'ENGINEERING_CANDIDATE_NOT_AUTHORIZED', 'STATUS')
    need(record['official_training_authorized'] is False and record['run_executable'] is False,
         'AUTHORIZATION_IS_FORBIDDEN_AT_THIS_STAGE')
    s = record['source']
    need(isinstance(s, dict) and s == {
        'baseline_git_commit': COMMIT, 'frozen_manifest_sha256': MANIFEST_SHA,
        'frozen_recipe_sha256': RECIPE_SHA, 'runtime_data_yaml_sha256': YAML_SHA},
         'SOURCE_IDENTITY')
    e = record['experiment']
    need(isinstance(e, dict) and e == {
        'dataset': 'GRAZPEDWRI-DX_FROZEN_PATIENT_DISJOINT_SPLIT_B',
        'train_split': 'B-TRAIN', 'development_split': 'B-VAL', 'test_split': 'SEALED',
        'arms': ARMS, 'physical_gpu': {'MV00': 0, 'MV01': 1},
        'epochs_per_arm': 100, 'batch_per_arm': 8, 'image_size': 1024,
        'optimizer': 'SGD', 'amp': True, 'seed': 42,
        'initialize_from': 'APPROVED_ORIGINAL_SCCONV_EARLY_PRETRAINED',
        'prohibit_rehearsal_checkpoints': True}, 'SCIENCE_CONTRACT')
    g = record['release_gates']
    need(isinstance(g, dict) and g == {
        'explicit_user_authorization_required': True,
        'production_worker_not_yet_approved': True,
        'notebook_save_run_all_not_yet_approved': True,
        'nonterminal_boundary_requires_original_full_state_capsule': True,
        'terminal_epoch100_requires_separate_native_proof': True,
        'independent_external_persistence_mandatory': True,
        'b_test_unseal_requires_separate_approval': True}, 'RELEASE_GATES')
    return {'status': 'PASS_REVIEW_CONTRACT_ONLY_NOT_AUTHORIZED',
            'official_training_authorized': False, 'gpu_execution': 'PROHIBITED',
            'b_test_access': 'NONE'}


def verify_repository_bytes(repo_root):
    """Re-run content-byte verification without executing/importing the science."""
    root = Path(repo_root).resolve(strict=True) / 'research' / 'r9_training'
    need(root.is_dir(), 'MISSING_R9_ROOT')
    manifest_file = root/'code'/'R9_CANDIDATE_MANIFEST_R1.json'
    recipe_file = root/'R9_RELEASE_RECIPE_CANDIDATE_R1.json'
    need(manifest_file.is_file() and not manifest_file.is_symlink(), 'MISSING_PINNED_MANIFEST')
    need(recipe_file.is_file() and not recipe_file.is_symlink(), 'MISSING_PINNED_RECIPE')
    need(sha256(manifest_file) == MANIFEST_SHA, 'FROZEN_MANIFEST_DRIFT')
    need(sha256(recipe_file) == RECIPE_SHA, 'FROZEN_RECIPE_DRIFT')
    recipe = read_json(recipe_file)
    need(recipe.get('official_training_authorized') is False and
         recipe.get('status') == 'RELEASE_CANDIDATE_NOT_AUTHORIZED',
         'HISTORICAL_RECIPE_GOVERNANCE_DRIFT')
    manifest = read_json(manifest_file)
    need(manifest.get('candidate_recipe_sha256') == RECIPE_SHA, 'MANIFEST_RECIPE_DRIFT')
    counts = {}
    for field, subdir, exact_count in (
        ('candidate_source_sha256', 'frozen_source', 336),
        ('unchanged_integration_modules_sha256', 'integration_modules', 7),
    ):
        paths = manifest.get(field)
        need(isinstance(paths, dict) and len(paths) == exact_count, 'MANIFEST_COUNT_'+field)
        base = root/subdir
        need(base.is_dir() and not base.is_symlink(), 'SOURCE_DIR_'+subdir)
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        need(actual == set(paths), 'SOURCE_INVENTORY_'+subdir)
        for relative, digest in paths.items():
            need(isinstance(relative, str) and relative and not relative.startswith('/') and
                 '..' not in Path(relative).parts and
                 isinstance(digest, str) and re.fullmatch('[a-f0-9]{64}', digest),
                 'BAD_MANIFEST_ENTRY')
            f = base/relative
            need(f.is_file() and not f.is_symlink() and sha256(f) == digest,
                 'PINNED_BYTES_'+subdir+'_'+relative)
        counts[subdir] = exact_count
    return {'status': 'PASS_ORIGINAL_FROZEN_SOURCE_BYTES', 'files_verified': counts,
            'recipe_sha256': RECIPE_SHA, 'manifest_sha256': MANIFEST_SHA,
            'actual_git_head': 'NOT_VERIFIED_BY_THIS_FILE_CHECK'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contract', required=True)
    ap.add_argument('--repo-root', help='Optional local git worktree root with all original R9 files')
    args = ap.parse_args(argv)
    report = {'contract': check_contract(read_json(args.contract)),
              'repository': 'NOT_REQUESTED',
              'release_ready': False, 'actual_training_run': False}
    if args.repo_root:
        report['repository'] = verify_repository_bytes(args.repo_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    print('NO_TRAINING_NO_AUTHORIZATION_NO_KAGGLE_NO_TEST_ACCESS')
    return report

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print('FAIL_CLOSED:', exc, file=sys.stderr)
        sys.exit(2)
