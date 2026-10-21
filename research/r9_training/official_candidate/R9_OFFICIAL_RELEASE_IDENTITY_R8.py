#!/usr/bin/env python3
"""R8 immutable release-checkout identity gate; no GPU execution or authorization.

The scientific source baseline is a pinned ancestor, never the release HEAD.
An exact external release SHA, supplied in a reviewed Kaggle notebook after the
release commit exists, must equal HEAD. All changes since science baseline must
be ADDITIONS confined to official_candidate/ and notebooks/. Original science
bytes are checked independently by R1/R6 preflight, NOT inferred from ancestry.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

BASE = 'b783eb3a470ab010785a8a24a90a666e9a3f2ed6'
FULL_SHA = re.compile(r'[0-9a-f]{40}\Z')
ALLOWED_PREFIXES = ('research/r9_training/official_candidate/',
                    'research/r9_training/notebooks/')
REQUIRED_FILES = (
    'research/r9_training/official_candidate/R9_OFFICIAL_RELEASE_IDENTITY_R8.py',
    'research/r9_training/official_candidate/R9_OFFICIAL_R8_NEWPATH_GPU_ACCEPTANCE.py',
    'research/r9_training/official_candidate/R9_OFFICIAL_NATIVE_SESSION_R6.py',
    'research/r9_training/official_candidate/R9_OFFICIAL_R7_NEWPATH_GPU_ACCEPTANCE.py',
)
APPROVED_REMOTES = {
    'https://github.com/nithoukhan1/ResEMA',
    'https://github.com/nithoukhan1/ResEMA.git',
    'git@github.com:nithoukhan1/ResEMA.git',
}


def need(ok, code):
    if not ok:
        raise RuntimeError('R9_R8_RELEASE_IDENTITY_FAIL_CLOSED_' + code)


def git(repo, *args):
    proc = subprocess.run(['git','-C',str(repo),*args], capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def pin_shape(sha):
    need(type(sha) is str and FULL_SHA.fullmatch(sha) is not None and sha != BASE,
         'NEW_FULL_RELEASE_SHA_REQUIRED')
    return sha


def validate_checkout(checkout, pinned_release_sha, *, require_remote=True):
    """Check a *published* release checkout, with no writes, network or training.

    This is not authority for official training. The pin MUST be sourced outside
    the repository (for example, literal exact SHA in Kaggle Save Version).
    """
    pin_shape(pinned_release_sha)
    root = Path(checkout).resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), 'REPO_DIR')
    rc, out, _ = git(root, 'rev-parse', '--show-toplevel')
    need(rc == 0 and Path(out.decode().strip()).resolve() == root, 'EXACT_REPO_ROOT')
    rc, out, _ = git(root, 'rev-parse', '--verify', 'HEAD')
    need(rc == 0 and out.decode().strip() == pinned_release_sha,
         'EXACT_PINNED_RELEASE_HEAD')
    rc, out, _ = git(root, 'merge-base', '--is-ancestor', BASE, 'HEAD')
    need(rc == 0, 'FROZEN_BASE_ANCESTOR')
    if require_remote:
        rc, out, _ = git(root, 'remote', 'get-url', 'origin')
        need(rc == 0 and out.decode().strip() in APPROVED_REMOTES,
             'APPROVED_GITHUB_ORIGIN')
    for args, code in ((('diff','--quiet'),'DIRTY_TRACKED_WORKTREE'),
                       (('diff','--cached','--quiet'),'DIRTY_INDEX')):
        rc, _, _ = git(root,*args)
        need(rc == 0, code)
    rc, out, _ = git(root, 'diff', '--name-status', '--no-renames', '-z', BASE, 'HEAD')
    need(rc == 0 and out, 'RELEASE_ADDITIONS_REQUIRED')
    tokens = out.split(b'\0')
    need(tokens[-1] == b'' and len(tokens) % 2 == 1, 'GIT_DIFF_FORMAT')
    changes = []
    for i in range(0, len(tokens)-1, 2):
        status, path = tokens[i], tokens[i+1]
        need(status == b'A', 'BASELINE_ONLY_ADDITIONS_ALLOWED')
        try:
            filename = path.decode('utf-8',errors='strict')
        except UnicodeDecodeError:
            need(False, 'NONUTF8_RELEASE_PATH')
        need(any(filename.startswith(p) for p in ALLOWED_PREFIXES)
             and '\\' not in filename and '..' not in Path(filename).parts,
             'CHANGED_OUTSIDE_RELEASE_DIR')
        changes.append(filename)
    need(len(changes) == len(set(changes)), 'DUPLICATE_DIFF_PATH')
    for rel in REQUIRED_FILES:
        rc, _, _ = git(root,'cat-file','-e','HEAD:' + rel)
        need(rc == 0 and (root/rel).is_file() and not (root/rel).is_symlink(),
             'REQUIRED_VERSIONED_RELEASE_FILE_' + rel.rsplit('/',1)[-1])
    return {'status':'R8_EXACT_CHECKOUT_SHA_AND_ADDITIVE_DIFF_VERIFIED_NOT_REMOTE_PROOF',
            'frozen_science_commit':BASE,
            'published_release_commit':pinned_release_sha,
            'additions_since_base':len(changes),
            'tracked_source_diff_prohibited':True,
            'frozen_source_bytes_independently_verified':False,
            'remote_publication_independently_verified':False,
            'official_training_authorized':False,
            'gpu_executed':False,
            'kaggle_persistence_verified':False,
            'b_test_access':'NONE'}
