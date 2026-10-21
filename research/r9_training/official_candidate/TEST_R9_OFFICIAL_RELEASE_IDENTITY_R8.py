#!/usr/bin/env python3
"""CPU-only real local Git regression tests; no user repo access or network."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import R9_OFFICIAL_RELEASE_IDENTITY_R8 as m


def run(repo, *args):
    p = subprocess.run(['git',*args],cwd=repo,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError(f'git {args}: {p.stderr}')
    return p.stdout.strip()

class Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)/'repo';self.root.mkdir()
        run(self.root,'init','-q');run(self.root,'config','user.email','local@example.invalid')
        run(self.root,'config','user.name','Local Test')
        (self.root/'README.md').write_text('pinned science\n')
        run(self.root,'add','README.md');run(self.root,'commit','-qm','baseline')
        self.base=run(self.root,'rev-parse','HEAD')
        for rel in m.REQUIRED_FILES:
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('# review candidate\n')
        run(self.root,'add','.');run(self.root,'commit','-qm','review release')
        self.pin=run(self.root,'rev-parse','HEAD')
        run(self.root,'remote','add','origin','https://github.com/nithoukhan1/ResEMA.git')
        self.ctx=patch.object(m,'BASE',self.base);self.ctx.start();self.addCleanup(self.ctx.stop)
    def checked(self):return m.validate_checkout(self.root,self.pin)
    def test_exact_pinned_valid_additions(self):
        r=self.checked();self.assertFalse(r['official_training_authorized'])
        self.assertFalse(r['frozen_source_bytes_independently_verified'])
        self.assertEqual(r['published_release_commit'],self.pin)
    def test_frozen_sha_as_release_forbidden(self):
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root,self.base)
    def test_truncated_pin_rejected(self):
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root,self.pin[:7])
    def test_uppercase_pin_rejected(self):
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root,self.pin.upper())
    def test_wrong_sha_rejected(self):
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root,'a'*40)
    def test_mutated_tracked_release_rejected(self):
        (self.root/m.REQUIRED_FILES[0]).write_text('local changed!')
        with self.assertRaises(RuntimeError):self.checked()
    def test_staged_change_rejected(self):
        (self.root/'README.md').write_text('tamper')
        run(self.root,'add','README.md')
        with self.assertRaises(RuntimeError):self.checked()
    def test_wrong_remote_rejected(self):
        run(self.root,'remote','set-url','origin','https://invalid.example/repo')
        with self.assertRaises(RuntimeError):self.checked()
    def test_missing_origin_rejected(self):
        run(self.root,'remote','remove','origin')
        with self.assertRaises(RuntimeError):self.checked()
    def test_changes_to_science_after_base_rejected(self):
        (self.root/'README.md').write_text('changed scientific source')
        run(self.root,'add','README.md');run(self.root,'commit','-qm','unapproved science change')
        pin=run(self.root,'rev-parse','HEAD')
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root,pin)
    def test_prior_intermediate_edit_still_release_addition(self):
        f=self.root/m.REQUIRED_FILES[0];f.write_text('alter existing release')
        run(self.root,'add','.');run(self.root,'commit','-qm','modify candidate')
        pin=run(self.root,'rev-parse','HEAD')
        # Baseline -> HEAD still only additions, so this is allowed and pinned.
        self.assertEqual(m.validate_checkout(self.root,pin)['published_release_commit'],pin)
    def test_reused_other_git_root_rejected(self):
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root/'research',self.pin)
    def test_non_descendant_commit_rejected(self):
        with patch.object(m,'BASE','f'*40):
            with self.assertRaises(RuntimeError):self.checked()
    def test_missing_required_files_rejected(self):
        (self.root/m.REQUIRED_FILES[1]).unlink()
        run(self.root,'add','-A');run(self.root,'commit','-qm','remove worker')
        pin=run(self.root,'rev-parse','HEAD')
        with self.assertRaises(RuntimeError):m.validate_checkout(self.root,pin)
    def test_without_repo_does_not_pass(self):
        with self.assertRaises(Exception):m.validate_checkout(Path(self.t.name)/'missing',self.pin)

if __name__=='__main__':unittest.main(verbosity=2)
