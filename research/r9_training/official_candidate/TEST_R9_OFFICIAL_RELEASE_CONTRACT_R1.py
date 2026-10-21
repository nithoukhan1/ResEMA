#!/usr/bin/env python3
"""Only CPU/static negative tests. Not a native trainer test."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from R9_OFFICIAL_RELEASE_PREFLIGHT_R1 import check_contract, need

HERE = Path(__file__).resolve().parent
BASE = json.loads((HERE/'R9_OFFICIAL_RELEASE_CONTRACT_R1.json').read_text())

class ContractTests(unittest.TestCase):
    def test_original_candidate_valid_and_nonexecutable(self):
        out = check_contract(BASE)
        self.assertEqual(out['status'], 'PASS_REVIEW_CONTRACT_ONLY_NOT_AUTHORIZED')
        self.assertIs(out['official_training_authorized'], False)
    def mutate(self, path, value):
        data=copy.deepcopy(BASE)
        ptr=data
        for key in path[:-1]: ptr=ptr[key]
        ptr[path[-1]]=value
        with self.assertRaises(RuntimeError): check_contract(data)
    def test_authorized_true_is_rejected(self):
        self.mutate(['official_training_authorized'], True)
    def test_executable_true_is_rejected(self):
        self.mutate(['run_executable'], True)
    def test_test_unsealing_rejected(self):
        self.mutate(['experiment','test_split'], 'B-TEST')
    def test_rehearsal_start_is_rejected(self):
        self.mutate(['experiment','initialize_from'], 'R9_EPOCH2_REHEARSAL')
    def test_epoch_change_rejected(self):
        self.mutate(['experiment','epochs_per_arm'], 99)
    def test_gpu_swap_rejected(self):
        self.mutate(['experiment','physical_gpu'], {'MV00':1,'MV01':0})
    def test_source_commit_change_rejected(self):
        self.mutate(['source','baseline_git_commit'], 'a'*40)
    def test_removed_terminal_protection_rejected(self):
        self.mutate(['release_gates','terminal_epoch100_requires_separate_native_proof'], False)
    def test_unknown_key_rejected(self):
        changed=copy.deepcopy(BASE);changed['unapproved_extra']='x'
        with self.assertRaises(RuntimeError):check_contract(changed)

if __name__=='__main__':unittest.main(verbosity=2)
