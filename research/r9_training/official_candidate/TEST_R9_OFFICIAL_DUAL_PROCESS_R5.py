#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest
import R9_OFFICIAL_DUAL_PROCESS_R5 as d

class ProcessTests(unittest.TestCase):
    def test_gpu_mapping(self):
        e=d.make_child_environment('MV00',{'WORLD_SIZE':'2','LOCAL_RANK':'1'})
        self.assertEqual(e['CUDA_VISIBLE_DEVICES'],'0')
        self.assertNotIn('WORLD_SIZE',e)
        self.assertEqual(d.make_child_environment('MV01',{})['CUDA_VISIBLE_DEVICES'],'1')
    def test_duplicate_run_ids(self):
        specs={a:{'run_id':'same','argv':[sys.executable,'-c','pass']} for a in d.ARM_GPU}
        with self.assertRaises(RuntimeError):d.validate_process_descriptors(specs)
    def test_missing_arm(self):
        with self.assertRaises(RuntimeError):d.validate_process_descriptors({'MV00':{'argv':['x','y'],'run_id':'a'}})
    def test_synthetic_both_exit_zero(self):
        specs={a:{'run_id':a,'argv':[sys.executable,'-c',
             'import os;print(os.environ["CUDA_VISIBLE_DEVICES"])']} for a in d.ARM_GPU}
        with tempfile.TemporaryDirectory() as tmp:
            out=d.run_children_cpu_or_separately_authorized(descriptors=specs,
                        log_root=str(Path(tmp)/'logs'),timeout_seconds=5)
            self.assertEqual(out['status'],'TWO_CHILDREN_EXIT_ZERO_UNVERIFIED_APPLICATION_RESULT')
            self.assertFalse(out['gpu_training_verified'])
            self.assertIn('0',(Path(tmp)/'logs'/'MV00.stdout_stderr.log').read_text())
            self.assertIn('1',(Path(tmp)/'logs'/'MV01.stdout_stderr.log').read_text())
    def test_synthetic_child_failure(self):
        specs={'MV00':{'run_id':'aa','argv':[sys.executable,'-c','raise SystemExit(8)']},
               'MV01':{'run_id':'bb','argv':[sys.executable,'-c','import time;time.sleep(2)']}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                d.run_children_cpu_or_separately_authorized(descriptors=specs,
                            log_root=str(Path(tmp)/'logs'),timeout_seconds=5)

if __name__=='__main__':unittest.main(verbosity=2)
