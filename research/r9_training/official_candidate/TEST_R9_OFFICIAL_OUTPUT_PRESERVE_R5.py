#!/usr/bin/env python3
import hashlib
from pathlib import Path
import tempfile
import unittest
import R9_OFFICIAL_OUTPUT_PRESERVE_R5 as p

class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.original=self.root/'original';self.original.mkdir()
        self.staging=self.root/'staging';self.staging.mkdir()
        self.zip=self.original/'MV00_epoch003_G4_LOCAL_EXPORT.zip'
        self.zip.write_bytes(b'synthetic NOT genuine model bytes')
        self.sha=hashlib.sha256(self.zip.read_bytes()).hexdigest()
        self.side=Path(str(self.zip)+'.sha256')
        self.side.write_text(f'{self.sha}  {self.zip.name}\n')
        self.dest=self.staging/(self.zip.name+'.zipblob')
    def invoke(self,arm='MV00'):
        return p.stage_original_zip(arm=arm,original_zip=self.zip,sidecar=self.side,destination=self.dest)
    def test_exact_preservation(self):
        result=self.invoke()
        self.assertEqual(self.sha,result['zip_sha256']);self.assertEqual(self.zip.read_bytes(),self.dest.read_bytes())
        self.assertFalse(result['offsite_verified']);self.assertTrue(self.zip.exists())
    def test_reject_no_overwrite(self):
        self.invoke()
        with self.assertRaises(RuntimeError):self.invoke()
    def test_reject_wrong_sha(self):
        self.side.write_text('0'*64+'  '+self.zip.name)
        with self.assertRaises(RuntimeError):self.invoke()
    def test_reject_wrong_sidecar_name(self):
        pth=self.original/'fake.sha256';pth.write_text(self.side.read_text())
        with self.assertRaises(RuntimeError):
            p.stage_original_zip(arm='MV00',original_zip=self.zip,sidecar=pth,destination=self.dest)
    def test_reject_wrong_arm(self):
        with self.assertRaises(RuntimeError):self.invoke(arm='MV02')
    def test_reject_same_folder(self):
        dst=self.original/(self.zip.name+'.zipblob')
        with self.assertRaises(RuntimeError):
            p.stage_original_zip(arm='MV00',original_zip=self.zip,sidecar=self.side,destination=dst)

if __name__=='__main__':unittest.main(verbosity=2)
