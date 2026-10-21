#!/usr/bin/env python3
import json,tempfile,unittest
from pathlib import Path
import R9_OFFICIAL_R7_ACCEPTANCE_EXPORT as m

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.d=Path(self.tmp.name);self.proofs={}
  for arm,n in (('MV00',0),('MV01',1)):
   f=self.d/arm;f.mkdir();z=f/f'{arm}_epoch{n+1:03d}_G4_LOCAL_EXPORT.zip'
   z.write_bytes((arm+'-bytes').encode());sha=m.hash_file(z)
   sc=Path(str(z)+'.sha256');sc.write_text(f'{sha}  {z.name}\n')
   rp=f/'R7_NEWPATH_ACCEPTANCE_REPORT.json'
   info={'schema':m.SCHEMA,'status':'NONOFFICIAL_NEWPATH_GPU_LOCAL_G4_VERIFIED_EXTERNAL_NOT_VERIFIED',
         'arm':arm,'run_id':'R7_RUN_'+arm,'completed_epoch_zero_based':n,
         'terminal':False,'official_training_authorized':False,
         'independent_offsite_verified':False,'b_test_access':'NONE',
         'local_zip':str(z),'local_zip_sha256':sha}
   rp.write_text(json.dumps(info))
   self.proofs[arm]={'report':str(rp),'zip':str(z),'sidecar':str(sc)}
 def inspect(self,path,sidecar):
  arm=Path(path).name[:4]
  epoch=int(Path(path).name.split('_epoch')[1][:3])-1
  return {'status':'PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION',
          'arm':arm,'completed_epoch':epoch,'terminal':False,'zip_sha256':m.hash_file(path)}
 def run_stage(self,proofs=None,max_bytes=100000):
  return m.stage_two(proofs=self.proofs if proofs is None else proofs,
      root=self.d/'stage',inspector=self.inspect,max_bytes=max_bytes)
 def mutate(self,arm,key,val):
  p=Path(self.proofs[arm]['report']);d=json.loads(p.read_text());d[key]=val;p.write_text(json.dumps(d))
 def test_success(self):
  x=self.run_stage();self.assertFalse(x['offsite_verified']);self.assertEqual(len(x['arms']),2)
 def test_zipblob_identical(self):
  self.run_stage()
  for arm in m.ARMS:
   src=Path(self.proofs[arm]['zip']);copy=self.d/'stage'/arm/(src.name+'.zipblob')
   self.assertEqual(m.hash_file(src),m.hash_file(copy))
 def test_no_overwrite(self):
  self.run_stage()
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_one_arm_not_allowed(self):
  with self.assertRaises(RuntimeError):self.run_stage({'MV00':self.proofs['MV00']})
 def test_budget(self):
  with self.assertRaises(RuntimeError):self.run_stage(max_bytes=10)
 def test_report_fake_offsite(self):
  self.mutate('MV00','independent_offsite_verified',True)
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_report_fake_authority(self):
  self.mutate('MV00','official_training_authorized',True)
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_wrong_cursor(self):
  self.mutate('MV01','completed_epoch_zero_based',2)
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_wrong_sha(self):
  self.mutate('MV00','local_zip_sha256','0'*64)
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_bad_sidecar(self):
  Path(self.proofs['MV00']['sidecar']).write_text('0'*64+'  MV00_epoch001_G4_LOCAL_EXPORT.zip\n')
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_wrong_inspector(self):
  with self.assertRaises(RuntimeError):m.stage_two(proofs=self.proofs,
    root=self.d/'stage',inspector=lambda *a,**kw:{'status':'FAIL'},max_bytes=100000)
 def test_early_terminal(self):
  self.mutate('MV01','terminal',True)
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_btest(self):
  self.mutate('MV00','b_test_access','READ')
  with self.assertRaises(RuntimeError):self.run_stage()
 def test_no_report(self):
  Path(self.proofs['MV00']['report']).unlink()
  with self.assertRaises(RuntimeError):self.run_stage()

if __name__=='__main__':unittest.main(verbosity=2)
