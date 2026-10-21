#!/usr/bin/env python3
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import R9_OFFICIAL_SESSION_EXPORT_INDEX_R6 as e
import R9_OFFICIAL_DUAL_RECONCILE_R6 as d

class Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.sources={}
        for arm,n in [('MV00',3),('MV01',5)]:
            folder=self.root/arm;folder.mkdir()
            name=f'{arm}_epoch{n+1:03d}_G4_LOCAL_EXPORT.zip'
            z=folder/name;z.write_bytes((arm+'-checkpoint').encode())
            sha=e.digest(z)
            sc=folder/(name+'.sha256');sc.write_text(f'{sha}  {name}\n')
            report={'schema':'DETECTION_CHAT25_R6_LOCAL_NATIVE_SESSION_RESULT',
                'status':'GPU_EXECUTION_LOCAL_BYTES_ONLY_OFFSITE_PENDING',
                'arm':arm,'run_id':'R6TRIAL_'+arm,'completed_epoch_zero_based':n,
                'terminal':False,'local_zip':str(z),'local_zip_sha256':sha,
                'b_test_access':'NONE','independent_offsite_verified':False}
            rp=folder/'R6_NATIVE_SESSION_REPORT.json';rp.write_text(json.dumps(report))
            self.sources[arm]={'report':str(rp),'zip':str(z),'sidecar':str(sc)}
    def inspector(self,path,sidecar):
        s=str(path)
        arm='MV00' if 'MV00' in s else 'MV01'
        name=Path(path).name
        epoch=int(name.split('_epoch')[1][:3])-1
        return {'status':'PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION',
                'arm':arm,'completed_epoch':epoch,'terminal':False,'zip_sha256':e.digest(path)}
    def test_both_staged_bytes(self):
        result=e.create_output_index(proof_paths=self.sources,
                 output_root=self.root/'outputs',inspector=self.inspector)
        self.assertEqual(result['status'],'LOCAL_COPY_VERIFIED_EXTERNAL_PERSISTENCE_PENDING')
        self.assertFalse(result['offsite_persistence_verified'])
        for arm in ('MV00','MV01'):
            src=Path(self.sources[arm]['zip'])
            copied=self.root/'outputs'/arm/(src.name+'.zipblob')
            self.assertEqual(e.digest(src),e.digest(copied))
    def test_no_overwrite(self):
        (self.root/'outputs').mkdir()
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths=self.sources,
            output_root=self.root/'outputs',inspector=self.inspector)
    def test_reject_budget(self):
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths=self.sources,
            output_root=self.root/'outputs',inspector=self.inspector,max_bytes=4)
    def test_reject_only_one_arm(self):
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths={'MV00':self.sources['MV00']},
            output_root=self.root/'outputs',inspector=self.inspector)
    def test_reject_corrupt_report_hash(self):
        p=Path(self.sources['MV00']['report']);v=json.loads(p.read_text());v['local_zip_sha256']='0'*64;p.write_text(json.dumps(v))
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths=self.sources,
            output_root=self.root/'outputs',inspector=self.inspector)
    def test_reject_fake_offsite_verified(self):
        p=Path(self.sources['MV00']['report']);v=json.loads(p.read_text());v['independent_offsite_verified']=True;p.write_text(json.dumps(v))
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths=self.sources,
            output_root=self.root/'outputs',inspector=self.inspector)
    def test_reject_bad_inspector(self):
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths=self.sources,
            output_root=self.root/'outputs',inspector=lambda *a,**k:{'status':'FAIL'})
    def test_reject_test_access(self):
        p=Path(self.sources['MV01']['report']);v=json.loads(p.read_text());v['b_test_access']='READ';p.write_text(json.dumps(v))
        with self.assertRaises(RuntimeError):e.create_output_index(proof_paths=self.sources,
            output_root=self.root/'outputs',inspector=self.inspector)
    def records(self):
        return {arm:json.loads(Path(v['report']).read_text()) for arm,v in self.sources.items()}
    def test_independent_arm_progress(self):
        x=d.reconcile(self.records(),'R6TRIAL')
        self.assertEqual(x['arms']['MV00']['next_epoch_zero_based'],4)
        self.assertEqual(x['arms']['MV01']['next_epoch_zero_based'],6)
    def test_terminal_100_one_arm(self):
        x=self.records();x['MV01']['completed_epoch_zero_based']=99;x['MV01']['terminal']=True
        y=d.reconcile(x,'R6TRIAL');self.assertTrue(y['arms']['MV01']['terminal'])
    def test_early_terminal_rejected(self):
        x=self.records();x['MV01']['terminal']=True
        with self.assertRaises(RuntimeError):d.reconcile(x,'R6TRIAL')
    def test_ninety_nine_not_terminal_rejected(self):
        x=self.records();x['MV01']['completed_epoch_zero_based']=99
        with self.assertRaises(RuntimeError):d.reconcile(x,'R6TRIAL')
    def test_run_prefix_rejected(self):
        with self.assertRaises(RuntimeError):d.reconcile(self.records(),'OTHER')
    def test_one_arm_missing_rejected(self):
        x=self.records();x.pop('MV00')
        with self.assertRaises(RuntimeError):d.reconcile(x,'R6TRIAL')
    def test_wrong_arm_mismatch(self):
        x=self.records();x['MV01']['arm']='MV00'
        with self.assertRaises(RuntimeError):d.reconcile(x,'R6TRIAL')

if __name__=='__main__':unittest.main(verbosity=2)
