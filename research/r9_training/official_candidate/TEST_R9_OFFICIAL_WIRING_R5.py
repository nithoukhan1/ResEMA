#!/usr/bin/env python3
import json
import unittest
from pathlib import Path
import R9_OFFICIAL_NATIVE_WIRING_R5 as w


class DummyAdapter:
    def r9_official_bind_boundary(self): pass
DummyAdapter.__name__='R9OfficialAdaptiveBoundaryMixin'

class NativeMethods:
    def d4e_recovery_epoch_end(self): pass
    def d4e_configure_recovery(self): pass
    def d4e_g3b5_bind_guard(self): pass
    def g4_bind_epoch_publication(self): pass
    def final_eval(self): pass
    def d4e_g3b_guard_terminal(self): pass

Native0=type('EarlyPairedTrainer',(NativeMethods,),{})
Native1=type('VGRADetectionTrainer',(NativeMethods,),{})

class G4: pass
G4.__name__='G4IntegratedEpochMixin'
class Guard: pass
Guard.__name__='G3B5PrivacySafeGuardMixin'


def binder(native): return type(native.__name__, (G4,Guard,native),{})

class TestR5Wiring(unittest.TestCase):
    def test_mv00_order(self):
        c=w.compose_native_candidate(Native0,binder,DummyAdapter)
        self.assertEqual(c.__name__,'EarlyPairedTrainer')
        names=[x.__name__ for x in c.__mro__]
        self.assertLess(names.index('R9OfficialAdaptiveBoundaryMixin'),names.index('G4IntegratedEpochMixin'))
    def test_mv01_order(self):
        self.assertEqual(w.compose_native_candidate(Native1,binder,DummyAdapter).__name__,'VGRADetectionTrainer')
    def test_no_unknown_native(self):
        with self.assertRaises(w.R5WiringError):w.compose_native_candidate(type('Unknown',(),{}),binder,DummyAdapter)
    def test_no_old_bound_wrapper(self):
        old=type('R9BPostPublicationSessionStopMixin',(),{})
        def bad(native):return type(native.__name__,(old,G4,Guard,native),{})
        with self.assertRaises(w.R5WiringError):w.compose_native_candidate(Native0,bad,DummyAdapter)
    def test_no_double_adaptive(self):
        def bad(native):return type(native.__name__,(DummyAdapter,G4,Guard,native),{})
        with self.assertRaises(w.R5WiringError):w.compose_native_candidate(Native0,bad,DummyAdapter)
    def test_no_g4(self):
        with self.assertRaises(w.R5WiringError):w.compose_native_candidate(Native0,lambda x:x,DummyAdapter)
    def test_missing_native_methods(self):
        bad=type('EarlyPairedTrainer',(),{})
        with self.assertRaises(w.R5WiringError):w.compose_native_candidate(bad,binder,DummyAdapter)
    def test_release_record_remains_disabled(self):
        base=json.loads(Path(__file__).with_name('R9_OFFICIAL_RELEASE_CONTRACT_R1.json').read_text())
        review=json.loads(Path(__file__).with_name('R9_OFFICIAL_R5_RELEASE_RECORD.json').read_text())
        self.assertEqual(w.verify_external_release_boundary(original_contract=base,new_release=review)['status'],'PASS_REVIEW_WIRING_ONLY')
    def test_forged_authorization_rejected(self):
        base=json.loads(Path(__file__).with_name('R9_OFFICIAL_RELEASE_CONTRACT_R1.json').read_text())
        review=json.loads(Path(__file__).with_name('R9_OFFICIAL_R5_RELEASE_RECORD.json').read_text())
        review['gpu_execution_allowed']=True
        with self.assertRaises(w.R5WiringError): w.verify_external_release_boundary(original_contract=base,new_release=review)

if __name__=='__main__':unittest.main(verbosity=2)
