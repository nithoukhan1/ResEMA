import unittest
from R9_SESSION_BOUNDARY_POLICY_R1 import decide_at_verified_boundary as decide

BASE=dict(completed_epoch_zero_based=0,g4_publication_verified=True,
          g4_manifest_sha256='a'*64,terminal=False,stop_flag=False,
          seconds_until_hard_deadline=12000,last_completed_epoch_seconds=1600,
          conservative_next_epoch_seconds=1800,export_and_save_reserve_seconds=900)

class Tests(unittest.TestCase):
    def choose(self, **kwargs):
        return decide(**{**BASE,**kwargs})
    def test_continue(self):
        self.assertEqual(self.choose().action,'CONTINUE_ONE_MORE_EPOCH')
    def test_safe_stop_for_short_time(self):
        s=self.choose(seconds_until_hard_deadline=3200)
        self.assertEqual(s.action,'STOP_AFTER_VERIFIED_G4')
        self.assertEqual(s.next_epoch_one_based,2)
    def test_manual_stop(self):
        self.assertEqual(self.choose(requested_boundary_epoch_one_based=1).action,
                         'STOP_AFTER_VERIFIED_G4')
    def test_stop_only_after_g4(self):
        with self.assertRaises(RuntimeError):self.choose(g4_publication_verified=False)
    def test_stop_rejects_early_termination(self):
        with self.assertRaises(RuntimeError):self.choose(stop_flag=True)
    def test_last_epoch_must_be_terminal(self):
        with self.assertRaises(RuntimeError):self.choose(completed_epoch_zero_based=99)
    def test_exact_terminal(self):
        s=self.choose(completed_epoch_zero_based=99,terminal=True,stop_flag=True)
        self.assertEqual(s.action,'TERMINAL_FINALIZE_REQUIRED')
    def test_premature_terminal_rejected(self):
        with self.assertRaises(RuntimeError):self.choose(terminal=True,stop_flag=True)
    def test_bad_manifest(self):
        with self.assertRaises(RuntimeError):self.choose(g4_manifest_sha256='bad')
    def test_nonfinite_deadline(self):
        with self.assertRaises(RuntimeError):self.choose(seconds_until_hard_deadline=float('nan'))
    def test_invalid_manual_boundary(self):
        with self.assertRaises(RuntimeError):self.choose(requested_boundary_epoch_one_based=100)
    def test_exact_budget_stop(self):
        self.assertEqual(self.choose(seconds_until_hard_deadline=3300).action,
                         'STOP_AFTER_VERIFIED_G4')

if __name__=='__main__':unittest.main(verbosity=2)
