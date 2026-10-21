#!/usr/bin/env python3
"""Synthetic native-hook tests; no GPU, no checkpoint loading or training."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

filename = Path(__file__).with_name('R9_OFFICIAL_ADAPTIVE_BOUNDARY_R4.py')
spec = importlib.util.spec_from_file_location('boundary_r4', filename)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyntheticG4:
    def d4e_recovery_epoch_end(self, epoch, nb, last_opt_step):
        if epoch < 99 and not getattr(self, 'simulate_missing_publication', False):
            self._g4_completed.append({'arm': 'MV00', 'completed_epoch': epoch,
                        'terminal': False, 'manifest_sha256': 'a'*64})
            self._d4e_epoch_receipts.append({'trusted_receipt_sha256': 'b'*64})
        return 'super_hook_completed'


class FakeTrainer(module.R9OfficialAdaptiveBoundaryMixin, SyntheticG4):
    pass


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = [100.0]
        self.worker = FakeTrainer()
        self.worker._d4e_recovery_configured = True
        self.worker._g4_pub_bound = True
        self.worker._d4e_recovery_identity = {'arm':'MV00', 'run_id':'candidate'}
        self.worker._g4_pub_root = Path(self.tmp.name)
        self.worker._g4_completed = []
        self.worker._d4e_epoch_receipts = []
        self.worker._d4e_optimizer_updates = 1
        self.worker._d4_optimizer_steps_observed = 1
        self.worker.ema = SimpleNamespace(updates=1)
        self.worker.scaler = SimpleNamespace(is_enabled=lambda: True,
                                             get_scale=lambda: 1.0)
        self.worker.stop = False
        self.checked = []
        def verify(**kw):
            self.checked.append(kw)
            return {'status':'G4C_LOCAL_VERIFICATION_PASS_ONLY',
                    'completed_epoch':kw['epoch'], 'terminal': kw['epoch']==99}
        self.verify = verify

    def bind(self, **kw):
        self.worker.r9_official_bind_boundary(arm='MV00', target_epoch_one_based=kw.pop('target',3),
             monotonic_deadline_seconds=kw.pop('deadline',10000.),
             minimum_next_epoch_seconds=kw.pop('epoch_seconds',600.),
             reserve_export_seconds=kw.pop('reserve',900.),
             verify_epoch_fn=kw.pop('verifier',self.verify),
             clock=lambda:self.now[0], **kw)

    def call(self, epoch, now=250.):
        self.now[0] = now
        return self.worker.d4e_recovery_epoch_end(epoch, 1, 0)

    def test_can_continue_after_real_ordered_g4(self):
        self.bind(); self.assertEqual(self.call(0), 'super_hook_completed')
        self.assertEqual(self.worker._r9_official_events[0]['reason'], 'CONTINUE')
    def test_stops_after_verified_target_not_before(self):
        self.bind(target=2); self.call(0,200.)
        with self.assertRaises(module.R9OfficialVerifiedPostG4Stop): self.call(1,350.)
        self.assertEqual(self.worker._r9_official_events[-1]['next_epoch_zero_based'],2)
        self.assertEqual(len(self.checked),2)
    def test_budget_stops_after_validated_epoch(self):
        self.bind(target=100,deadline=2000.)
        with self.assertRaises(module.R9OfficialVerifiedPostG4Stop): self.call(0,900.)
        self.assertEqual(self.worker._r9_official_events[-1]['reason'],'INSUFFICIENT_TIME_MARGIN')
    def test_missing_g4_rejected(self):
        self.bind(); self.worker.simulate_missing_publication=True
        with self.assertRaisesRegex(RuntimeError,'MISSING_G4'): self.call(0)
    def test_wrong_arm_rejected(self):
        self.bind(); self.worker._g4_completed.append({'arm':'MV01','completed_epoch':0,
                          'terminal':False,'manifest_sha256':'a'*64})
        self.worker.simulate_missing_publication=True
        with self.assertRaisesRegex(RuntimeError,'G4_ENTRY'): self.call(0)
    def test_early_stop_rejected(self):
        self.bind(); self.worker.stop=True
        with self.assertRaisesRegex(RuntimeError,'UNEXPECTED_EARLY_STOP'): self.call(0)
    def test_ema_mismatch_rejected(self):
        self.bind(); self.worker.ema.updates=99
        with self.assertRaisesRegex(RuntimeError,'OPTIMIZER_EMA'): self.call(0)
    def test_repeated_bind_rejected(self):
        self.bind()
        with self.assertRaisesRegex(RuntimeError,'ALREADY_BOUND'): self.bind()
    def test_bad_deadline_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'TIME_VALUES'): self.bind(deadline=float('nan'))
    def test_insufficient_start_margin_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'INSUFFICIENT_START_MARGIN'):
            self.bind(deadline=800.)
    def test_target100_never_interrupts_final_epoch_end(self):
        self.bind(target=100)
        self.assertEqual(self.call(99, now=500.), 'super_hook_completed')
        self.assertEqual(self.worker._r9_official_last_result,'AWAIT_NATIVE_FINAL_EVAL_AND_TERMINAL_G4')
    def test_target_less_than100_rejects_unexpected_epoch100(self):
        self.bind(target=99)
        with self.assertRaisesRegex(RuntimeError,'UNEXPECTED_TERMINAL_CURSOR'): self.call(99)
    def test_terminal_proof_requires_final_eval(self):
        self.bind(target=100)
        self.worker._g4_completed.append({'arm':'MV00','completed_epoch':99,
                            'terminal':True,'manifest_sha256':'a'*64})
        with self.assertRaisesRegex(RuntimeError,'FINAL_EVAL_NOT_ATTESTED'):
            module.verify_terminal_after_train(trainer=self.worker, verify_epoch_fn=self.verify,arm='MV00')
    def test_terminal_proof_passes_local_only(self):
        self.bind(target=100)
        self.worker._g4_completed.append({'arm':'MV00','completed_epoch':99,
                            'terminal':True,'manifest_sha256':'a'*64})
        self.worker._g4_eval_completed = True
        o = module.verify_terminal_after_train(trainer=self.worker, verify_epoch_fn=self.verify,arm='MV00')
        self.assertEqual(o['status'],'PASS_LOCAL_TERMINAL_G4_ONLY_EXTERNAL_PERSISTENCE_PENDING')
    def test_terminal_wrong_epoch_rejected(self):
        self.bind(target=100)
        self.worker._g4_eval_completed=True
        self.worker._g4_completed.append({'arm':'MV00','completed_epoch':98,
                            'terminal':True,'manifest_sha256':'a'*64})
        with self.assertRaisesRegex(RuntimeError,'TERMINAL_NOT_EPOCH100'):
            module.verify_terminal_after_train(trainer=self.worker,verify_epoch_fn=self.verify,arm='MV00')

if __name__ == '__main__': unittest.main(verbosity=2)
