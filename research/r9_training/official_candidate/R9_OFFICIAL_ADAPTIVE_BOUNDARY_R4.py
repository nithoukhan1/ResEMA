#!/usr/bin/env python3
"""R9 official multi-epoch post-G4 boundary adapter, engineering-only R4.

Pure integration candidate: never initiates training, does not modify frozen science,
never asserts remote persistence/authorization. Must be placed before authenticated
G4IntegratedEpochMixin in the MRO and bound only by a separately approved worker.
A safe boundary aborts native train only after post-save/post-capsule G4 verification;
a post-boundary worker must still export ZIP, verify bytes and preserve them externally.
Terminal epoch100 must run original final_eval and terminal G4 publication, and is
intentionally NOT intercepted as a nonterminal safe stop.
"""
from __future__ import annotations
import math
import re
import time
from pathlib import Path
from typing import Callable

HEX = re.compile(r"[0-9a-f]{64}\Z")


class R9OfficialVerifiedPostG4Stop(Exception):
    """A real verified nonterminal native publication boundary (not an offsite receipt)."""


def need(x, why):
    if not x:
        raise RuntimeError("R9_OFFICIAL_BOUNDARY_R4_FAIL_CLOSED_" + why)


class R9OfficialAdaptiveBoundaryMixin:
    """Override the guarded per-epoch hook, AFTER inherited G4 work completes.

    Fixed max target is a session cap (1..100) not a change to epochs=100.
    A target=100 requests the real native terminal path; a quota decision can
    still stop at any earlier authenticated nonterminal G4 boundary.
    """
    def r9_official_bind_boundary(self, *, arm: str, target_epoch_one_based: int,
                                 monotonic_deadline_seconds: float,
                                 minimum_next_epoch_seconds: float,
                                 reserve_export_seconds: float,
                                 verify_epoch_fn: Callable,
                                 epoch_multiplier: float = 1.75,
                                 clock: Callable = time.monotonic):
        need(not getattr(self, "_r9_official_bound", False), "ALREADY_BOUND")
        need(arm in ("MV00", "MV01"), "ARM")
        need(type(target_epoch_one_based) is int and 1 <= target_epoch_one_based <= 100,
             "TARGET_EPOCH")
        need(callable(verify_epoch_fn) and callable(clock), "CALLBACK")
        need(getattr(self, "_d4e_recovery_configured", False), "RECOVERY_NOT_BOUND")
        need(getattr(self, "_g4_pub_bound", False), "G4_NOT_BOUND")
        identity = getattr(self, "_d4e_recovery_identity", None)
        need(isinstance(identity, dict) and identity.get("arm") == arm, "ARM_IDENTITY")
        vals = (monotonic_deadline_seconds, minimum_next_epoch_seconds,
                reserve_export_seconds, epoch_multiplier)
        need(all(type(x) in (int, float) and math.isfinite(x) for x in vals), "TIME_VALUES")
        need(minimum_next_epoch_seconds > 0 and reserve_export_seconds > 0 and
             epoch_multiplier >= 1.0, "TIME_RESERVE")
        now = clock()
        need(type(now) in (float, int) and math.isfinite(now) and
             monotonic_deadline_seconds > now + reserve_export_seconds,
             "INSUFFICIENT_START_MARGIN")
        pub = Path(self._g4_pub_root)
        need(pub.is_dir() and not pub.is_symlink(), "PUB_ROOT")
        self._r9_official_arm = arm
        self._r9_official_target = target_epoch_one_based
        self._r9_official_deadline = monotonic_deadline_seconds
        self._r9_official_min_epoch_seconds = minimum_next_epoch_seconds
        self._r9_official_export_reserve = reserve_export_seconds
        self._r9_official_multiplier = epoch_multiplier
        self._r9_official_verify_epoch = verify_epoch_fn
        self._r9_official_clock = clock
        self._r9_official_epoch_start = now
        self._r9_official_events = []
        self._r9_official_last_result = None
        self._r9_official_bound = True

    def d4e_recovery_epoch_end(self, epoch: int, nb: int, last_opt_step: int):
        need(getattr(self, "_r9_official_bound", False), "NOT_BOUND")
        result = super().d4e_recovery_epoch_end(epoch, nb, last_opt_step)
        need(type(epoch) is int and 0 <= epoch < 100, "EPOCH_RANGE")
        if epoch == 99:
            # The verified G4 implementation performs terminal publication
            # *after* native final_eval; there is no resumable epoch100 capsule.
            need(self._r9_official_target == 100, "UNEXPECTED_TERMINAL_CURSOR")
            self._r9_official_last_result = "AWAIT_NATIVE_FINAL_EVAL_AND_TERMINAL_G4"
            return result
        need(not getattr(self, "stop", False), "UNEXPECTED_EARLY_STOP")
        entries = getattr(self, "_g4_completed", None)
        need(isinstance(entries, list) and entries, "MISSING_G4")
        last = entries[-1]
        pin = last.get("manifest_sha256")
        need(last.get("arm") == self._r9_official_arm and
             last.get("completed_epoch") == epoch and last.get("terminal") is False and
             type(pin) is str and HEX.fullmatch(pin) is not None, "G4_ENTRY")
        checked = self._r9_official_verify_epoch(root=self._g4_pub_root,
                            arm=self._r9_official_arm, epoch=epoch,
                            external_manifest_sha256=pin)
        need(checked.get("status") == "G4C_LOCAL_VERIFICATION_PASS_ONLY" and
             checked.get("completed_epoch") == epoch and
             checked.get("terminal") is False, "G4_BYTES")
        receipts = getattr(self, "_d4e_epoch_receipts", None)
        need(isinstance(receipts, list) and receipts and
             type(receipts[-1].get("trusted_receipt_sha256")) is str and
             HEX.fullmatch(receipts[-1]["trusted_receipt_sha256"]) is not None,
             "TRUSTED_CAPSULE_RECEIPT")
        updates = getattr(self, "_d4e_optimizer_updates", None)
        need(type(updates) is int and updates > 0 and
             getattr(self, "_d4_optimizer_steps_observed", None) == updates and
             getattr(getattr(self, "ema", None), "updates", None) == updates,
             "OPTIMIZER_EMA")
        scaler = getattr(self, "scaler", None)
        need(scaler is not None and scaler.is_enabled() and
             float(scaler.get_scale()) >= 1.0, "AMP_SCALER")
        now = self._r9_official_clock()
        need(type(now) in (float, int) and math.isfinite(now) and
             now >= self._r9_official_epoch_start, "MONOTONIC_CLOCK")
        actual = max(now - self._r9_official_epoch_start, 0.001)
        needed = max(actual * self._r9_official_multiplier,
                     self._r9_official_min_epoch_seconds) + self._r9_official_export_reserve
        remaining = self._r9_official_deadline - now
        manual = (epoch + 1 == self._r9_official_target)
        budget = remaining <= needed
        event = {
            "arm": self._r9_official_arm, "completed_epoch_zero_based": epoch,
            "next_epoch_zero_based": epoch + 1, "g4_manifest_sha256": pin,
            "local_g4_verified": True, "terminal": False,
            "exported_off_session": False, "official_training_authorized_here": False,
            "remaining_seconds": round(remaining, 3),
            "required_next_epoch_and_export_seconds": round(needed, 3),
            "reason": "MANUAL_SESSION_BOUNDARY" if manual else
                      "INSUFFICIENT_TIME_MARGIN" if budget else "CONTINUE"
        }
        self._r9_official_events.append(event)
        self._r9_official_epoch_start = now
        if manual or budget:
            self._r9_official_last_result = "NONTERMINAL_LOCAL_G4_VERIFIED_EXPORT_PENDING"
            raise R9OfficialVerifiedPostG4Stop(event["reason"])
        self._r9_official_last_result = "CONTINUE_ONE_EPOCH"
        return result


def verify_terminal_after_train(*, trainer, verify_epoch_fn, arm: str):
    """Pure native-return gate; requires terminal entry exactly at zero-based 99.

    Not evidence of cross-session transport or of a completed test set evaluation.
    """
    need(getattr(trainer, "_r9_official_bound", False), "TERMINAL_UNBOUND")
    need(getattr(trainer, "_r9_official_target", None) == 100, "TERMINAL_TARGET")
    need(getattr(trainer, "_g4_eval_completed", False), "FINAL_EVAL_NOT_ATTESTED")
    items = getattr(trainer, "_g4_completed", None)
    need(isinstance(items, list) and items, "TERMINAL_G4_MISSING")
    last = items[-1]
    pin = last.get("manifest_sha256")
    need(last.get("arm") == arm and last.get("completed_epoch") == 99 and
         last.get("terminal") is True and type(pin) is str and
         HEX.fullmatch(pin) is not None, "TERMINAL_NOT_EPOCH100")
    checked = verify_epoch_fn(root=trainer._g4_pub_root, arm=arm, epoch=99,
                              external_manifest_sha256=pin)
    need(checked.get("status") == "G4C_LOCAL_VERIFICATION_PASS_ONLY" and
         checked.get("completed_epoch") == 99 and checked.get("terminal") is True,
         "TERMINAL_BYTES")
    return {"status": "PASS_LOCAL_TERMINAL_G4_ONLY_EXTERNAL_PERSISTENCE_PENDING",
            "completed_epoch_zero_based": 99, "terminal": True,
            "g4_manifest_sha256": pin, "official_training_authorized_here": False,
            "b_test_access": "NONE"}
