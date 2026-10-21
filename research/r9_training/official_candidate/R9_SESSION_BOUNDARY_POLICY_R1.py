#!/usr/bin/env python3
"""Pure policy for a future *post-G4* official native epoch-boundary stop.

NOT an integrated trainer mixin, not a system-clock timer, and never interrupts
mid-epoch. This function must only be called AFTER authenticated G4 publication.
It returns conservative advice; separate integration and native tests required.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundaryResult:
    action: str
    reason: str
    next_epoch_one_based: int | None
    remaining_seconds: float
    required_seconds: float


def decide_at_verified_boundary(*, completed_epoch_zero_based: int,
                                g4_publication_verified: bool,
                                g4_manifest_sha256: str,
                                terminal: bool,
                                stop_flag: bool,
                                seconds_until_hard_deadline: float,
                                last_completed_epoch_seconds: float,
                                conservative_next_epoch_seconds: float,
                                export_and_save_reserve_seconds: float,
                                epoch_time_multiplier: float = 1.5,
                                requested_boundary_epoch_one_based: int | None = None):
    """Return STOP/CAN_CONTINUE/TERMINAL_FINALIZE_REQUIRED; never perform I/O.

    Stops only *after* verified nonterminal G4; incomplete training cannot be
    labeled an authorized checkpoint. Deadline must include notebook output
    retention/remote-publication overhead rather than only GPU time.
    """
    if (type(completed_epoch_zero_based) is not int or
            not 0 <= completed_epoch_zero_based <= 99 or
            type(g4_publication_verified) is not bool or
            type(terminal) is not bool or type(stop_flag) is not bool):
        raise RuntimeError('R9_POLICY_BAD_EPOCH_OR_STATE')
    if not isinstance(g4_manifest_sha256, str) or len(g4_manifest_sha256) != 64 or any(
            c not in '0123456789abcdef' for c in g4_manifest_sha256):
        raise RuntimeError('R9_POLICY_UNPINNED_G4_MANIFEST')
    if not g4_publication_verified:
        raise RuntimeError('R9_POLICY_UNVERIFIED_PUBLICATION')
    times=(seconds_until_hard_deadline,last_completed_epoch_seconds,
           conservative_next_epoch_seconds,export_and_save_reserve_seconds,epoch_time_multiplier)
    if not all(isinstance(t,(float,int)) and not isinstance(t,bool) and math.isfinite(t) for t in times):
        raise RuntimeError('R9_POLICY_TIME_INVALID')
    if (last_completed_epoch_seconds<=0 or conservative_next_epoch_seconds<=0 or
            export_and_save_reserve_seconds<=0 or epoch_time_multiplier<1):
        raise RuntimeError('R9_POLICY_RESERVE_INVALID')
    if (requested_boundary_epoch_one_based is not None and
        (type(requested_boundary_epoch_one_based) is not int or not 1<=requested_boundary_epoch_one_based<=99)):
        raise RuntimeError('R9_POLICY_MANUAL_BOUNDARY_INVALID')
    if terminal:
        if completed_epoch_zero_based != 99:
            raise RuntimeError('R9_POLICY_EARLY_TERMINAL_IS_FAILURE_FOR_100_EPOCH_PLAN')
        if not stop_flag:
            raise RuntimeError('R9_POLICY_TERMINAL_STOP_FLAG_MISSING')
        return BoundaryResult('TERMINAL_FINALIZE_REQUIRED','EXACTLY_EPOCH100',None,
                              seconds_until_hard_deadline,export_and_save_reserve_seconds)
    if completed_epoch_zero_based == 99:
        raise RuntimeError('R9_POLICY_EPOCH100_MUST_BE_TERMINAL')
    if stop_flag:
        raise RuntimeError('R9_POLICY_NATIVE_EARLY_STOP_NOT_RESUMABLE')
    needed=max(last_completed_epoch_seconds*epoch_time_multiplier,
               conservative_next_epoch_seconds)+export_and_save_reserve_seconds
    if (requested_boundary_epoch_one_based == completed_epoch_zero_based+1 or
            seconds_until_hard_deadline <= needed):
        why=('MANUALLY_PINNED_EPOCH_BOUNDARY'
             if requested_boundary_epoch_one_based == completed_epoch_zero_based+1
             else 'INSUFFICIENT_DEADLINE_MARGIN')
        return BoundaryResult('STOP_AFTER_VERIFIED_G4',why,completed_epoch_zero_based+2,
                              seconds_until_hard_deadline,needed)
    return BoundaryResult('CONTINUE_ONE_MORE_EPOCH','WITHIN_CONSERVATIVE_BUDGET',
                          completed_epoch_zero_based+2,seconds_until_hard_deadline,needed)
