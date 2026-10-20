"""PRIVATE-STAGE CANDIDATE ONLY: enforce independently computed R0 file identity.

This intentionally does not authorize a GPU run, Kaggle source mutation, or 100 epochs.
R3 cannot proceed until external runtime-YAML provenance, source staging, capsule/
best.pt transfer and actual GPU comparison receive independent acceptance.

Install beside R2.1 recovery and R0 verification modules in a *disposable private*
source stage, never the scientific checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

from .d4e_recovery_integration import D4ERecoveryTrainerMixin
from .d4e_r3_read_only_identity_preflight import (
    safe_file, verify_inputs,
)

ACCEPTED_R0_FILE_CHECK = 'PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE'


class D4EVerifiedRecoveryTrainerMixin(D4ERecoveryTrainerMixin):
    """Replace caller-made identity with direct pinned file-byte derivation.

    A successful call only configures recovery state. It is not authority to
    execute R3 or full training. The separately approved YAML hash must come
    from reviewed evidence, not be self-certified by this interface.
    """

    def d4e_configure_recovery(
        self, *, mode: str, config_path: str | Path, recipe_path: str | Path,
        approved_runtime_yaml_sha256: str, arm: str, run_id: str,
        checkpoint_root: str | Path,
        checkpoint_path: str | Path | None = None,
        trusted_receipt_sha256: str | None = None,
    ) -> None:
        if getattr(self, '_d4e_recovery_configured', False):
            raise RuntimeError('R3_ALREADY_CONFIGURED')
        expected = ('MV00' if type(self).__name__ == 'EarlyPairedTrainer'
                    else 'MV01' if type(self).__name__ == 'VGRADetectionTrainer'
                    else None)
        if expected is None or arm != expected:
            raise ValueError('R3_TRAINER_ARM_MISMATCH_BEFORE_IDENTITY_HASHING')
        # No identity or hash dictionary is accepted from the caller.
        config = json.loads(safe_file(str(config_path)).read_text(encoding='utf-8'))
        record = verify_inputs(
            config, arm=arm, run_id=run_id,
            approved_yaml_sha=approved_runtime_yaml_sha256,
            recipe_path=str(recipe_path),
        )
        if record.get('status') != ACCEPTED_R0_FILE_CHECK:
            raise ValueError('R3_REQUIRE_AUTHENTIC_REAL_PINNED_FILE_VERIFICATION')
        identity = record['identity']
        if identity['arm'] != arm or identity['run_id'] != run_id:
            raise ValueError('R3_VERIFIED_IDENTITY_ARM_OR_RUN_MISMATCH')
        # No config state is committed until the inherited R2.1 recipe/mode/
        # checkpoint guards accept this independently DERIVED identity.
        super().d4e_configure_recovery(
            mode=mode, identity=identity, checkpoint_root=checkpoint_root,
            checkpoint_path=checkpoint_path,
            trusted_receipt_sha256=trusted_receipt_sha256,
        )
        self._d4e_verified_identity = dict(identity)
        self._d4e_verified_r0_record_digest = record['all_input_hashes_sha256']

    def d4e_recovery_start(self, nb: int) -> int:
        if not getattr(self, '_d4e_verified_identity', None):
            raise RuntimeError('R3_UNVERIFIED_RECOVERY_START_FORBIDDEN')
        if self._d4e_verified_identity != getattr(self, '_d4e_recovery_identity', None):
            raise RuntimeError('R3_VERIFIED_IDENTITY_CHANGED_AFTER_CONFIG')
        return super().d4e_recovery_start(nb)

    def d4e_recovery_epoch_end(self, epoch: int, nb: int, last_opt_step: int) -> None:
        if self._d4e_verified_identity != getattr(self, '_d4e_recovery_identity', None):
            raise RuntimeError('R3_UNVERIFIED_CAPSULE_WRITE_FORBIDDEN')
        return super().d4e_recovery_epoch_end(epoch, nb, last_opt_step)
