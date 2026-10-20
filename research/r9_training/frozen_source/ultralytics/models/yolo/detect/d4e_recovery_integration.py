"""Step14.4B R2 private candidate integration for D4-E native trainers.

NOT authorized for full 100-epoch training. This mixin becomes active ONLY when
explicitly configured after the existing guarded trainer constructor. The native
trainer's resume mechanism stays disabled, and fresh Early authentication stays mandatory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .d4e_transform_rng_state import capture_augmentation_rng
from .d4e_trusted_recovery_capsule import (
    _identity_check, read_authenticated_capsule, restore_capsule,
    save_capsule, snapshot_epoch_boundary,
)

SOURCE_HEAD = "48cc059cfa133ae5fbfcec607f15daeda843d3ba"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _sampler(trainer: Any):
    loader = trainer.train_loader
    if int(loader.num_workers) != 0:
        raise RuntimeError('D4E_RECOVERY_REQUIRES_WORKERS_ZERO')
    inner = getattr(loader, 'vgra_pair_batch_sampler', None)
    if inner is None or not callable(getattr(inner, 'set_epoch', None)):
        raise RuntimeError('D4E_RECOVERY_REQUIRES_PAIRED_SAMPLER')
    if not callable(getattr(loader, 'reset', None)):
        raise RuntimeError('D4E_RECOVERY_REQUIRES_RESETTABLE_LOADER')
    return loader, inner


class D4ERecoveryTrainerMixin:
    """Governed recovery hooks inserted into the *private patched* native training loop."""

    def d4e_configure_recovery(
        self, *, mode: str, identity: dict[str, str], checkpoint_root: str | Path,
        checkpoint_path: str | Path | None = None, trusted_receipt_sha256: str | None = None,
    ) -> None:
        if getattr(self, '_d4e_recovery_configured', False):
            raise RuntimeError('D4E_RECOVERY_ALREADY_CONFIGURED')
        if mode not in {'FRESH_FROM_EARLY', 'RESUME_FROM_OUR_OWN_RUN'}:
            raise ValueError('D4E_RECOVERY_MODE_UNRECOGNIZED')
        _identity_check(identity)
        if identity['source_commit'] != SOURCE_HEAD:
            raise ValueError('D4E_SOURCE_HEAD_MISMATCH')
        # Guarded __init__ already authenticated the pinned Early pretrained state.
        # Never delegate to native resume or an arbitrary model injection.
        if bool(getattr(self, 'resume', False)) or bool(getattr(self.args, 'resume', False)):
            raise ValueError('D4E_NATIVE_RESUME_FORBIDDEN')
        if int(self.args.batch) != 8 or int(self.args.imgsz) != 1024 or not bool(self.args.amp):
            raise ValueError('R7C_ONLY_VERIFIED_BATCH8_AMP_1024_TEST_ONLY')
        if int(self.args.workers) != 0 or int(self.args.seed) != 42 or int(self.args.nbs) != 64:
            raise ValueError('D4E_RECOVERY_SAMPLER_OR_ACCUMULATION_RECIPE_DRIFT')
        if str(self.args.optimizer).upper() != 'SGD' or int(self.args.epochs) != 100:
            raise ValueError('D4E_RECOVERY_OPTIMIZER_OR_EPOCH_RECIPE_DRIFT')
        if not bool(getattr(self.args, 'save', False)) or not bool(getattr(self.args, 'val', False)):
            raise ValueError('D4E_RECOVERY_REQUIRES_SAVED_BEST_AND_EPOCH_VALIDATION')
        arm = 'MV00' if type(self).__name__ == 'EarlyPairedTrainer' else (
            'MV01' if type(self).__name__ == 'VGRADetectionTrainer' else None
        )
        if arm is None or identity['arm'] != arm:
            raise ValueError('D4E_RECOVERY_TRAINER_ARM_MISMATCH')
        root = Path(checkpoint_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('D4E_RECOVERY_ROOT_NOT_DIRECTORY')
        if mode == 'FRESH_FROM_EARLY':
            if checkpoint_path is not None or trusted_receipt_sha256 is not None:
                raise ValueError('D4E_FRESH_MODE_MUST_NOT_SUPPLY_RESUME_CHECKPOINT')
        else:
            if checkpoint_path is None or not trusted_receipt_sha256:
                raise ValueError('D4E_RESUME_REQUIRES_EXTERNAL_RECEIPT_PIN')
        self._d4e_recovery_configured = True
        self._d4e_recovery_mode = mode
        self._d4e_recovery_identity = dict(identity)
        self._d4e_checkpoint_root = root
        self._d4e_resume_checkpoint_path = checkpoint_path
        self._d4e_resume_receipt_pin = trusted_receipt_sha256
        self._d4e_recovery_restored = False
        self._d4e_optimizer_updates = 0
        self._d4e_epoch_receipts = []

    def optimizer_step(self):
        # Exactly one counter increment after every successful native optimizer update.
        result = super().optimizer_step()
        if getattr(self, '_d4e_recovery_configured', False):
            self._d4e_optimizer_updates += 1
        return result

    def d4e_recovery_start(self, nb: int) -> int:
        if not getattr(self, '_d4e_recovery_configured', False):
            raise RuntimeError('D4E_RECOVERY_NOT_CONFIGURED')
        loader, sampler = _sampler(self)
        if nb != len(loader) or getattr(self, 'world_size', 1) > 1:
            raise ValueError('D4E_RECOVERY_NATIVE_LOADER_LENGTH_OR_DDP_DRIFT')
        if self._d4e_recovery_mode == 'FRESH_FROM_EARLY':
            # The native loop performs its usual zero_grad() on fresh startup.
            self._d4e_recovery_restored = False
            self._d4e_optimizer_updates = 0
            return -1

        payload = read_authenticated_capsule(
            checkpoint_path=self._d4e_resume_checkpoint_path,
            root=self._d4e_checkpoint_root,
            trusted_receipt_sha256=self._d4e_resume_receipt_pin,
            identity=self._d4e_recovery_identity,
        )
        cursor = payload['cursor']
        if cursor['nb'] != nb or cursor['next_epoch'] >= int(self.epochs):
            raise ValueError('D4E_RECOVERY_BATCH_COUNT_OR_EPOCH_BOUNDARY_MISMATCH')
        if cursor['sampler_next_epoch'] != cursor['next_epoch']:
            raise ValueError('D4E_RECOVERY_SAMPLER_EPOCH_DRIFT')
        state = payload['early_stopper_state']
        required = {'best_fitness', 'stopper_best_fitness', 'stopper_best_epoch',
                    'stopper_possible_stop', 'selected_best_sha256'}
        if set(state) != required:
            raise ValueError('D4E_RECOVERY_STOPPER_SCHEMA_MISMATCH')
        # For mid-run interruption the selected best.pt and its identity must be
        # independently transferred. Merely restoring model tensors is insufficient.
        best = Path(self.best)
        if not best.is_file() or _file_sha256(best) != state['selected_best_sha256']:
            raise ValueError('D4E_RECOVERY_SELECTED_BEST_NOT_AUTHENTICATED')
        if self.args.close_mosaic and cursor['next_epoch'] >= int(self.epochs) - int(self.args.close_mosaic):
            self._close_dataloader_mosaic()
        # Native InfiniteDataLoader starts an iterator during construction.
        # Reset it to the exact next pair-epoch *before* R1 restores the RNG last.
        sampler.set_epoch(cursor['next_epoch'])
        loader.reset()
        # InfiniteDataLoader.reset() constructs a fresh iterator but workers=0
        # does NOT call VGRAPairBatchSampler.__iter__ yet. Its epoch advances
        # only when the first recovered batch is actually requested. Never
        # advance here: doing so would skip the intended next pair epoch.
        if sampler.epoch != cursor['next_epoch']:
            raise RuntimeError('D4E_RECOVERY_RESET_PRECONSUMED_NEXT_SAMPLER_EPOCH')
        restored = restore_capsule(
            payload=payload, model=self.model, optimizer=self.optimizer,
            ema_model=self.ema.ema, scheduler=self.scheduler, scaler=self.scaler,
            loader_generator=getattr(loader, 'generator', None),
            augmentation_transform_tree=loader.dataset.transforms,
        )
        if int(restored['ema_updates']) < 0:
            raise ValueError('D4E_RECOVERY_INVALID_EMA_UPDATES')
        self.ema.updates = restored['ema_updates']
        self.best_fitness = float(state['best_fitness'])
        self.stopper.best_fitness = float(state['stopper_best_fitness'])
        self.stopper.best_epoch = int(state['stopper_best_epoch'])
        self.stopper.possible_stop = bool(state['stopper_possible_stop'])
        self.start_epoch = cursor['next_epoch']
        self._d4e_optimizer_updates = cursor['optimizer_updates']
        if hasattr(self, '_d4_optimizer_steps_observed'):
            self._d4_optimizer_steps_observed = cursor['optimizer_updates']
        self.accumulate = cursor['accumulate']
        self._d4e_recovery_restored = True
        return cursor['last_opt_step']

    def d4e_recovery_epoch_end(self, epoch: int, nb: int, last_opt_step: int) -> None:
        """Capture after native save_model() and fit-epoch callbacks, before next epoch."""
        if not getattr(self, '_d4e_recovery_configured', False):
            raise RuntimeError('D4E_RECOVERY_NOT_CONFIGURED')
        loader, sampler = _sampler(self)
        if self.stop or (epoch + 1 >= int(self.epochs)):
            return  # Do not write resumable capsule for a terminal epoch.
        if nb != len(loader) or sampler.epoch != epoch + 1:
            raise RuntimeError('D4E_RECOVERY_EPOCH_SAMPLER_CURSOR_MISMATCH')
        best = Path(self.best)
        if not best.is_file():
            raise RuntimeError('D4E_RECOVERY_BEST_MUST_EXIST_FOR_RESUMABILITY')
        cursor = {
            'completed_epoch': epoch, 'next_epoch': epoch + 1, 'nb': nb,
            'last_opt_step': last_opt_step,
            'optimizer_updates': self._d4e_optimizer_updates,
            'accumulate': int(self.accumulate), 'sampler_next_epoch': epoch + 1,
        }
        stopper = {
            'best_fitness': float(self.best_fitness),
            'stopper_best_fitness': float(self.stopper.best_fitness),
            'stopper_best_epoch': int(self.stopper.best_epoch),
            'stopper_possible_stop': bool(self.stopper.possible_stop),
            'selected_best_sha256': _file_sha256(best),
        }
        capsule = snapshot_epoch_boundary(
            model=self.model, optimizer=self.optimizer, ema_model=self.ema.ema,
            ema_updates=self.ema.updates, scheduler=self.scheduler, scaler=self.scaler,
            identity=self._d4e_recovery_identity, cursor=cursor,
            loader_generator=getattr(loader, 'generator', None), early_stopper_state=stopper,
            augmentation_rng_state=capture_augmentation_rng(loader.dataset.transforms),
        )
        path = self._d4e_checkpoint_root / f"{self._d4e_recovery_identity['run_id']}_epoch{epoch+1:03d}.pt"
        receipt = save_capsule(payload=capsule, checkpoint_path=path, root=self._d4e_checkpoint_root)
        self._d4e_epoch_receipts.append(receipt)
        # Caller MUST move checkpoint, receipt, selected best and external pin off Kaggle ephemeral disk.
        print(f"D4E_RECOVERY_EPOCH_RECEIPT={receipt['receipt']} SHA256={receipt['trusted_receipt_sha256']}", flush=True)
