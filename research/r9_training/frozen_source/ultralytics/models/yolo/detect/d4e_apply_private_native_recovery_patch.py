"""Offline candidate patcher for *PRIVATE* pinned checkout. NOT official train authority.

The pinned Git file must match the exact byte-level LF Git blob. This script
changes only private source staging and verifies each unique _do_train anchor.
Never apply to an in-place user checkout or previously modified production tree.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PINNED_BASE_BLOB = 'a3ad643da2141a425b92cee6c51bcd3417ec3506'
PINNED_HEAD = '48cc059cfa133ae5fbfcec607f15daeda843d3ba'
BRANCH_CLASS = 'class VGRADetectionTrainer(GuardedTrainerMixin, DetectionTrainer):'
RECOVERY_CLASS = 'class VGRADetectionTrainer(D4ERecoveryTrainerMixin, GuardedTrainerMixin, DetectionTrainer):'
RECOVERY_IMPORT = 'from .d4e_recovery_integration import D4ERecoveryTrainerMixin\n'


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def patch_native(source: bytes) -> bytes:
    if blob_sha(source) != PINNED_BASE_BLOB:
        raise ValueError('R2_NATIVE_GIT_BLOB_NOT_PINNED_REFUSE_PATCH')
    text = source.decode('utf-8')
    start = '    def _do_train(self):\n'
    stop = '    def auto_batch(self, max_num_obj=0):\n'
    if text.count(start) != 1 or text.count(stop) != 1:
        raise ValueError('R2_NATIVE_TRAIN_METHOD_UNEXPECTED')
    prefix, tail = text.split(start)
    body, suffix = tail.split(stop, 1)
    transforms = [
        (
            '        self._setup_train()\n',
            '        self._setup_train()\n'
            '        if hasattr(self, "d4e_recovery_start") and not getattr(self, "_d4e_recovery_configured", False):\n'
            '            raise RuntimeError("D4E_R2_GOVERNED_RECOVERY_CONFIGURATION_REQUIRED")\n',
        ),
        (
            '        self.run_callbacks("on_train_start")\n',
            '        self.run_callbacks("on_train_start")\n'
            '        if getattr(self, "_d4e_recovery_configured", False):\n'
            '            last_opt_step = self.d4e_recovery_start(nb)\n',
        ),
        (
            '        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start\n',
            '        if not getattr(self, "_d4e_recovery_restored", False):\n'
            '            self.optimizer.zero_grad()  # preserve authenticated pending grads only on governed restore\n',
        ),
        (
            '            self.run_callbacks("on_fit_epoch_end")\n',
            '            self.run_callbacks("on_fit_epoch_end")\n'
            '            if getattr(self, "_d4e_recovery_configured", False):\n'
            '                self.d4e_recovery_epoch_end(epoch, nb, last_opt_step)\n',
        ),
    ]
    for src, dst in transforms:
        if body.count(src) != 1:
            raise ValueError('R2_TRAIN_ANCHOR_MISSING_OR_AMBIGUOUS: ' + repr(src))
        body = body.replace(src, dst, 1)
    patched = prefix + start + body + stop + suffix
    ast.parse(patched)
    return patched.encode('utf-8')


def patch_specialized(source: bytes) -> bytes:
    text = source.decode('utf-8')
    if text.count(BRANCH_CLASS) != 1 or text.count('from ultralytics.models.yolo.detect.train import DetectionTrainer\n') != 1:
        raise ValueError('R2_VGRA_TRAINER_ANCHORS_DRIFT')
    if 'D4ERecoveryTrainerMixin' in text:
        raise ValueError('R2_DUPLICATE_RECOVERY_MIXIN_FORBIDDEN')
    text = text.replace('from ultralytics.models.yolo.detect.train import DetectionTrainer\n',
                        'from ultralytics.models.yolo.detect.train import DetectionTrainer\n' + RECOVERY_IMPORT, 1)
    text = text.replace(BRANCH_CLASS, RECOVERY_CLASS, 1)
    ast.parse(text)
    return text.encode('utf-8')


def apply_in_private_source(*, private_root: Path, expected_vgra_sha256: str, receipt_path: Path) -> dict:
    private_root = private_root.resolve(strict=True)
    # Caller supplies a private staged checkout, NOT the canonical checkout.
    if (private_root / '.git').exists():
        raise ValueError('R2_REFUSE_TO_PATCH_GIT_WORKTREE')
    marker = private_root / '.d4e_r2_disposable_private_stage'
    if not marker.is_file() or marker.is_symlink() or marker.read_text().strip() != PINNED_HEAD:
        raise ValueError('R2_EXPLICIT_DISPOSABLE_PRIVATE_STAGE_MARKER_REQUIRED')
    expected = {
        'ultralytics/engine/trainer.py': None,
        'ultralytics/models/yolo/detect/vgra_trainer.py': expected_vgra_sha256,
    }
    paths = {k: private_root / k for k in expected}
    before = {k: v.read_bytes() for k, v in paths.items()}
    if hashlib.sha256(before['ultralytics/models/yolo/detect/vgra_trainer.py']).hexdigest() != expected_vgra_sha256:
        raise ValueError('R2_VGRA_OVERLAY_SOURCE_SHA_DRIFT')
    after = {
        'ultralytics/engine/trainer.py': patch_native(before['ultralytics/engine/trainer.py']),
        'ultralytics/models/yolo/detect/vgra_trainer.py': patch_specialized(
            before['ultralytics/models/yolo/detect/vgra_trainer.py']),
    }
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError('R2_PATCH_RECEIPT_ALREADY_EXISTS')
    for name, b in after.items():
        dest = paths[name]
        if dest.is_symlink() or dest.stat().st_nlink != 1:
            raise ValueError('R2_NONPRIVATE_OR_LINKED_DESTINATION_FORBIDDEN')
        # File source is checked completely before the first mutation.
    record = {
        'schema': 'D4E_STEP14_4B_R2_PRIVATE_NATIVE_INTEGRATION_CANDIDATE',
        'source_commit': PINNED_HEAD,
        'official_training_authorized': False,
        'b_test_access': 'NONE',
        'files': {k: {
            'before_sha256': hashlib.sha256(before[k]).hexdigest(),
            'after_sha256': hashlib.sha256(after[k]).hexdigest(),
        } for k in after},
    }
    # Prevent partial-stage ambiguity: caller should stage a disposable checkout.
    for name, data in after.items():
        paths[name].write_bytes(data)
    with receipt_path.open('x', encoding='utf-8') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    return record
