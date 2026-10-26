"""DDP-B16 native source adaptation, exact-blob gated and IN-MEMORY only.

This does not alter frozen_source, execute training, or grant GPU authorization.
The caller supplies raw bytes from the immutable C1 release and receives
candidate replacement bytes for an eventual isolated private source stage.

Changes are deliberately narrow:
- enforce two-rank/global batch 16 in preflight + constructor;
- use paired DDP TRAIN loader on both ranks (retain existing train-only weights);
- use complete non-distributed validation loader for rank-0 later;
- connect the actual sampler epoch reset to the frozen native training loop;
- allow only 2-rank AMP in a new candidate source identity.

Not complete until gradient/loss equivalence, validation barriers, DDP checkpoint
state and true cross-session GPU acceptance have been separately verified.
"""
from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping


SRC_ROOT = "research/r9_training/frozen_source/"
BASE_BLOBS = {
    "ultralytics/engine/trainer.py": "697cdc1388a3b02bc739fe4cca5f97f1d63f8170",
    "ultralytics/models/yolo/detect/vgra_trainer.py": "ef705bac5961b37e666e134ea425a5d0bfeff958",
    "ultralytics/models/yolo/detect/early_paired_trainer.py": "28f2de9baeeb6d7e48afcbc385b57d24fb350e75",
    "ultralytics/models/yolo/detect/vgra_validator.py": "012d56f9d449183e8fb176d5660700aed9fc10e6",
    "ultralytics/models/yolo/detect/early_paired_validator.py": "45a7cdf9ea80418a1f76e269d827fad08fd4c4af",
}


class DDPSourcePatchError(RuntimeError):
    """Unexpected C1 scientific source; do not patch or fall back."""


def git_blob_sha(content: bytes) -> str:
    """Compute the Git blob hash for exact C1 source-byte authenticity."""
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\x00" + content).hexdigest()


def _replace_once(source: str, old: str, new: str, *, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise DDPSourcePatchError(f"{name}: source anchor count={count}; expected 1")
    return source.replace(old, new, 1)


def _patch_vgra(source: str) -> str:
    source = _replace_once(
        source,
        'from ultralytics.data.vgra import build_vgra_dataloader, build_vgra_yolo_dataset\n',
        'from ultralytics.data.vgra import build_vgra_dataloader, build_vgra_yolo_dataset\n'
        'from research.r9_training.ddp_b16.ddp_batch_loader import build_distributed_pair_training_loader\n',
        name="VGRA_LOADER_IMPORT",
    )
    old = '''    if isinstance(device, (tuple, list)) and len(device) > 1:
        raise ValueError("D4 multi-GPU configuration forbidden before native construction")
    if isinstance(device, str) and "," in device:
        raise ValueError("D4 multi-GPU configuration forbidden before native construction")
'''
    new = '''    # DDP-B16 source candidate: only the reviewed global batch-16 / two-T4 case.
    if str(device) != "0,1" or int(getattr(effective, "batch", -1)) != 16:
        raise ValueError("DDP_B16 requires device='0,1' and global batch=16")
'''
    source = _replace_once(source, old, new, name="VGRA_NATIVE_PREFLIGHT")
    source = _replace_once(
        source,
        '    MODEL_YAML_PATH = VGRA_YAML_PATH\n',
        '    MODEL_YAML_PATH = VGRA_YAML_PATH\n\n'
        '    def train(self):\n'
        '        from research.r9_training.ddp_b16.ddp_launch_guard import require_authenticated_ddp_child\n'
        '        require_authenticated_ddp_child(self)\n'
        '        return super().train()\n',
        name="VGRA_FAIL_CLOSED_DDP_LAUNCH",
    )
    source = _replace_once(
        source,
        '        if self.world_size > 1:\n            raise NotImplementedError("VGRA V1 distributed training is not authorized")\n',
        '        if self.world_size != 2 or int(self.args.batch) != 16:\n'
        '            raise RuntimeError("DDP_B16 requires exactly two ranks and global batch 16")\n',
        name="VGRA_INIT_WORLD_SIZE",
    )
    source = _replace_once(
        source,
        '        if rank != -1 or self.world_size > 1:\n'
        '            raise NotImplementedError("VGRA V1 requires non-distributed training/validation")\n',
        '        if self.world_size != 2 or rank not in (0, 1):\n'
        '            raise RuntimeError("DDP_B16 requires rank 0 or 1")\n',
        name="VGRA_GET_DATALOADER_GUARD",
    )
    old = '''        return build_vgra_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=(mode == "train"),
            seed=self.args.seed,
            rank=rank,
            drop_last=False,
            pin_memory=True,
        )
'''
    new = '''        if mode == "train":
            if batch_size != 8:
                raise RuntimeError("DDP_B16 requires per-rank TRAIN batch 8")
            return build_distributed_pair_training_loader(
                dataset, rank=rank, global_batch=16,
                seed=self.args.seed, workers=self.args.workers,
            )
        # Whole B-VAL evaluation remains unsharded. Rank-0 use of this loader
        # and rank synchronization MUST be separately verified.
        return build_vgra_dataloader(
            dataset, batch=batch_size, workers=0, shuffle=False,
            seed=self.args.seed, rank=-1, drop_last=False, pin_memory=True,
        )
'''
    source = _replace_once(source, old, new, name="VGRA_DDP_TRAIN_RETURN")
    return source


def _patch_early(source: str) -> str:
    source = _replace_once(
        source,
        'from ultralytics.data.vgra import build_vgra_dataloader\n',
        'from ultralytics.data.vgra import build_vgra_dataloader\n'
        'from research.r9_training.ddp_b16.ddp_batch_loader import build_distributed_pair_training_loader\n',
        name="EARLY_LOADER_IMPORT",
    )
    source = _replace_once(
        source,
        '        if rank != -1 or self.world_size > 1 or mode not in {"train", "val"}:\n'
        '            raise NotImplementedError("D4 MV-00 requires non-distributed TRAIN/VAL only")\n',
        '        if self.world_size != 2 or rank not in (0, 1) or mode not in {"train", "val"}:\n'
        '            raise RuntimeError("DDP_B16 MV00 requires rank 0/1 and TRAIN/VAL only")\n',
        name="EARLY_GET_DATALOADER_GUARD",
    )
    old = '''        return build_vgra_dataloader(
            dataset, batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=mode == "train", seed=self.args.seed, rank=rank,
            drop_last=False, pin_memory=True,
        )
'''
    new = '''        if mode == "train":
            if batch_size != 8:
                raise RuntimeError("DDP_B16 requires per-rank TRAIN batch 8")
            return build_distributed_pair_training_loader(
                dataset, rank=rank, global_batch=16,
                seed=self.args.seed, workers=self.args.workers,
            )
        return build_vgra_dataloader(
            dataset, batch=batch_size, workers=0, shuffle=False,
            seed=self.args.seed, rank=-1, drop_last=False, pin_memory=True,
        )
'''
    source = _replace_once(source, old, new, name="EARLY_DDP_TRAIN_RETURN")
    return source


def _patch_vgra_validator(source: str) -> str:
    return _replace_once(
        source,
        '        if trainer.world_size not in {0, 1} or RANK not in {-1, 0}:\n'
        '            raise NotImplementedError("VGRA V1 DDP validation not supported")\n',
        '        if trainer.world_size == 2:\n'
        '            if RANK != 0 or getattr(trainer, "_ddp_b16_rank0_validation_ready", False) is not True:\n'
        '                raise RuntimeError("DDP_B16_VGRA_RANK0_GOVERNED_VALIDATION_ONLY")\n'
        '        elif trainer.world_size not in {0, 1} or RANK not in {-1, 0}:\n'
        '            raise NotImplementedError("VGRA V1 DDP validation not supported")\n',
        name="VGRA_VAL_RANK0_GUARD",
    )


def _patch_early_validator(source: str) -> str:
    return _replace_once(
        source,
        '        if trainer.world_size not in {0, 1} or RANK not in {-1, 0}:\n'
        '            raise NotImplementedError("D4 Early paired DDP validation unauthorized")\n',
        '        if trainer.world_size == 2:\n'
        '            if RANK != 0 or getattr(trainer, "_ddp_b16_rank0_validation_ready", False) is not True:\n'
        '                raise RuntimeError("DDP_B16_EARLY_RANK0_GOVERNED_VALIDATION_ONLY")\n'
        '        elif trainer.world_size not in {0, 1} or RANK not in {-1, 0}:\n'
        '            raise NotImplementedError("D4 Early paired DDP validation unauthorized")\n',
        name="EARLY_VAL_RANK0_GUARD",
    )


def _patch_engine(source: str) -> str:
    # Existing source emits rank-specific DDP commands. Keep its pre-existing
    # torch.distributed wrapping and official numerical guards, modifying
    # single-world-size AMP checks, sampler epoch and rank-0 validation.
    source = _replace_once(
        source,
        '        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP\n',
        '        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP\n',
        name="ENGINE_AMP_START_ANCHOR",
    )
    source = _replace_once(
        source,
        '            if not getattr(self, "_r7c_amp_gate_ready", False) or self.world_size != 1:\n',
        '            if not getattr(self, "_r7c_amp_gate_ready", False) or self.world_size != 2:\n',
        name="ENGINE_R7C_AMP_WORLD_SIZE",
    )
    source = _replace_once(
        source,
        '            if not self.amp or self.world_size != 1:\n',
        '            if not self.amp or self.world_size != 2:\n',
        name="ENGINE_R7C_SCALER_WORLD_SIZE",
    )
    source = _replace_once(
        source,
        '                self.train_loader.sampler.set_epoch(epoch)\n',
        '                from research.r9_training.ddp_b16.ddp_batch_loader import begin_distributed_pair_epoch\n'
        '                begin_distributed_pair_epoch(self.train_loader, epoch)\n',
        name="ENGINE_PAIR_SAMPLER_EPOCH",
    )
    source = _replace_once(
        source,
        '                self.metrics, self.fitness = self.validate()\n',
        '                from research.r9_training.ddp_b16.ddp_epoch_boundary import rank0_validate_and_sync\n'
        '                rank0_validate_and_sync(self, epoch=epoch)\n',
        name="ENGINE_RANK0_VAL_AND_SYNC",
    )
    source = _replace_once(
        source,
        '            if getattr(self, "_d4e_recovery_configured", False):\n'
        '                self.d4e_g3b_guard_epoch_boundary(epoch=epoch)\n'
        '                self.d4e_recovery_epoch_end(epoch, nb, last_opt_step)\n',
        '            if self.world_size == 2:\n'
        '                from research.r9_training.ddp_b16.ddp_native_g4_hook import ddp_b16_g4_epoch_end\n'
        '                ddp_b16_g4_epoch_end(self, epoch=epoch, nb=nb, last_opt_step=last_opt_step)\n'
        '            elif getattr(self, "_d4e_recovery_configured", False):\n'
        '                self.d4e_g3b_guard_epoch_boundary(epoch=epoch)\n'
        '                self.d4e_recovery_epoch_end(epoch, nb, last_opt_step)\n',
        name="ENGINE_DDP_NATIVE_G4_CALLBACK",
    )
    return source


PATCHES = {
    "ultralytics/engine/trainer.py": _patch_engine,
    "ultralytics/models/yolo/detect/vgra_trainer.py": _patch_vgra,
    "ultralytics/models/yolo/detect/early_paired_trainer.py": _patch_early,
    "ultralytics/models/yolo/detect/vgra_validator.py": _patch_vgra_validator,
    "ultralytics/models/yolo/detect/early_paired_validator.py": _patch_early_validator,
}


def create_candidate_sources(originals: Mapping[str, bytes]) -> dict[str, bytes]:
    """Transform verified C1 bytes in memory, atomically (no filesystem IO).

    Callers MUST stage a private copy with an independent source manifest.
    Rejects unexpected code bytes, missing or extra file paths, non-UTF8,
    ambiguous anchors and syntactically invalid candidate source.
    """
    if set(originals) != set(BASE_BLOBS):
        raise DDPSourcePatchError("DDP_B16 source file set differs from C1")
    result: dict[str, bytes] = {}
    for name, expected in BASE_BLOBS.items():
        content = originals[name]
        if not isinstance(content, bytes) or git_blob_sha(content) != expected:
            raise DDPSourcePatchError(f"DDP_B16 original Git blob mismatch: {name}")
        try:
            text = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise DDPSourcePatchError(f"DDP_B16 non-UTF8 source: {name}") from e
        patched = PATCHES[name](text)
        if patched == text:
            raise DDPSourcePatchError(f"DDP_B16 no-op patch: {name}")
        try:
            ast.parse(patched, filename=name)
        except SyntaxError as e:
            raise DDPSourcePatchError(f"DDP_B16 invalid adapted Python: {name}") from e
        result[name] = patched.encode("utf-8")
    return result
