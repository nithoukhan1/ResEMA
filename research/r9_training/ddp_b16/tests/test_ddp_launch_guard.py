"""CPU-only refusal contracts for the DDP-specific native startup boundary."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.r9_training.ddp_b16.ddp_launch_guard import (
    DDPB16LaunchError, require_authenticated_ddp_child,
)
from research.r9_training.ddp_b16.ddp_startup_boundary import (
    DDPB16StartupError, require_ddp_b16_native_preloop,
)
from research.r9_training.ddp_b16.native_source_adapter import (
    BASE_BLOBS, SRC_ROOT, create_candidate_sources,
)

IDENTITY = {
    "arm": "MV00", "run_id": "DDP_B16_TEST",
    "ddp_release_sha": "a" * 40,
    "ddp_recipe_sha256": "b" * 64,
    "assignment_sha256": "c" * 64,
    "pretrained_sha256": "d" * 64,
    "stage_source_manifest_sha256": "e" * 64,
}


class EarlyPairedTrainer:
    def __init__(self, *, ready=False, g4=False, legacy=False, ddp=False,
                 batch=16, device="0,1", native_ready=False, root=None):
        self.args = SimpleNamespace(
            batch=batch, device=device, imgsz=1024, epochs=100, seed=42,
            workers=0, nbs=64, optimizer="SGD", amp=True, resume=False,
            val=True, save=True,
        )
        self.world_size = 2
        self.ddp = ddp
        self._ddp_b16_verified_worker_ready = ready
        self._ddp_b16_g4_native_bound = g4
        self._ddp_b16_g4_identity = dict(IDENTITY)
        self._d4e_recovery_configured = legacy
        self._ddp_b16_native_state_ready = native_ready
        self._ddp_b16_native_mode = "FRESH_FROM_EARLY"
        self._ddp_b16_g4_root = root
        self.train_loader = SimpleNamespace(
            vgra_pair_batch_sampler=SimpleNamespace(
                rank=0, num_images=14227, num_pairs=6496, num_singles=1235,
            )
        )


class DDPB16LaunchGuardTests(unittest.TestCase):
    def test_refuses_stock_parent_and_wrong_torchrun_mapping(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, clear=True):
            with self.assertRaisesRegex(DDPB16LaunchError, "REFUSE_ULTRALYTICS_AUTO_LAUNCH"):
                require_authenticated_ddp_child(EarlyPairedTrainer(ddp=True))
        for env in (
            {"WORLD_SIZE": "2", "RANK": "0"},
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"},
            {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"},
        ):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(DDPB16LaunchError):
                    require_authenticated_ddp_child(EarlyPairedTrainer())

    def test_refuses_missing_worker_legacy_and_missing_new_g4(self):
        for rank in (0, 1):
            env = {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(DDPB16LaunchError, "VERIFIED_WORKER_NOT_IMPLEMENTED"):
                    require_authenticated_ddp_child(EarlyPairedTrainer())
                with self.assertRaisesRegex(DDPB16LaunchError, "LEGACY_BATCH8_RECOVERY_FORBIDDEN"):
                    require_authenticated_ddp_child(EarlyPairedTrainer(ready=True, g4=True, legacy=True))
                with self.assertRaisesRegex(DDPB16LaunchError, "TWO_RANK_G4_NOT_BOUND"):
                    require_authenticated_ddp_child(EarlyPairedTrainer(ready=True))
                # Structural guard success is NOT GPU authority: --execute stays locked.
                require_authenticated_ddp_child(EarlyPairedTrainer(ready=True, g4=True))

    def test_rejects_invalid_identity_arm_and_recipe(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, clear=True):
            t = EarlyPairedTrainer(ready=True, g4=True)
            t._ddp_b16_g4_identity["assignment_sha256"] = "bad"
            with self.assertRaisesRegex(DDPB16LaunchError, "IDENTITY_INVALID"):
                require_authenticated_ddp_child(t)
            t = EarlyPairedTrainer(ready=True, g4=True)
            t._ddp_b16_g4_identity["arm"] = "MV01"
            with self.assertRaisesRegex(DDPB16LaunchError, "ARM_MISMATCH"):
                require_authenticated_ddp_child(t)
            for change in ("batch", "device", "world_size"):
                t = EarlyPairedTrainer(ready=True, g4=True)
                if change == "world_size":
                    t.world_size = 1
                elif change == "batch":
                    t.args.batch = 8
                else:
                    t.args.device = "0"
                with self.subTest(change=change), self.assertRaises(DDPB16LaunchError):
                    require_authenticated_ddp_child(t)

    def test_private_engine_calls_preloop_not_legacy_ddp_configuration(self):
        root = Path(__file__).resolve().parents[4] / SRC_ROOT
        originals = {rel: (root / rel).read_bytes() for rel in BASE_BLOBS}
        patched = create_candidate_sources(originals)
        engine = patched["ultralytics/engine/trainer.py"].decode()
        self.assertIn("require_ddp_b16_native_preloop(self)", engine)
        self.assertIn('if self.world_size == 2:', engine)
        self.assertIn("D4E_R2_GOVERNED_RECOVERY_CONFIGURATION_REQUIRED", engine)
        self.assertIn("G3B4_MANDATORY_GUARD_MIXIN_MISSING", engine)
        vgra = patched["ultralytics/models/yolo/detect/vgra_trainer.py"].decode()
        early = patched["ultralytics/models/yolo/detect/early_paired_trainer.py"].decode()
        self.assertIn("require_authenticated_ddp_child(self)", vgra)
        self.assertIn("class EarlyPairedTrainer(VGRADetectionTrainer)", early)


class DDPB16NativePreloopTests(unittest.TestCase):
    def test_fresh_resume_unimplemented_and_legacy_forbidden(self):
        with tempfile.TemporaryDirectory() as t:
            candidate = EarlyPairedTrainer(ready=True, g4=True, root=t)
            candidate.train_loader.__len__ = None
            class Loader:
                def __init__(self, sampler):
                    self.vgra_pair_batch_sampler = sampler
                def __len__(self):
                    return 890
            candidate.train_loader = Loader(candidate.train_loader.vgra_pair_batch_sampler)
            with self.assertRaisesRegex(DDPB16StartupError, "FRESH_OR_RESUME_STATE_NOT_IMPLEMENTED"):
                require_ddp_b16_native_preloop(candidate)
            candidate._d4e_recovery_configured = True
            with self.assertRaisesRegex(DDPB16StartupError, "LEGACY_D4E_FORBIDDEN"):
                require_ddp_b16_native_preloop(candidate)
            candidate._d4e_recovery_configured = False
            candidate._ddp_b16_native_state_ready = True
            require_ddp_b16_native_preloop(candidate)  # synthetic invariant-only check
            candidate._ddp_b16_native_mode = "RESUME_FROM_DDP_B16"
            with self.assertRaisesRegex(DDPB16StartupError, "OFFSESSION_RESUME_NOT_VERIFIED"):
                require_ddp_b16_native_preloop(candidate)


if __name__ == "__main__":
    unittest.main()
