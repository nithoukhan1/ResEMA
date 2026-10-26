"""CPU-only refusal tests for the verified two-rank native DDP launch boundary."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research.r9_training.ddp_b16.ddp_launch_guard import (
    DDPB16LaunchError, require_authenticated_ddp_child,
)
from research.r9_training.ddp_b16.native_source_adapter import (
    BASE_BLOBS, SRC_ROOT, create_candidate_sources,
)
from pathlib import Path


def candidate(*, ready=False, recovery=False, ddp=False, batch=16, device="0,1"):
    return SimpleNamespace(
        args=SimpleNamespace(batch=batch, device=device),
        world_size=2,
        ddp=ddp,
        _ddp_b16_verified_worker_ready=ready,
        _d4e_recovery_configured=recovery,
    )


class DDPB16LaunchGuardTests(unittest.TestCase):
    def test_never_accept_stock_ddp_parent(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}):
            with self.assertRaisesRegex(DDPB16LaunchError, "REFUSE_ULTRALYTICS_AUTO_LAUNCH"):
                require_authenticated_ddp_child(candidate(ddp=True, ready=True, recovery=True))

    def test_refuses_absent_or_corrupt_torchrun_env(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}, clear=True):
            with self.assertRaisesRegex(DDPB16LaunchError, "REQUIRES_EXPLICIT_TORCHRUN"):
                require_authenticated_ddp_child(candidate(ready=True, recovery=True))
        for env in (
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"},
            {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"},
            {"WORLD_SIZE": "2", "RANK": "9", "LOCAL_RANK": "9"},
        ):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(DDPB16LaunchError):
                    require_authenticated_ddp_child(candidate(ready=True, recovery=True))

    def test_requires_verified_worker_and_recovery_even_under_torchrun(self):
        for rank in (0, 1):
            env = {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(DDPB16LaunchError, "VERIFIED_WORKER_NOT_IMPLEMENTED"):
                    require_authenticated_ddp_child(candidate(ready=False, recovery=False))
                with self.assertRaisesRegex(DDPB16LaunchError, "GOVERNED_RECOVERY_NOT_BOUND"):
                    require_authenticated_ddp_child(candidate(ready=True, recovery=False))
                require_authenticated_ddp_child(candidate(ready=True, recovery=True))

    def test_rejects_batch_device_or_world_drift(self):
        env = {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            for case in (
                candidate(ready=True, recovery=True, batch=8),
                candidate(ready=True, recovery=True, device="0"),
                candidate(ready=True, recovery=True),
            ):
                if case.args.batch == 16 and case.args.device == "0,1":
                    case.world_size = 1
                with self.assertRaises(DDPB16LaunchError):
                    require_authenticated_ddp_child(case)

    def test_patched_vgra_base_train_guards_both_model_arms(self):
        root = Path(__file__).resolve().parents[4] / SRC_ROOT
        original = {rel: (root / rel).read_bytes() for rel in BASE_BLOBS}
        patched = create_candidate_sources(original)
        vgra = patched["ultralytics/models/yolo/detect/vgra_trainer.py"].decode()
        early = patched["ultralytics/models/yolo/detect/early_paired_trainer.py"].decode()
        self.assertIn("require_authenticated_ddp_child(self)", vgra)
        self.assertIn("return super().train()", vgra)
        self.assertIn("class EarlyPairedTrainer(VGRADetectionTrainer)", early)


if __name__ == "__main__":
    unittest.main()
