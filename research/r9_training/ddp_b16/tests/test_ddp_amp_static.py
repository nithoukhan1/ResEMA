"""Static CPU guardrail for DDP-B16 AMP candidate. Not a CUDA/AMP acceptance."""
from __future__ import annotations

import unittest
from pathlib import Path

from research.r9_training.ddp_b16.native_source_adapter import (
    BASE_BLOBS, SRC_ROOT, create_candidate_sources,
)

REPO = Path(__file__).resolve().parents[4]


class DDPB16AMPStaticTests(unittest.TestCase):
    def test_two_rank_gate_and_guarded_scaler_retained(self):
        frozen = REPO / SRC_ROOT
        originals = {p: (frozen / p).read_bytes() for p in BASE_BLOBS}
        candidate = create_candidate_sources(originals)
        native = candidate["ultralytics/engine/trainer.py"].decode()

        self.assertIn('self.world_size != 2', native)
        self.assertNotIn('self.world_size != 1', native)
        self.assertIn('if not getattr(self, "_r7c_amp_gate_ready", False)', native)
        self.assertIn('if not self.amp or self.world_size != 2:', native)
        self.assertIn('init_scale=1.0, growth_interval=2000', native)
        self.assertIn('dist.broadcast(self.amp.int(), src=0)', native)
        self.assertIn('find_unused_parameters=True', native)

        # The reviewed existing guarded optimizer must continue to refuse
        # nonfinite gradients, skipped optimizer updates and scaler drops.
        guard = (frozen / "ultralytics/models/yolo/detect/vgra_fail_closed.py").read_text()
        for expected in (
            'self.scaler.unscale_(self.optimizer)',
            'require_finite_gradients(model, "unscaled_before_clip")',
            'self.scaler.step(self.optimizer)',
            'self._d4_optimizer_steps_observed != prior_steps + 1',
            'scale_after < scale_before',
            'require_finite_model(model, "post_update_model")',
        ):
            self.assertIn(expected, guard)

    def test_candidate_does_not_change_scientific_vgra_math(self):
        frozen = REPO / SRC_ROOT
        untouched = [
            "ultralytics/nn/modules/vgra.py",
            "ultralytics/nn/modules/vgra_head.py",
            "ultralytics/models/yolo/detect/vgra_runtime.py",
            "ultralytics/models/yolo/detect/vgra_targets.py",
        ]
        self.assertTrue(all((frozen / p).is_file() for p in untouched))
        self.assertTrue(set(BASE_BLOBS).isdisjoint(untouched))


if __name__ == "__main__":
    unittest.main()
