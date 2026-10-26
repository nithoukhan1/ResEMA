"""Review frozen C1 -> DDP-B16 native trainer candidate patches, CPU only."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from research.r9_training.ddp_b16.native_source_adapter import (
    BASE_BLOBS, SRC_ROOT, DDPSourcePatchError, create_candidate_sources,
    git_blob_sha,
)


ROOT = Path(__file__).resolve().parents[4]


def original_bytes() -> dict[str, bytes]:
    return {rel: (ROOT / SRC_ROOT / rel).read_bytes() for rel in BASE_BLOBS}


class DDPB16NativeSourceAdapterTests(unittest.TestCase):
    def test_pinned_source_is_exact_and_candidate_is_syntactically_valid(self):
        original = original_bytes()
        for path, digest in BASE_BLOBS.items():
            self.assertEqual(git_blob_sha(original[path]), digest, path)

        result = create_candidate_sources(original)
        self.assertEqual(set(result), set(BASE_BLOBS))
        self.assertTrue(all(result[p] != original[p] for p in BASE_BLOBS))
        for path, data in result.items():
            ast.parse(data.decode("utf8"), filename=path)
            self.assertEqual((ROOT / SRC_ROOT / path).read_bytes(), original[path])

        engine = result["ultralytics/engine/trainer.py"].decode()
        self.assertIn("begin_distributed_pair_epoch(self.train_loader, epoch)", engine)
        self.assertNotIn("self.train_loader.sampler.set_epoch(epoch)", engine)
        self.assertIn('self.world_size != 2', engine)
        self.assertIn("nn.parallel.DistributedDataParallel(", engine)
        self.assertIn("rank0_validate_and_sync(self, epoch=epoch)", engine)
        self.assertNotIn("self.metrics, self.fitness = self.validate()", engine)
        self.assertIn("if getattr(self, \"_d4e_recovery_configured\"", engine)

        vgra = result["ultralytics/models/yolo/detect/vgra_trainer.py"].decode()
        self.assertIn("build_distributed_pair_training_loader(", vgra)
        self.assertIn("train_state_weight_binding(dataset, nc=9, split=\"train\")", vgra)
        self.assertIn('self.world_size != 2', vgra)
        self.assertIn('rank=-1, drop_last=False', vgra)
        self.assertIn("DDP_B16 requires per-rank TRAIN batch 8", vgra)

        for name in ("vgra_validator.py", "early_paired_validator.py"):
            adapted = result["ultralytics/models/yolo/detect/" + name].decode()
            self.assertIn("_ddp_b16_rank0_validation_ready", adapted)
            self.assertIn("RANK != 0", adapted)
            self.assertIn("3049", adapted)

        early = result["ultralytics/models/yolo/detect/early_paired_trainer.py"].decode()
        self.assertIn("build_distributed_pair_training_loader(", early)
        self.assertIn("DDP_B16 requires per-rank TRAIN batch 8", early)
        self.assertIn('rank=-1, drop_last=False', early)

    def test_refuse_modified_original_bytes(self):
        originals = original_bytes()
        for name in originals:
            tampered = dict(originals)
            tampered[name] += b"# tampered"
            with self.subTest(file=name), self.assertRaises(DDPSourcePatchError):
                create_candidate_sources(tampered)

    def test_refuse_missing_extra_or_double_patched_file(self):
        originals = original_bytes()
        missing = dict(originals)
        missing.pop(next(iter(missing)))
        with self.assertRaises(DDPSourcePatchError):
            create_candidate_sources(missing)
        extra = dict(originals)
        extra["arbitrary.py"] = b"pass\n"
        with self.assertRaises(DDPSourcePatchError):
            create_candidate_sources(extra)
        once = create_candidate_sources(originals)
        with self.assertRaises(DDPSourcePatchError):
            create_candidate_sources(once)

    def test_repeat_candidate_output_is_byte_deterministic(self):
        originals = original_bytes()
        self.assertEqual(create_candidate_sources(originals), create_candidate_sources(originals))


if __name__ == "__main__":
    unittest.main()
