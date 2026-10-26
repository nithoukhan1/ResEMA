"""CPU-only DDP-B16 sampler contract tests. No Kaggle, models or GPUs."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ddp_b16_pair_sampler", HERE / "distributed_pair_sampler.py"
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
Sampler = mod.DistributedPairBatchSampler


def make_units(pairs: int, singles: int):
    out = [(2 * i, 2 * i + 1) for i in range(pairs)]
    out.extend((2 * pairs + j,) for j in range(singles))
    return out


class TwoRankPairSamplerTests(unittest.TestCase):
    def test_frozen_training_population_exact_coverage(self):
        units = make_units(6496, 1235)
        a = Sampler(units, rank=0, seed=42)
        b = Sampler(units, rank=1, seed=42)
        audit = a.audit(0)
        self.assertEqual(audit["images_total"], 14227)
        self.assertEqual(audit["complete_pairs_total"], 6496)
        self.assertEqual(audit["singles_total"], 1235)
        self.assertEqual(audit["steps_per_rank"], 890)
        self.assertEqual(audit["min_local_batch"], 7)
        self.assertEqual(audit["max_local_batch"], 8)
        self.assertEqual(audit["rank0_images"] + audit["rank1_images"], 14227)
        self.assertLessEqual(abs(audit["rank0_images"] - audit["rank1_images"]), 1)
        self.assertEqual(audit, b.audit(0))

        ba, bb = list(a), list(b)
        self.assertEqual(len(ba), len(bb))
        self.assertEqual(len(ba), 890)
        positions = {}
        for step, two in enumerate(zip(ba, bb)):
            for rank, batch in enumerate(two):
                self.assertTrue(1 <= len(batch) <= 8)
                for idx in batch:
                    self.assertNotIn(idx, positions)
                    positions[idx] = (step, rank)
        self.assertEqual(set(positions), set(range(14227)))
        for ap, lat in units[:6496]:
            self.assertEqual(positions[ap], positions[lat])
        self.assertEqual((a.epoch, b.epoch), (1, 1))

    def test_reproducible_epoch_shuffle_across_ranks_and_instances(self):
        units = make_units(24, 47)
        ranks = [Sampler(units, rank=r, seed=42) for r in (0, 1)]
        for sam in ranks:
            sam.set_epoch(7)
        first = [list(sam) for sam in ranks]
        replicas = [Sampler(units, rank=r, seed=42) for r in (0, 1)]
        for sam in replicas:
            sam.set_epoch(7)
        self.assertEqual(first, [list(sam) for sam in replicas])
        for sam in ranks:
            sam.set_epoch(8)
        later = [list(sam) for sam in ranks]
        self.assertNotEqual(first, later)

    def test_validation_does_not_advance_epoch(self):
        units = make_units(16, 31)
        ranks = [Sampler(units, rank=r, shuffle=False) for r in (0, 1)]
        initial = [list(sam) for sam in ranks]
        self.assertEqual(initial, [list(sam) for sam in ranks])
        self.assertEqual([sam.epoch for sam in ranks], [0, 0])
        self.assertEqual(len(initial[0]), 4)

    def test_no_very_small_tail(self):
        units = make_units(16, 31)
        a = Sampler(units, rank=0)
        b = Sampler(units, rank=1)
        self.assertEqual(a.audit(0)["min_local_batch"], 7)
        self.assertEqual(sorted(len(x) for x in a), [8, 8, 8, 8])
        self.assertEqual(sorted(len(x) for x in b), [7, 8, 8, 8])

    def test_rejects_invalid_parameters(self):
        units = make_units(16, 31)
        for kw in (
            {"world_size": 1},
            {"rank": 2},
            {"per_rank_batch": 16},
            {"seed": -1},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Sampler(units, rank=0, **kw)

    def test_rejects_bad_units(self):
        for units in (
            [],
            [(0, 1, 2)],
            [(0, 1), (1,)],
            [(0, 2), (3,)],
            [(-1, 0), (1,)],
            [(0, 1), (True,)],
            [(0, 1)],  # cannot place both ranks on one step
        ):
            with self.subTest(units=units), self.assertRaises(ValueError):
                Sampler(units, rank=0)

    def test_set_epoch_enforces_valid_cursor(self):
        sample = Sampler(make_units(16, 31), rank=0)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                sample.set_epoch(invalid)


if __name__ == "__main__":
    unittest.main()
