"""CPU-only child argument and rank environment refusal tests."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from research.r9_training.ddp_b16.ddp_child_review import (
    DDPChildReviewError, main, parse_rank_env,
)


class ChildReviewTests(unittest.TestCase):
    def test_execution_locked_even_without_a_plan(self):
        with self.assertRaisesRegex(DDPChildReviewError, "EXECUTION_NOT_IMPLEMENTED"):
            main(["--plan", "/does-not-exist", "--execute"])

    def test_requires_actual_two_rank_torchrun_mapping(self):
        for env in (
            {},
            {"WORLD_SIZE": "2", "RANK": "0"},
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"},
            {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"},
        ):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(DDPChildReviewError):
                    parse_rank_env()
        for rank in (0, 1):
            with patch.dict(os.environ, {
                "WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)
            }, clear=True):
                self.assertEqual(parse_rank_env(), (rank, 2))


if __name__ == "__main__":
    unittest.main()
