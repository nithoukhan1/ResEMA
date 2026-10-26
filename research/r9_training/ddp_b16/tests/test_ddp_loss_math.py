"""DDP-B16 numerical loss contract tests (CPU-only, standard-library math).

These tests derive the existing C1 VGRA visibility loss and PyTorch DDP
mean-gradient semantics. They DO NOT replace real torch distributed backward
or FP16/AMP acceptance. No new scientific loss or hyperparameter is proposed.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
LAMBDA = 0.25
WORLD = 2


def ce_and_derivative(logits, target, weights, feature):
    """Weighted 4-class CE and exact derivative wrt one synthetic parameter.

    logits_j(theta) = logits_j(0) + feature_j * theta, theta=0.
    """
    shift = max(logits)
    shifted = [math.exp(x - shift) for x in logits]
    norm = sum(shifted)
    probs = [x / norm for x in shifted]
    ce = (shift + math.log(norm)) - logits[target]
    weight = weights[target]
    return weight * ce, weight * (sum(p * f for p, f in zip(probs, feature)) - feature[target])


def rank_visibility_loss_grad(pairs, classes):
    """Exact frozen V1: 0.25 * (2*Npairs) * mean_pair,class(weighted CE)."""
    if not pairs:
        return 0.0, 0.0
    weighted = [ce_and_derivative(logits, target, weights, feature)
                for pair in pairs for logits, target, weights, feature in pair]
    count = len(pairs)
    assert all(len(pair) == classes for pair in pairs)
    coeff = LAMBDA * 2 * count / (count * classes)
    return (coeff * sum(value for value, _ in weighted),
            coeff * sum(derivative for _, derivative in weighted))


def pair_case(seed):
    result = []
    for c in range(3):
        logits = [0.7 + 0.01 * seed, -0.4 + 0.05 * c, 0.2 - 0.02 * seed, 1.1]
        feature = [0.1 + c * 0.3, -0.5, 0.25 + seed * 0.01, 0.6]
        weights = [1.3, 0.7, 1.1, 0.9]
        result.append((logits, (seed + c) % 4, weights, feature))
    return result


def simulated_ddp_average_after_native_world_factor(grad0, grad1):
    """Frozen native: self.loss *= world_size; DDP averages rank gradients."""
    return (WORLD * grad0 + WORLD * grad1) / WORLD


class DDPVisibilityLossMathTests(unittest.TestCase):
    def test_pair_loss_and_gradient_match_combined_pair_population(self):
        rank0 = [pair_case(i) for i in range(4)]
        rank1 = [pair_case(i) for i in range(4, 7)]
        total = rank_visibility_loss_grad(rank0 + rank1, 3)
        left = rank_visibility_loss_grad(rank0, 3)
        right = rank_visibility_loss_grad(rank1, 3)
        self.assertAlmostEqual(left[0] + right[0], total[0], places=12)
        self.assertAlmostEqual(
            simulated_ddp_average_after_native_world_factor(left[1], right[1]),
            total[1], places=12,
        )
        self.assertNotAlmostEqual((left[1] + right[1]) / 2, total[1], places=9)

    def test_zero_pair_rank_has_zero_aux_gradient_not_fake_targets(self):
        rank0 = [pair_case(i) for i in range(4)]
        left = rank_visibility_loss_grad(rank0, 3)
        right = rank_visibility_loss_grad([], 3)
        self.assertEqual(right, (0.0, 0.0))
        self.assertAlmostEqual(
            simulated_ddp_average_after_native_world_factor(left[1], right[1]),
            left[1], places=12,
        )

    def test_single_view_count_does_not_inflate_visibility_term(self):
        pairs = [pair_case(i) for i in range(2)]  # 4 paired images
        ce_total = sum(ce_and_derivative(*entry)[0]
                       for pair in pairs for entry in pair)
        # The frozen formula multiplies mean(pair,class) by PAIR_IMAGES.
        frozen = LAMBDA * (2 * len(pairs)) * ce_total / (len(pairs) * 3)
        self.assertAlmostEqual(frozen, rank_visibility_loss_grad(pairs, 3)[0])
        for singles in (0, 1, 4):
            # Wrong alternative: multiplying by TOTAL LOCAL BATCH size.
            wrong = LAMBDA * (2 * len(pairs) + singles) * ce_total / (len(pairs) * 3)
            if singles:
                self.assertNotAlmostEqual(wrong, frozen)
            else:
                self.assertAlmostEqual(wrong, frozen)

    def test_frozen_source_keeps_four_component_and_ddp_world_scaling(self):
        base = REPO / "research/r9_training/frozen_source/ultralytics"
        native = (base / "engine/trainer.py").read_text()
        vgra = (base / "models/yolo/detect/vgra_runtime.py").read_text()
        self.assertIn("self.loss = loss.sum()", native)
        self.assertIn("self.loss *= self.world_size", native)
        self.assertIn("vis_term = VISIBILITY_LOSS_WEIGHT_V1 * pair_images * vis_mean", vgra)
        self.assertIn("loss_vector = torch.cat((native_total, vis_term.reshape(1)))", vgra)
        self.assertIn("native_total.new_zeros(())", vgra)

    def test_rank_local_yolo_target_denominators_are_not_full_global_loss(self):
        # Native YOLO divides class/box sums by per-rank target score sums
        # and THEN multiplies by local image count. World-size scaling makes
        # the DDP objective sum(rank-local native losses), NOT necessarily
        # the unsharded-global16 native loss. Do not claim bitwise equivalence.
        batch0, batch1 = 8, 7
        numerator0, numerator1 = 2.0, 3.0
        scores0, scores1 = 1.0, 4.0
        rank_sum = batch0 * numerator0 / scores0 + batch1 * numerator1 / scores1
        global_unsharded = (batch0 + batch1) * (numerator0 + numerator1) / (scores0 + scores1)
        self.assertNotAlmostEqual(rank_sum, global_unsharded)
        self.assertAlmostEqual(
            simulated_ddp_average_after_native_world_factor(
                batch0 * numerator0 / scores0, batch1 * numerator1 / scores1),
            rank_sum,
        )


if __name__ == "__main__":
    unittest.main()
