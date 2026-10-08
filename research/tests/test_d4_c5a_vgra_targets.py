from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.vgra_targets import (
    VISIBILITY_LOSS_WEIGHT_V1,
    canonical_pair_indices,
    class_presence_from_collated_labels,
    train_state_counts_from_dataset,
    train_state_weight_binding,
    visibility_targets_from_batch,
    visibility_weighted_ce,
)


def batch_fixture():
    return {
        "img": torch.zeros(5, 3, 16, 16),
        "cls": torch.tensor([[3.], [2.], [3.], [5.], [4.], [3.], [7.]]),
        "batch_idx": torch.tensor([0., 0., 1., 1., 2., 3., 4.]),
        "vgra_pair_index": torch.tensor([1, 0, 3, 2, -1]),
        "vgra_pair_valid": torch.tensor([True, True, True, True, False]),
        "vgra_view_code": torch.tensor([1, 2, 1, 2, 1]),
        "vgra_pair_id": ("P1", "P1", "P2", "P2", ""),
    }


def test_target_formula_and_unpaired_exclusion():
    b = batch_fixture()
    target, ap, lat = visibility_targets_from_batch(b, nc=9)
    assert ap.tolist() == [0, 2]
    assert lat.tolist() == [1, 3]
    assert target.shape == (2, 9)
    assert target[0, 2].item() == 1    # AP-only
    assert target[0, 3].item() == 3    # both AP/LAT
    assert target[0, 5].item() == 2    # LAT-only
    assert target[0, 0].item() == 0    # neither
    assert target[1, 4].item() == 1    # AP only in pair 2
    assert target[1, 3].item() == 2    # LAT only in pair 2
    assert target[1, 0].item() == 0
    assert target[:, 7].sum().item() == 0  # single image class must not appear in pair targets


def test_presence_is_detection_gt_derived():
    p = class_presence_from_collated_labels(batch_fixture(), nc=9)
    assert p.shape == (5, 9)
    assert p[0, 2] and p[0, 3]
    assert p[1, 3] and p[1, 5]
    assert p[4, 7]   # single view represented in image presence
    assert p[3, 3] and p[3].sum().item() == 1


def test_reject_nonreciprocal_companion():
    b = batch_fixture()
    b["vgra_pair_index"] = torch.tensor([1, 2, 3, 2, -1])
    with pytest.raises(ValueError, match="reciprocal"):
        canonical_pair_indices(b)


def test_reject_same_view_pair():
    b = batch_fixture()
    b["vgra_view_code"][1] = 1
    with pytest.raises(ValueError, match="AP/LAT"):
        canonical_pair_indices(b)


def test_reject_missing_pair_member():
    b = batch_fixture()
    b["vgra_pair_index"][1] = -1
    b["vgra_pair_valid"][1] = False
    with pytest.raises(ValueError, match="reciprocal"):
        canonical_pair_indices(b)


def test_reject_fractional_detection_label():
    b = batch_fixture()
    b["cls"][0, 0] = 3.25
    with pytest.raises(ValueError, match="non-integer"):
        class_presence_from_collated_labels(b, 9)


def test_reject_out_of_bounds_detection_label():
    b = batch_fixture()
    b["cls"][0, 0] = 9.
    with pytest.raises(ValueError, match="out-of-range"):
        class_presence_from_collated_labels(b, 9)


def test_weighted_ce_matches_reference_and_backward():
    torch.manual_seed(7)
    logits = torch.randn(2, 9, 4, requires_grad=True)
    target, _, _ = visibility_targets_from_batch(batch_fixture(), nc=9)
    weights = torch.rand(9, 4) + 0.1
    actual = visibility_weighted_ce(logits, target, weights)
    flat = F.cross_entropy(logits.reshape(-1, 4), target.flatten(), reduction="none")
    reference = (flat.view(2, 9) * weights[torch.arange(9)[None, :], target]).mean()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    (VISIBILITY_LOSS_WEIGHT_V1 * actual).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_empty_pairs_produce_zero_differentiable_auxiliary_loss():
    logits = torch.empty(0, 9, 4, requires_grad=True)
    target = torch.empty(0, 9, dtype=torch.long)
    weights = torch.ones(9, 4)
    loss = visibility_weighted_ce(logits, target, weights)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None


def test_train_state_counts_are_train_only():
    # Two AP/LAT pairs, one single. Four-state frequencies must each sum to two.
    labels = [
        {"cls": torch.tensor([[3], [5]])},
        {"cls": torch.tensor([[3]])},
        {"cls": torch.tensor([[2]])},
        {"cls": torch.tensor([], dtype=torch.long)},
        {"cls": torch.tensor([[4]])},
    ]
    meta = [
        {"view_code":"AP", "pair_valid":True, "pair_id":"P1"},
        {"view_code":"LAT", "pair_valid":True, "pair_id":"P1"},
        {"view_code":"AP", "pair_valid":True, "pair_id":"P2"},
        {"view_code":"LAT", "pair_valid":True, "pair_id":"P2"},
        {"view_code":"AP", "pair_valid":False, "pair_id":""},
    ]
    ds = SimpleNamespace(labels=labels, vgra_metadata=meta,
                         vgra_units=[(0,1), (2,3), (4,)], vgra_split="train")
    counts, weights = train_state_weight_binding(ds, 9, split="train")
    assert torch.all(counts.sum(1) == 2)
    assert counts[3, 3].item() == 1  # fracture shared on P1
    assert counts[3, 0].item() == 1  # fracture absent on P2
    assert counts[5, 1].item() == 1
    assert counts[2, 1].item() == 1
    torch.testing.assert_close(weights.mean(1), torch.ones(9))
    with pytest.raises(ValueError, match="B-TRAIN"):
        train_state_counts_from_dataset(ds, 9, split="val")
    ds.vgra_split = "val"
    with pytest.raises(ValueError, match="B-TRAIN"):
        train_state_counts_from_dataset(ds, 9, split="train")


def test_empty_detections_yield_all_neither_state():
    b = {"img": torch.zeros(2,3,8,8), "cls": torch.empty(0,1),
         "batch_idx": torch.empty(0),
         "vgra_pair_index": torch.tensor([1,0]),
         "vgra_pair_valid": torch.tensor([True,True]),
         "vgra_view_code": torch.tensor([1,2]),
         "vgra_pair_id": ("P1","P1")}
    y, _, _ = visibility_targets_from_batch(b, 9)
    assert torch.equal(y, torch.zeros((1,9), dtype=torch.long))


def test_negative_beta_can_invert_semantic_gate():
    # Document this as an intentionally OPEN design-verification gate.
    beta = 2 * torch.tanh(torch.tensor(-0.2))
    positive_shared_gate = torch.tensor(0.8)
    m = torch.tensor(0.3)
    assert (beta * positive_shared_gate * m).item() < 0
