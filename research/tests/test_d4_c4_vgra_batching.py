from __future__ import annotations

from types import SimpleNamespace

import torch

from ultralytics.data.vgra import (
    PAIR_SAFE_DISABLED_AUGMENTATIONS,
    VGRAPairBatchSampler,
    VGRAYOLODataset,
    bind_vgra_assignments,
    build_vgra_dataloader,
    pair_safe_hyp,
)


def _row(
    stem,
    pair_id="",
    structural_role="SINGLE",
    operational_role="SINGLE_CANDIDATE",
    view_code="AP",
    split="train",
):
    return {
        "split": split,
        "filestem": stem,
        "pair_id": pair_id,
        "structural_role": structural_role,
        "operational_role": operational_role,
        "view_code": view_code,
    }


def test_pair_safe_hyp_disables_only_cross_study_composition():
    hyp = SimpleNamespace(
        mosaic=1.0,
        mixup=0.25,
        cutmix=0.2,
        copy_paste=0.3,
        fliplr=0.5,
        hsv_h=0.015,
    )
    safe = pair_safe_hyp(hyp)
    assert all(getattr(safe, k) == 0.0 for k in PAIR_SAFE_DISABLED_AUGMENTATIONS)
    assert safe.fliplr == 0.5
    assert safe.hsv_h == 0.015
    assert hyp.mosaic == 1.0  # original object must remain unchanged


def test_bind_assignments_builds_exact_pair_and_single_units():
    rows = [
        _row("a_ap", "PAIR-A", "PAIRED", "PAIR_CANDIDATE", "AP"),
        _row("a_lat", "PAIR-A", "PAIRED", "PAIR_CANDIDATE", "LAT"),
        _row("single", "", "SINGLE", "SINGLE_CANDIDATE", "AP"),
    ]
    meta, units = bind_vgra_assignments(
        ["root/a_lat.png", "root/single.png", "root/a_ap.png"], rows, "train"
    )
    assert len(meta) == 3
    assert sorted(len(u) for u in units) == [1, 2]
    pair = next(u for u in units if len(u) == 2)
    assert {meta[i]["view_code"] for i in pair} == {"AP", "LAT"}


def test_val_unreadable_companion_fallback_is_single_unit():
    rows = [
        _row("good", "PAIR-X", "PAIRED", "SINGLE_FALLBACK_COMPANION_UNREADABLE", "AP", "val"),
        _row("bad", "PAIR-X", "PAIRED", "EXCLUDED_UNREADABLE", "LAT", "val"),
    ]
    meta, units = bind_vgra_assignments(["root/good.png"], rows, "val")
    assert len(units) == 1 and len(units[0]) == 1
    assert meta[0]["pair_valid"] is False


def test_pair_batch_sampler_never_splits_pairs_and_covers_every_index():
    units = [(0, 1), (2, 3), (4,), (5,), (6, 7), (8,), (9,)]
    sampler = VGRAPairBatchSampler(units, batch_size=5, shuffle=True, seed=42)
    batches = list(iter(sampler))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(10))
    assert len(flat) == len(set(flat))

    batch_of = {}
    for bi, batch in enumerate(batches):
        for idx in batch:
            batch_of[idx] = bi
    for a, b in [(0, 1), (2, 3), (6, 7)]:
        assert batch_of[a] == batch_of[b]


def test_pair_batch_sampler_is_seed_deterministic_across_instances_and_epochs():
    units = [(0, 1), (2, 3), (4, 5), (6,), (7,), (8,), (9,), (10,), (11,)]
    a = VGRAPairBatchSampler(units, batch_size=5, shuffle=True, seed=123)
    b = VGRAPairBatchSampler(units, batch_size=5, shuffle=True, seed=123)
    a0, b0 = list(iter(a)), list(iter(b))
    a1, b1 = list(iter(a)), list(iter(b))
    assert a0 == b0
    assert a1 == b1
    assert a0 != a1


def _sample(stem, pair_id, valid, view_code):
    return {
        "batch_idx": torch.zeros(1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "cls": torch.tensor([[3.0]], dtype=torch.float32),
        "img": torch.zeros(3, 16, 16, dtype=torch.uint8),
        "im_file": f"/tmp/{stem}.png",
        "ori_shape": (16, 16),
        "ratio_pad": (1.0, 1.0),
        "resized_shape": (16, 16),
        "vgra_filestem": stem,
        "vgra_operational_role": "PAIR_CANDIDATE" if valid else "SINGLE_CANDIDATE",
        "vgra_pair_id": pair_id,
        "vgra_pair_valid": valid,
        "vgra_view_code": view_code,
    }


def test_vgra_collate_emits_companion_map_and_preserves_yolo_batch_idx():
    batch = [
        _sample("ap", "P1", True, 1),
        _sample("lat", "P1", True, 2),
        _sample("single", "", False, 1),
    ]
    out = VGRAYOLODataset.collate_fn(batch)
    assert out["img"].shape == (3, 3, 16, 16)
    assert torch.equal(out["vgra_pair_index"], torch.tensor([1, 0, -1]))
    assert torch.equal(out["vgra_pair_valid"], torch.tensor([True, True, False]))
    assert torch.equal(out["batch_idx"], torch.tensor([0.0, 1.0, 2.0]))


def test_vgra_collate_fails_closed_if_valid_pair_is_split():
    batch = [_sample("ap", "P1", True, 1), _sample("single", "", False, 1)]
    try:
        VGRAYOLODataset.collate_fn(batch)
    except ValueError as exc:
        assert "split across batches" in str(exc)
    else:
        raise AssertionError("expected split valid pair to fail closed")


def test_vgra_dataloader_rejects_ddp_before_using_dataset():
    try:
        build_vgra_dataloader(
            dataset=None,
            batch=16,
            workers=0,
            shuffle=True,
            seed=42,
            rank=0,
        )
    except NotImplementedError as exc:
        assert "single-process" in str(exc)
    else:
        raise AssertionError("VGRA V1 must reject DDP in C4")
