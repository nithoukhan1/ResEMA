from __future__ import annotations

import importlib.util
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "d4_vgra_pair_manifest.py"
SPEC = importlib.util.spec_from_file_location("d4_vgra_pair_manifest", TOOL)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def _row(stem, patient, study, lat, projection, split="train", timehash="100"):
    return {
        "filestem": stem,
        "patient_id": str(patient),
        "study_number": str(study),
        "timehash": str(timehash),
        "laterality": lat,
        "projection": str(projection),
        "split": split,
    }


def test_exact_pair_and_single_accounting():
    rows = [
        _row("a_ap", 1, 1, "R", 1),
        _row("a_lat", 1, 1, "R", 2),
        _row("b_ap", 2, 1, "L", 1),
    ]
    pairs, assignments, summary = mod.build_manifests(rows, "train")
    assert len(pairs) == 1
    assert len(assignments) == 3
    assert summary["exact_pair_groups"] == 1
    assert summary["structurally_paired_images"] == 2
    assert summary["structural_single_images"] == 1
    assert summary["all_images_accounted_exactly_once"] is True


def test_nonexact_ap_lat_group_is_not_forced_into_pair():
    rows = [
        _row("x_ap", 1, 1, "R", 1),
        _row("x_lat", 1, 1, "R", 2),
        _row("x_obl", 1, 1, "R", 3),
    ]
    pairs, assignments, summary = mod.build_manifests(rows, "train")
    assert pairs == []
    assert summary["pair_capable_groups"] == 1
    assert summary["ambiguous_ap_lat_groups"] == 1
    assert all(r["structural_role"] == "SINGLE" for r in assignments)
    assert all(r["single_reason"] == "AMBIGUOUS_NONEXACT_AP_LAT_GROUP" for r in assignments)


def test_known_unreadable_val_member_degrades_pair_to_single_fallback():
    bad = "1502_0635264266_05_WRI-R2_M015"
    rows = [
        _row("good_ap", 1502, 5, "R", 1, split="val"),
        _row(bad, 1502, 5, "R", 2, split="val"),
    ]
    pairs, assignments, summary = mod.build_manifests(
        rows, "val", known_unreadable={bad}
    )
    assert len(pairs) == 1
    assert pairs[0]["operational_pair_status"] == "COMPANION_UNREADABLE_FALLBACK"
    by_stem = {r["filestem"]: r for r in assignments}
    assert by_stem[bad]["operational_role"] == "EXCLUDED_UNREADABLE"
    assert by_stem["good_ap"]["operational_role"] == "SINGLE_FALLBACK_COMPANION_UNREADABLE"
    assert summary["operational_excluded_unreadable"] == 1
    assert summary["operational_single_fallback_images"] == 1


def test_pair_id_is_deterministic():
    rows = [
        _row("lat", 9, 2, "L", 2),
        _row("ap", 9, 2, "L", 1),
    ]
    p1, _, _ = mod.build_manifests(rows, "train")
    p2, _, _ = mod.build_manifests(list(reversed(rows)), "train")
    assert p1 == p2
    assert p1[0]["pair_id"] == "TRAIN-P0009-S02-L"


def test_visibility_schema_is_detection_label_based_and_test_firewalled():
    schema = mod.visibility_schema()
    assert schema["target_formula"] == "state = ap_present + 2 * lat_present"
    assert schema["metadata_columns_are_not_detection_targets"] is True
    assert schema["actual_label_read_in_d4_c2"] is False
    assert schema["training_target_source"] == "B-TRAIN YOLO label files only"
    assert "forbidden" in schema["test_policy"].lower()
