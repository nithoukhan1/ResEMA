from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "research/04_data/manifests/"
      "DATA01_ACTIVE_DATASET_BINDINGS.json"
)

EXPERIMENTS = (
    ROOT
    / "research/05_experiments/EXPERIMENTS.csv"
)


def load_manifest() -> dict:
    return json.loads(
        MANIFEST.read_text(
            encoding="utf-8",
            errors="strict",
        )
    )


def load_experiments() -> list[dict[str, str]]:
    with EXPERIMENTS.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def test_data01_manifest_contract():
    manifest = load_manifest()

    assert manifest["schema_version"] == (
        "DATA01-active-dataset-bindings-v1.0"
    )
    assert manifest["created_date"] == "2026-09-22"
    assert manifest["status"] == "COMPLETE"
    assert manifest["repository_parent_commit"] == (
        "76980ecd05416361cd4f133f869981d3af1bef8b"
    )
    assert manifest["training_authorization"] == (
        "LOCKED_PENDING_INIT01_TRAIN01_RESUME01"
    )

    assert manifest["binding_ids"] == {
        "A_AUG": "DATA01:A-AUG:v1",
        "B_ORG": "DATA01:B-ORG:v1",
        "B_AUG": "DATA01:B-AUG-HIST:v1",
    }


def test_data01_a_aug_binding():
    binding = load_manifest()["bindings"]["A_AUG"]

    assert binding["kaggle_dataset_ref"] == (
        "nithoukhan/grazpedwri-dx-aug"
    )

    assert binding["runtime_counts"] == {
        "train_aug_images": 28408,
        "train_aug_labels": 28408,
        "original_component": 14204,
        "augmented_component": 14204,
        "other_component": 0,
        "validation_images": 4094,
        "validation_labels": 4094,
        "test_images": 2029,
        "test_labels": 2029,
    }

    assert binding["naming_contract"]["original_prefix"] == "orig_"
    assert binding["naming_contract"]["augmented_prefix"] == "aug_"
    assert binding["naming_contract"]["original_bases_equal_augmented_bases"]
    assert binding["naming_contract"]["both_base_sets_equal_train_data_csv"]

    assert binding["patients"] == {
        "train": 4263,
        "validation": 1218,
        "test": 610,
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
    }

    assert binding["train_local_label_pair_integrity"]["pass"]

    firewall = binding["cross_split_firewall"]
    assert firewall["A_train_overlap_B_train_images"] == 9852
    assert firewall["A_train_overlap_B_validation_images"] == 2171
    assert firewall["A_train_overlap_B_test_images"] == 2181


def test_data01_b_org_binding():
    binding = load_manifest()["bindings"]["B_ORG"]

    assert binding["kaggle_dataset_ref"] == (
        "utopianstar/grazpedwri-dx-split-b"
    )

    assert binding["runtime_counts"] == {
        "train_images": 14227,
        "train_labels": 14227,
        "validation_images": 3050,
        "validation_labels": 3050,
        "test_images": 3050,
        "test_labels": 3050,
    }

    assert binding["patients"] == {
        "train": 4264,
        "validation": 914,
        "test": 913,
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
    }

    hashes = binding["membership_sha256_final_newline"]

    assert hashes["train"] == (
        "4ab032d669ee40daeaa6d37680def3ba4"
        "aad4dc64b9d0f16b7b658be9e48812b"
    )

    assert hashes["validation"] == (
        "f36040d4a798cbda113909907ba21e3da"
        "c1bb91b268124ec51c4fb2899e4a816"
    )

    assert hashes["test"] == (
        "edbcaa1393f874cd27903e3b5ffec419f"
        "3be3f821ef32fab1d2b99a8bd4e6128"
    )


def test_data01_b_aug_historical_binding():
    binding = load_manifest()["bindings"]["B_AUG"]

    assert binding["binding_id"] == (
        "DATA01:B-AUG-HIST:v1"
    )

    assert binding["runtime_counts"]["train_aug_historical_images"] == 28454
    assert binding["runtime_counts"]["original_component"] == 14227
    assert binding["runtime_counts"]["augmented_component"] == 14227

    integrity = binding["membership_integrity"]

    assert integrity["original_component_equals_B_ORG_train"]
    assert integrity["augmented_bases_equal_B_ORG_train"]
    assert integrity["validation_base_overlap"] == 0
    assert integrity["test_base_overlap"] == 0
    assert integrity["label_pairs_checked"] == 14227
    assert integrity["label_byte_mismatches"] == 0
    assert integrity["pass"]

    assert binding["provenance_caveat"]["status"] == (
        "HISTORICAL_COMPARATOR_ONLY"
    )


def test_data01_runtime_evidence_and_containment():
    manifest = load_manifest()
    evidence = manifest["runtime_evidence"]

    assert evidence["DATA01_READONLY_DISCOVERY.json"]["sha256"] == (
        "9e18e9b4ee70f48c3116fd674318358f"
        "41ef1667f600e923b10ecb6278704dc2"
    )

    assert evidence["DATA01_BINDING_VERIFICATION_R2.json"]["sha256"] == (
        "414b59ba9a06e3ea43cd939ac360a302"
        "3804bbaf5fff48abece34163c0292057"
    )

    assert evidence[
        "DATA01_SPLIT_A_PROVENANCE_DIAGNOSTIC_R3.json"
    ]["sha256"] == (
        "661df71eda58b09f26b2d352e8b60bd2"
        "03b9f2b147f08aa7a8a1ca12d11c4daf"
    )

    assert evidence[
        "DATA01_SPLIT_A_FINAL_BINDING_R4.json"
    ]["sha256"] == (
        "553bfb7f99109f2f09bd88b155806b86"
        "a759b2afa68fa4254f6c4af3a2552a77"
    )

    incident = manifest["contained_incident"]

    assert incident["status"] == "SUPERSEDED"
    assert incident["inferred_split_b_test_label_files_opened"] == 2181
    assert not incident["model_predictions_or_test_metrics_accessed"]
    assert not incident["used_for_development_or_model_selection"]
    assert not incident["authoritative_evidence"]


def test_data01_experiment_rows_are_bound():
    rows = load_experiments()

    expected = {
        "BASE-B-ORG-PT-S42": "DATA01:B-ORG:v1",
        "BASE-B-ORG-SCR-S42": "DATA01:B-ORG:v1",
        "BASE-B-AUG-PT-S42": "DATA01:B-AUG-HIST:v1",
        "BASE-B-AUG-SCR-S42": "DATA01:B-AUG-HIST:v1",
        "BASE-A-AUG-PT-S42": "DATA01:A-AUG:v1",
        "BASE-A-AUG-SCR-S42": "DATA01:A-AUG:v1",
    }

    observed = {
        row["experiment_id"]: row["data_binding"]
        for row in rows
    }

    for experiment_id, binding in expected.items():
        assert observed[experiment_id] == binding

    for row in rows:
        assert row["status"] == "NOT_STARTED"
        assert not row["source_commit"]
        assert not row["init_checkpoint_sha256"]


def test_data01_test_firewall_and_training_lock():
    manifest = load_manifest()

    policy = manifest["test_policy"]

    assert policy["development_use"] == "FORBIDDEN"
    assert policy["split_b_primary_development_environment"]
    assert policy["A_AUG_firewalled_from_B_development"]

    scientific = manifest["scientific_state"]

    assert not scientific["model_training_performed_during_DATA01"]
    assert not scientific["model_architecture_changed_during_DATA01"]
    assert not scientific["test_model_outcomes_accessed_during_DATA01"]
    assert not scientific["baseline_experiments_authorized_to_start"]


def test_data01_remote_closure_advances_to_init01():
    current = (
        ROOT / "research/CURRENT.md"
    ).read_text(
        encoding="utf-8",
        errors="strict",
    )

    decisions = (
        ROOT / "research/DECISIONS.md"
    ).read_text(
        encoding="utf-8",
        errors="strict",
    )

    assert "DATA-01 is complete and remotely CI-verified." in current
    assert "`INIT-01` - freeze the exact official YOLO11s pretrained" in current
    assert "4473ff57126e2427c6a6e6e5f24c40528f76e5f7" in current
    assert "35711226411" in current

    assert "## 2026-09-22 - DATA-01 remotely closed" in decisions
    assert "4473ff57126e2427c6a6e6e5f24c40528f76e5f7" in decisions
    assert "35711226411" in decisions
    assert "The project advances to `INIT-01`." in decisions
