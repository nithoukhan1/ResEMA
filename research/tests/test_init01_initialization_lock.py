from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LOCK = (
    ROOT
    / "research/01_provenance/INIT01_INITIALIZATION_LOCK.json"
)

EXPERIMENTS = (
    ROOT
    / "research/05_experiments/EXPERIMENTS.csv"
)

CHECKPOINT_SHA256 = (
    "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
)


def load_lock() -> dict:
    return json.loads(
        LOCK.read_text(
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


def test_init01_official_checkpoint_identity():
    lock = load_lock()

    assert lock["schema_version"] == "INIT01-initialization-lock-v1.0"
    assert lock["status"] == "COMPLETE"
    assert lock["repository_parent_commit"] == (
        "60abdfce6ede6a954ffcd804daa08c224856481a"
    )
    assert lock["training_authorization"] == (
        "LOCKED_PENDING_TRAIN01_RESUME01"
    )

    checkpoint = lock["official_checkpoint"]

    assert checkpoint["source_repository"] == "ultralytics/assets"
    assert checkpoint["release_tag"] == "v8.4.0"
    assert checkpoint["asset_id"] == 340060754
    assert checkpoint["bytes"] == 19313732
    assert checkpoint["sha256"] == CHECKPOINT_SHA256
    assert checkpoint["external_artifact_only"]

    evidence = lock["runtime_evidence"]

    assert evidence["report_sha256"] == (
        "d233133814a0c1ad145b76a8f33c85a36349c4ff915d632aab3d8af60329e181"
    )
    assert evidence["offline_git_bundle"]["sha256"] == (
        "77f9f9cd7380606a72a8c1855bcc2bbe3252d168792b8693531cb6edda8d9f83"
    )


def test_init01_exact_runtime_and_model_identity():
    lock = load_lock()

    runtime = lock["exact_fork_runtime"]

    assert runtime["ultralytics_version"] == "8.4.7"
    assert runtime["python"] == "3.12.13"
    assert runtime["torch"] == "2.10.0+cu128"
    assert runtime["gpus"] == ["Tesla T4", "Tesla T4"]

    model = lock["checkpoint_model_identity"]

    assert model["task"] == "detect"
    assert model["model_class"] == "DetectionModel"
    assert model["scale"] == "s"
    assert model["source_classes"] == 80
    assert model["parameters"] == 9458752
    assert model["state_dict_items"] == 499
    assert model["stride"] == [8.0, 16.0, 32.0]

    source_args = model["source_checkpoint_train_args_provenance_only"]
    assert source_args["not_our_baseline_training_recipe"]


def test_init01_nine_class_transfer_contract():
    contract = load_lock()["nine_class_initialization_contract"]

    assert contract["contract_id"] == (
        "INIT01:YOLO11S-9C-PTVSCR-S42:v1"
    )
    assert contract["target_scale"] == "s"
    assert contract["target_classes"] == 9
    assert contract["seed"] == 42
    assert contract["target_parameters"] == 9431275
    assert contract["scratch_parameters"] == 9431275
    assert contract["target_state_dict_items"] == 499

    assert contract["same_architecture_pretrained_and_scratch"]
    assert contract["same_seeded_initial_state_before_transfer"]
    assert contract["transferable_state_items"] == 493
    assert contract["nontransferable_target_state_items"] == 6
    assert contract["nontransferable_source_state_items"] == 6
    assert contract["all_transferable_tensors_loaded_exactly"]
    assert contract[
        "all_nontransferable_target_tensors_"
        "preserved_from_seeded_initialization"
    ]

    assert contract["nontransferable_target_keys"] == [
        "model.23.cv3.0.2.weight",
        "model.23.cv3.0.2.bias",
        "model.23.cv3.1.2.weight",
        "model.23.cv3.1.2.bias",
        "model.23.cv3.2.2.weight",
        "model.23.cv3.2.2.bias",
    ]


def test_init01_experiment_bindings_and_training_lock():
    rows = load_experiments()

    for row in rows:
        assert row["status"] == "NOT_STARTED"
        assert not row["source_commit"]

        if row["initialization"] == "pretrained":
            assert row["init_checkpoint_sha256"] == CHECKPOINT_SHA256
            assert row["notes"] == "INIT01:YOLO11S-9C-PTVSCR-S42:v1"

        elif row["initialization"] == "scratch":
            assert not row["init_checkpoint_sha256"]
            assert row["notes"] == "INIT01:scratch-seed42-no-checkpoint"

        else:
            raise AssertionError(
                f"Unexpected initialization: {row['initialization']}"
            )

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

    assert "`TRAIN-01` - implement the reusable governed baseline trainer" in current
    assert "Training remains locked until both `TRAIN-01` and `RESUME-01`" in current

    assert "## 2026-09-23 - INIT-01 initialization contract frozen" in decisions
    assert "The project advances to `TRAIN-01`." in decisions
