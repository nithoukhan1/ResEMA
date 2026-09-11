from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "research/04_data/manifests/"
      "D00_INPUT_PROVENANCE_MANIFEST.json"
)

SPLIT_DIR = (
    ROOT
    / "research/04_data/split_b"
)

EXPECTED_DATASET_SHA256 = (
    "55a53cbe9673e5f29ef0f2f6074ae791"
    "f1c27a3b48efadcd0b771225ff6379e8"
)

EXPECTED_ARCHIVE_SHA256 = {
    "GRAZPEDWRI_DX_SPLIT_B_FROZEN_V1.zip":
        "2269ae12be544c3835a63d383fd2c749"
        "d5ed3ef7f1bc9ec9596675c96a286610",

    "GRAZPEDWRI_DX_SPLIT_B_CANDIDATE_V2.zip":
        "6b9c0b06ae843cc446b5a61f294f523"
        "33b0a17dc7cba676cfa59c09f31bef82b",

    "GRAZPEDWRI_DX_UNSPLIT_EDA_V2_1.zip":
        "b4e165c62e51df5fee8ae80c6874b721"
        "863a0f00b2f5fa133ff2159dcd869dfc",

    "_audit_original_v1_1.zip":
        "05490d75b26f431e3b572614e3355a411"
        "681badeac38df380e027006f97c2a0c",

    "folder_structure.zip":
        "244c03c08cecac766a96efdd94cd46097"
        "cbc735fb75ee225ce5c42889f6041a5",
}

FORBIDDEN_PRE_P10 = {
    "split_B_test.csv",
    "split_B_test_patients.csv",
    "split_B_all_images.csv",
    "split_B_all_patients.csv",
}

EXPECTED_PHYSICAL_STEM_SHA256 = {
    "train":
        "4ab032d669ee40daeaa6d37680def3ba4"
        "aad4dc64b9d0f16b7b658be9e48812b",

    "val":
        "f36040d4a798cbda113909907ba21e3dac"
        "1bb91b268124ec51c4fb2899e4a816",

    "test":
        "edbcaa1393f874cd27903e3b5ffec419f3"
        "be3f821ef32fab1d2b99a8bd4e6128",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8",
            errors="strict",
        )
    )


def test_d00_input_manifest_is_frozen():
    manifest = load_manifest()

    assert (
        manifest["schema_version"]
        == "D00-input-provenance-v1.0"
    )

    assert (
        manifest["created_date"]
        == "2026-09-11"
    )

    assert (
        manifest["status"]
        == "FROZEN_FOR_P1_D00"
    )

    assert (
        manifest["dataset"]["dataset_csv"]["sha256"]
        == EXPECTED_DATASET_SHA256
    )

    assert (
        manifest["test_seal"]["status"]
        == "SEALED_UNTIL_P10"
    )


def test_d00_archive_fingerprints_are_authoritative():
    manifest = load_manifest()

    observed = {
        row["logical_path"]: row["sha256"]
        for row in manifest["source_archives"]
    }

    assert observed == EXPECTED_ARCHIVE_SHA256


def test_d00_repository_mirror_hashes_match_manifest():
    manifest = load_manifest()

    mirrors = manifest["committed_mirrors"]

    assert len(mirrors) == 18

    for row in mirrors:

        path = ROOT / row["repo_path"]

        assert path.is_file(), row

        assert (
            path.stat().st_size
            == row["repo_size_bytes"]
        ), row

        assert (
            sha256_file(path)
            == row["repo_sha256"]
        ), row

        raw = path.read_bytes()

        assert b"\r\n" not in raw, row
        assert b"\r" not in raw, row


def test_d00_test_membership_is_not_committed_pre_p10():
    present = {
        path.name
        for path in SPLIT_DIR.iterdir()
        if path.is_file()
    }

    leaked = (
        present
        & FORBIDDEN_PRE_P10
    )

    assert not leaked, sorted(leaked)

    manifest = load_manifest()

    external_only = set(
        manifest["test_seal"][
            "external_only_until_p10"
        ]
    )

    assert (
        FORBIDDEN_PRE_P10
        <= external_only
    )


def test_d00_runtime_package_is_bound_to_clean_split():
    manifest = load_manifest()

    runtime = manifest[
        "runtime_dataset_package"
    ]

    assert (
        runtime[
            "physical_membership_stem_sha256"
        ]
        == EXPECTED_PHYSICAL_STEM_SHA256
    )

    assert runtime[
        "physical_counts"
    ] == {
        "train": {
            "images": 14227,
            "labels": 14227,
        },
        "val": {
            "images": 3050,
            "labels": 3050,
        },
        "test": {
            "images": 3050,
            "labels": 3050,
        },
    }

    clean = runtime[
        "clean_dataset_yaml"
    ]

    assert (
        clean["train"]
        == "images/train"
    )

    assert (
        clean["val"]
        == "images/val"
    )

    assert (
        clean["test"]
        == "images/test"
    )


def test_d00_clean_yaml_does_not_select_historical_aug():
    path = (
        SPLIT_DIR
        / "split_B_original.yaml"
    )

    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert "train_aug_historical" not in text


def test_d00_historical_aug_config_is_non_authoritative():
    manifest = load_manifest()

    historical = manifest[
        "runtime_dataset_package"
    ][
        "historical_augmentation"
    ]

    assert (
        historical["status"]
        == "NON_AUTHORITATIVE_FOR_D00"
    )

    assert (
        historical["drift_status"]
        == "DETECTED; exact reason/time not reconstructed"
    )

    assert (
        historical[
            "historical_provenance_snapshot"
        ][
            "sha256"
        ]
        == "df8cfe5d70b9127dce2713aa02a0166ff7df13c61b4d67eee3f7c0934b687986"
    )

    assert (
        historical[
            "current_yaml"
        ][
            "sha256"
        ]
        == "d5e5336b401f88be6bfb8026e26e573738101462fc21d6b971a3d183819b7ff0"
    )

    assert not (
        SPLIT_DIR
        / "split_B_historical_aug.yaml"
    ).exists()


def test_d00_runtime_mount_must_be_verified_later():
    manifest = load_manifest()

    mount = manifest[
        "runtime_dataset_package"
    ][
        "kaggle_runtime_mount"
    ]

    assert mount.startswith(
        "UNVERIFIED_PRE_D00_A"
    )
