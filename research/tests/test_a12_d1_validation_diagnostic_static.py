from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL = ROOT / "research/06_diagnostics/A12_D1_STANDARDIZED_VALIDATION_DIAGNOSTIC_PROTOCOL.md"
RUNNER = ROOT / "research/runtime/a12_d1_validation_diagnostic.py"
ARTIFACTS = ROOT / "research/01_provenance/ARTIFACTS.csv"
BASELINE_RUNTIME = ROOT / "research/05_experiments/BASELINE_FREEZE_01_PERCLASS_INPUTS.json"

EXPECTED_IDS = (
    "BASE-B-ORG-PT-S42",
    "BORG-PT-S42-SCCONV-EARLY-E100",
    "BORG-PT-S42-SCCONV-4STAGE-E100",
    "BORG-PT-S42-DYSAMPLE-E100",
    "BORG-PT-S42-CANONICAL-EMA-E100",
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100",
)

EXPECTED_CHECKPOINTS = {
    "BASE-B-ORG-PT-S42":
        "65e3c59901e0429f70a7b368dc29ebb698c427a799cfcf66354a882ec48bc4c7",
    "BORG-PT-S42-SCCONV-EARLY-E100":
        "620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1",
    "BORG-PT-S42-SCCONV-4STAGE-E100":
        "5485ec2fd9cc6421bdff7b4f0cd8b724db3ed8b540d4d11c77519cdbbcbfb7c4",
    "BORG-PT-S42-DYSAMPLE-E100":
        "61f3a99e4d29f20fddddd103ef3af049861d0b2278f384a044897be93a5fd556",
    "BORG-PT-S42-CANONICAL-EMA-E100":
        "f9d262c22c2bcb075cafe8eac368c0d4083a99f765f64e8ed6781c337ff0b6ec",
    "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100":
        "531c9927592781e7e19b6af1a35ce910113758d7ad5842677a5b7065487c897d",
}

EXPECTED_YAML_HASHES = {
    "ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml":
        "01caddd418e280684380c4629cb2b594a8ce628c37666216b37bef4226b11882",
    "ultralytics/cfg/models/11/yolo11s-tpsc-g4-v1.yaml":
        "4ede84aa4b026a3058526b4a8cde11d661c07e10c8cfce0936c0d2586e99f124",
    "ultralytics/cfg/models/11/yolo11s-dysample-v2.yaml":
        "629e3d3aee9177035f04a4f0bfc64293549794744e5a2500699e39bd4975678a",
    "ultralytics/cfg/models/11/yolo11s-tpema-head-v1.yaml":
        "04b5085c03172b3c143a9d97e585c32f3dd19588c0ba812fbcab64a2269d33a9",
    "ultralytics/cfg/models/11/yolo11s-scconv-early-canonical-ema-v1.yaml":
        "6f9547db340b77b3947698fcd911a70d3cde63328e1190751bc135c11443df14",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d1_files_parse_and_exist():
    assert PROTOCOL.is_file()
    assert RUNNER.is_file()
    ast.parse(RUNNER.read_text(encoding="utf-8"))


def test_d1_protocol_freezes_six_models_and_firewall():
    text = PROTOCOL.read_text(encoding="utf-8")
    for experiment_id in EXPECTED_IDS:
        assert experiment_id in text
    assert "Split-B test access: NONE" in text
    assert "A12-D1 must not:" in text
    assert "run model validation" in text
    assert "run model prediction" in text
    assert "READY_FOR_REVIEW_BEFORE_A12_D2" in text


def test_d1_runner_is_preflight_only():
    text = RUNNER.read_text(encoding="utf-8")
    assert ".train(" not in text
    assert "model.val(" not in text
    assert "model.predict(" not in text
    assert 'A12_D2_AUTHORIZED=FALSE' in text
    assert 'VALIDATION_INFERENCE_STARTED=FALSE' in text
    assert 'TEST_ACCESS=NONE' in text


def test_d1_runner_contains_exact_six_experiment_ids():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXPERIMENT_IDS":
                    found = tuple(ast.literal_eval(node.value))
    assert found == EXPECTED_IDS


def test_d1_selected_checkpoint_registry_matches_frozen_hashes():
    with ARTIFACTS.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for experiment_id, expected_sha in EXPECTED_CHECKPOINTS.items():
        matches = [
            row
            for row in rows
            if row["experiment_id"] == experiment_id
            and row["artifact_role"] == "selected_checkpoint"
            and row["logical_name"] == "best.pt"
            and row["status"] == "VERIFIED"
        ]
        assert len(matches) == 1
        assert matches[0]["sha256"] == expected_sha


def test_d1_reuses_frozen_baseline_validation_runtime():
    payload = json.loads(BASELINE_RUNTIME.read_text(encoding="utf-8"))
    runtime = payload["runtime_contract"]
    assert runtime["python"] == "3.12.13"
    assert runtime["torch"] == "2.10.0+cu128"
    assert runtime["ultralytics"] == "8.4.7"
    assert runtime["gpu_inventory"] == ["Tesla T4", "Tesla T4"]
    assert runtime["evaluation_device"] == 0
    assert runtime["imgsz"] == 1024
    assert runtime["batch"] == 16
    assert runtime["conf"] is None
    assert runtime["iou"] == 0.7
    assert runtime["half"] is False
    assert runtime["split"] == "val"


def test_d1_corrected_yaml_identities_are_frozen():
    for rel, expected_sha in EXPECTED_YAML_HASHES.items():
        path = ROOT / rel
        assert path.is_file()
        assert sha256(path) == expected_sha
