from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"

ALLOWED_STATUS = {
    "NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE",
    "REJECTED", "SUPERSEDED", "LOCKED"
}
ALLOWED_PATH_STATUS = {"VERIFIED", "VERIFIED_DIRECTORY", "PATH_NOT_RECONSTRUCTED"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

def read_csv(rel):
    with (RESEARCH / rel).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def test_required_history_counts():
    experiments = read_csv("02_history/EXPERIMENT_REGISTRY.csv")
    results = read_csv("02_history/RESULTS_MASTER.csv")
    literature = read_csv("03_literature/Detection-Grazpedwri_RAW.csv")

    assert len(experiments) >= 10
    assert len(results) >= 10
    assert len(literature) >= 27

def test_experiment_ids_are_unique():
    rows = read_csv("02_history/EXPERIMENT_REGISTRY.csv")
    ids = [r["experiment_id"] for r in rows]
    assert len(ids) == len(set(ids))

def test_results_ids_exist_in_experiment_registry():
    exp_ids = {r["experiment_id"] for r in read_csv("02_history/EXPERIMENT_REGISTRY.csv")}
    for row in read_csv("02_history/RESULTS_MASTER.csv"):
        assert row["experiment_id"] in exp_ids

def test_project_tracker_status_vocabulary():
    for row in read_csv("02_history/PROJECT_TRACKER.csv"):
        assert row["status"] in ALLOWED_STATUS

def test_artifact_registry_hashes_and_statuses():
    rows = read_csv("02_history/ARTIFACT_REGISTRY.csv")
    assert len(rows) >= 8
    for row in rows:
        assert row["path_status"] in ALLOWED_PATH_STATUS
        assert HEX64.fullmatch(row["sha256"]), row

def test_no_large_model_artifacts_in_research_tree():
    forbidden = {".pt", ".pth", ".onnx", ".engine", ".tflite"}
    bad = [p for p in RESEARCH.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    assert not bad, bad

def test_test_policy_is_documented():
    text = (RESEARCH / "01_provenance/DATASET_PROVENANCE.md").read_text(encoding="utf-8")
    assert "Test remains sealed" in text or "test remains sealed" in text.lower()

def test_runtime_manifest_template_is_valid_json():
    path = RESEARCH / "tools/ARTIFACT_MANIFEST_TEMPLATE.json"
    json.loads(path.read_text(encoding="utf-8"))

def test_research_text_files_are_utf8():
    """All publication-governed text files must be strict UTF-8."""
    extensions = {
        ".md",
        ".csv",
        ".py",
        ".yml",
        ".yaml",
        ".json",
        ".txt",
        ".sh",
    }

    paths = [
        p
        for p in RESEARCH.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    workflows = ROOT / ".github" / "workflows"

    if workflows.exists():
        paths.extend(
            p
            for p in workflows.glob("research-*.yml")
            if p.is_file()
        )
        paths.extend(
            p
            for p in workflows.glob("research-*.yaml")
            if p.is_file()
        )

    bad = []

    for path in sorted(set(paths)):
        try:
            path.read_text(
                encoding="utf-8",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            bad.append(
                f"{path.relative_to(ROOT)}: {exc}"
            )

    assert not bad, (
        "Non-UTF-8 publication-governed text files:\n"
        + "\n".join(bad)
    )

def test_train01_contract_is_static_ci_verifiable():
    contract = json.loads(
        (
            RESEARCH
            / "05_experiments/TRAIN01_TRAINER_CONTRACT.json"
        ).read_text(
            encoding="utf-8",
            errors="strict",
        )
    )

    assert contract["schema_version"] == "TRAIN01-governed-trainer-v1.0"
    assert contract["status"] == "IMPLEMENTED_RESUME_AWARE"
    assert contract["training_authorization"] == (
        "LOCKED_UNTIL_EXPERIMENT_SOURCE_COMMIT_IS_BOUND"
    )

    firewall = contract["test_firewall"]
    assert firewall["runtime_yaml_contains_train"]
    assert firewall["runtime_yaml_contains_val"]
    assert not firewall["runtime_yaml_contains_test"]
    assert firewall["directory_discovery_prunes_test_paths"]


def test_train01_training_yaml_is_frozen_and_val_only():
    text = (
        RESEARCH
        / "05_experiments/TRAINING.yaml"
    ).read_text(
        encoding="utf-8",
        errors="strict",
    )

    required = [
        "schema_version: baseline_training_v2",
        "status: FROZEN_BASELINE_RECIPE_V2",
        "epochs: 100",
        "imgsz: 1024",
        "batch_global: 16",
        "optimizer: SGD",
        "seed: 42",
        "device: [0, 1]",
        "split: val",
        "test_key_in_runtime_yaml: forbidden",
        "exist_ok: false",
    ]

    for item in required:
        assert item in text

    assert "split: test" not in text


def test_train01_runtime_sources_exist_and_compile():
    sources = [
        ROOT / "ultralytics/research/baseline_trainer.py",
        RESEARCH / "runtime/baseline_runner.py",
        RESEARCH / "runtime/resume_runner.py",
    ]

    for path in sources:
        source = path.read_text(
            encoding="utf-8",
            errors="strict",
        )
        compile(source, str(path), "exec")


def test_train01_source_binding_is_atomic_and_fail_closed_when_blank():
    rows = read_csv("05_experiments/EXPERIMENTS.csv")
    assert len(rows) == 6
    assert all(row["status"] == "NOT_STARTED" for row in rows)

    bindings = {row["source_commit"].strip() for row in rows}
    assert bindings == {""} or (
        len(bindings) == 1
        and re.fullmatch(r"[0-9a-f]{40}", next(iter(bindings)))
    )

    runner = (
        RESEARCH
        / "runtime/baseline_runner.py"
    ).read_text(
        encoding="utf-8",
        errors="strict",
    )

    # The implementation-freeze commit is fail-closed while the binding is
    # blank. A later authorization commit may atomically bind the frozen SHA.
    assert "source_commit is blank." in runner
    assert "RESUME-01 closure must bind" in runner
    assert '"test_directory_scan_pruned": True' in runner

def test_resume01_matrix_binding_and_runtime_hash_guards_are_present():
    runner = (
        RESEARCH / "runtime/baseline_runner.py"
    ).read_text(encoding="utf-8", errors="strict")
    trainer = (
        ROOT / "ultralytics/research/baseline_trainer.py"
    ).read_text(encoding="utf-8", errors="strict")

    assert "def verify_experiment_matrix_binding" in runner
    assert "verify_experiment_matrix_binding(source_commit)" in runner
    assert "EXPERIMENTS.csv changed outside the permitted source_commit " in runner
    assert "binding: experiment=" in runner
    assert "RESEMA_PREFLIGHT_SHA256" in runner

    assert "RESEMA_RESUME_PREFLIGHT_SHA256" in trainer
    assert "Governed preflight manifest changed before trainer setup." in trainer
    assert "Governed runtime data YAML hash mismatch." in trainer
    assert "def _verify_preflight_before_dataset_access" in trainer
    assert "def get_dataset(self):" in trainer
    assert "changed before dataset access" in trainer
